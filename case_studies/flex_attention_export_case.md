# 案例：Export 路径下 FlexAttention 的图模式落地与多设备验收

---

## 本文概括

本文记录一次围绕 **FlexAttention 经 `torch.export` 导出** 这一关键路径的工程实践。FlexAttention 允许用户用 Python 自定义 attention 的掩码逻辑，编译期内联进融合内核；当它再叠一层 export 做部署捕获时，会同时牵出导出期嵌套编译、图中高阶算子（HOP）形态、多设备验收断层三类工程问题。本案例以 PyTorch 源码契约为依据，将测试从 `@requires_gpu` + 硬编码 CUDA 改为 `instantiate_device_type_tests` + 注入 `device` + 运行时白名单 skip，并在钉扎的 CPU/NPU 环境上完成对照验收。最终结论：CPU 上 Flex 导出契约成立（告警 + HOP + 数值对齐），NPU 上 Flex 正确 skip（运行时白名单不含 npu），NPU 上 BlockMask 套件全过（对照证明框架入口正常）。

---

## 一、背景

### 1.1 两种执行模式

PyTorch 默认采用 **Eager 模式**：每个算子调用立即在当前设备上执行并返回结果，与普通 Python 语义一致。优点是调试直观、控制流灵活；缺点是每步调度都经过 Python 运行时，无法跨算子做融合优化，也难以将整图下沉到加速器后端。

**图模式** 则先将模型前向捕获为一张计算图（IR），再由编译器统一优化与代码生成。两个主要入口：

| 入口 | 定位 |
|------|------|
| `torch.compile` | 训练/推理加速：TorchDynamo 在字节码层捕获执行流，交由后端（通常为 Inductor）生成融合内核 |
| `torch.export` | 部署向捕获：在更严格的约束下（无 Python 副作用、有限控制流）生成可序列化的 `ExportedProgram`，供 AOTI 等运行时离线加载 |

两者的关键差异：`compile` 仍保留运行时 guard 与重编译能力，适合在线加速；`export` 追求一次性、可复现的稳定图，适合离线部署。

### 1.2 FlexAttention 为什么和图模式强相关

普通 `scaled_dot_product_attention` 是固定算法。**FlexAttention**（`torch.nn.attention.flex_attention`）允许用 Python 写 `score_mod`（改 attention 分数）和 `mask_mod`（控制可见性，常配合 `create_block_mask` 生成块稀疏 `BlockMask`）。性能关键在于：这些 Python 函数不能在每次推理时当纯 Python 跑，而应在编译期内联进融合 attention 核。

为此，PyTorch 把 FlexAttention 收成 **Higher-Order Operator（HOP，高阶算子）**：主图里留一个 `flex_attention` 节点，自定义函数作为子图挂在节点上。后端看到这个节点，知道「这是 FlexAttention，有专门的融合模板」，直接生成高效内核。如果 HOP 丢了，后端只能面对一堆零散的矩阵乘和加法，无法识别整体意图，性能差几个数量级，甚至因中间结果（整个 attention score 矩阵）过大而 OOM。

### 1.3 业务常见写法

```python
block_mask = create_block_mask(mask_fn, ..., device=x.device)
out = torch.compile(flex_attention)(q, k, v, block_mask=block_mask)
```

当再叠一层 `torch.export` 做部署捕获时，同时牵出三条工程线：导出期嵌套编译、图形态契约、多设备验收。

### 1.4 关键概念速查

| 概念 | 说明 |
|------|------|
| Dynamo | 在 Python 字节码层跟踪张量运算，生成 FX 图 |
| FX Graph | PyTorch 常见图表示，节点为 `call_function`/`call_module` 等 |
| FakeTensor | 导出/追踪时用「假张量」推形状与 dtype，不分配真实大显存 |
| HOP | 图里带子图的算子（如 `flex_attention`），后端可按模板特化 |
| privateuse1 | PyTorch 留给外部后端的设备槽位；`torch_npu` 把 NPU 挂在这里 |
| `instantiate_device_type_tests` | 测试框架按设备复制用例并注入 `device` 参数 |
| `HAS_GPU`/`@requires_gpu` | 历史上多表示 CUDA/XPU+Triton，**常常不含 NPU** |

---

## 二、遇到的问题与对应目标

### 2.1 问题一：导出期嵌套编译

**现象**：export 过程中又触发 `torch.compile` / 嵌套 FakeTensor，mask 闭包捕获了外层 fake 张量，导致导出报错或图不稳定。

**根因**：业务代码写了 `torch.compile(flex_attention)`，而 export 的目的是把模型冻结成一张可带走的图。如果此时再开一层编译，图会变，导出就失去了「稳定可复现」的意义。

### 2.2 问题二：图中 HOP 丢失

**现象**：导出图里没有 `higher_order.flex_attention` 节点，后端无法按 FlexAttention 模板特化；或测试用一份写死 `device(type='cuda')` 的超长 golden 字符串做门禁，换设备或小版本即碎。

**根因**：FlexAttention 的价值在于以 HOP 形态留在图里，后端才能识别整体意图并生成融合内核。如果导出路径把 HOP 展开成零散算子，或测试用设备相关 golden 做门禁，多设备验收就无法成立。

### 2.3 问题三：多设备验收断层

**现象**：CUDA 上绿、NPU 上 0 条用例；或 NPU 上红在 `_validate_device`（运行时白名单不含 npu）；两者混在一起，分不清是测试入口问题还是算子能力问题。

**根因**：测例用 `@requires_gpu`（`HAS_GPU` 不含 NPU）做门禁，NPU 连矩阵都进不了。即使进了矩阵，FlexAttention 运行时白名单当前只允许 cpu/cuda/xpu/hpu/mps，NPU 应明确 skip 而非报失败。

### 2.4 对应目标

| 问题 | 目标 |
|------|------|
| 导出期嵌套编译 | 验证 export 内 `torch.compile` 降级为恒等并产生告警 |
| 图中 HOP 丢失 | 验证导出图含 `higher_order.flex_attention`，且数值与 eager 对齐 |
| 多设备验收断层 | NPU 进入测试矩阵；Flex 按 runtime 白名单正确 skip；BlockMask 套件 NPU 全过作对照 |

---

## 三、环境依赖与搭建

### 3.1 软件栈版本

| 项 | 钉扎 |
|----|------|
| 硬件 | Ascend 910B，aarch64 |
| 系统侧 | NPU 驱动 + CANN toolkit（建议固定 Docker 内运行，宿主机可转发） |
| Python | 3.12（`torch_npu` `ci/build.sh` 支持 3.9–3.13，本案例取 3.12） |
| PyTorch（跑 NPU） | **2.13.x**（与当前 `torch_npu` `SUPPORTED_TORCH_VERSION` 对齐） |
| torch_npu | 与 2.13 配套源码安装 |
| Triton | 3.2.x（含 ascend 后端包） |
| 入口脚本 | `npu-pytorch-setup/run_test.sh`（`cpu`/`npu` 两种模式） |

原则：验证「NPU 能否跑」必须用加速库允许的框架大版本；业务 fork 可更新测例，但不要把更新的 2.14 `torch` 直接 `pip install -e` 进 `venv_npu`；上游测例里不写死 `import torch_npu._inductor`。

### 3.2 前置条件检查

在已安装 NPU 驱动和 CANN 的容器或宿主机上，先确认基础环境：

```bash
# 架构
uname -m
# 预期：aarch64

# CANN 环境变量
env | grep ASCEND
# 预期：ASCEND_HOME_PATH / ASCEND_TOOLKIT_HOME 等已设置

# NPU 设备可见
npu-smi info
# 预期：能看到 910B 设备名称和数量

# 编译工具链
python3 --version && git --version && gcc --version && cmake --version && ninja --version
```

若 CANN 环境变量未设置，先 source 对应 `set_env.sh`：

```bash
source /usr/local/Ascend/cann-9.1.0-beta.1/set_env.sh
```

### 3.3 环境搭建步骤

使用 `npu-pytorch-setup/setup_pytorch_envs.sh` 一键完成：源码拉取、子模块初始化、兼容性 patch、venv 创建、PyTorch 编译、torch_npu 编译安装。

```bash
# 1. 放置脚本目录（假设已 clone npu-pytorch-setup 仓库）
cd /home/y00839695/npu-pytorch-setup

# 2. 执行搭建脚本（首次搭建，全量 clone + 编译）
./setup_pytorch_envs.sh

# 可选：指定 PyTorch / torch_npu commit
./setup_pytorch_envs.sh \
  --pytorch-commit fad74248e78716152917a729adb2b44ba2bab16e \
  --torch-npu-commit 45fbeae1ad57e5321562b2ab3b18f9ae58312b43

# 可选：仅重新编译（不重新 clone，适用于改了 C++ 后重编）
./setup_pytorch_envs.sh --only-build
```

脚本执行流程：

```text
1. 系统检查     → 确认 aarch64、NPU 驱动、CANN
2. 网络检查     → 确认 GitHub / GitCode 可访问
3. 软件检查     → Python、git、gcc、cmake、ninja 版本；缺失时自动安装
4. 拉取源码     → 克隆 PyTorch / torch_npu 到指定 commit
5. 子模块       → 全量更新子模块并校验完整性
6. 应用 patch   → NamedTensorCompat.h 兼容性修复
7. 创建 venv    → venv_cpu / venv_npu
8. 安装依赖     → PyTorch build deps + CANN TBE deps
9. 编译 PyTorch → CPU-only develop 模式，两个 venv 各装一份
10. 编译 torch_npu → 生成 wheel 并安装到 venv_npu
11. 验证        → import torch / torch_npu，检查 NPU 可用
```

搭建完成后目录结构：

```text
npu-pytorch-setup/
├── pytorch/          # PyTorch 源码（pin 树，venv 实际链接的 torch）
├── torch_npu/        # torch_npu 源码
├── venv_cpu/         # 仅含 CPU torch 的虚拟环境
├── venv_npu/         # 含 torch + torch_npu 的虚拟环境
├── log/              # 编译日志
├── run_test.sh       # 统一测试入口
├── setup_pytorch_envs.sh
└── patches/
```

### 3.4 环境冒烟验证

```bash
# 在容器内或经 run_test.sh 转发的 NPU 环境中
python -c "import torch, torch_npu; print(torch.__version__, torch.npu.is_available(), torch.npu.device_count())"
# 预期示例：2.13.0a0+gitfad7424 True 8
```

### 3.5 日常使用约定

- **改测例代码**：在业务 fork（如 `/home/y00839695/pytorch`）中修改；运行时仍用 pin 树的 `torch`。
- **跑 NPU 测试**：始终走 `run_test.sh npu ...`（自动设 `PYTORCH_TESTING_DEVICE_ONLY_FOR=npu`，宿主机无 CANN 时自动 `docker exec` 进容器）。
- **跑 CPU 对照**：`run_test.sh cpu ...`。
- **禁止**：将更新的 2.14 fork `pip install -e` 进 `venv_npu`（会破坏 `torch_npu` 兼容性）。

---

## 四、方案分析

### 4.1 导出区域内 `torch.compile` 降级为恒等

```3322:3331:torch/__init__.py
    if torch.compiler.is_exporting():
        from torch._higher_order_ops.utils import _in_hop_compile

        if not _in_hop_compile():
            warnings.warn(
                "torch.compile is ignored when called inside torch.export region",
                stacklevel=2,
            )
            return model
```

**机制**：`torch.compile` 检测到当前处于 export 区域时，直接返回原模型（变恒等），并打告警。这样导出捕获的图只有一层，稳定可复现。

**工程含义**：业务代码可继续写 `torch.compile(flex_attention)`，无需为 export 单独改写。验收时用 `assertWarnsRegex` 锁住告警文案，即证明这条契约生效。

### 4.2 图模式走 HOP，而非全材料化

```2543:2557:torch/nn/attention/flex_attention.py
    if torch.compiler.is_dynamo_compiling():
        for x in [query, key, value]:
            torch._dynamo.mark_static(x, -3)
            torch._dynamo.mark_static(x, -1)
        out, lse, max_scores = flex_attention_hop(
            query, key, value, score_mod, block_mask.as_tuple(), scale, kernel_options,
        )
```

HOP 定义：

```96:127:torch/_higher_order_ops/flex_attention.py
class FlexAttentionHOP(HigherOrderOperator):
    def __init__(self) -> None:
        super().__init__("flex_attention", cacheable=True)
    ...
flex_attention = FlexAttentionHOP()
```

在 `ProxyTorchDispatchMode` 下，`trace_flex_attention` 会用 fake example 再 `make_fx` 出 `score_mod`/`mask_mod` 子图，挂到 root tracer，并在 FX 图上留下一个 `call_function: flex_attention` 节点。后端看到此节点即可按融合模板特化。

**验收口径**：导出图中存在 `torch.ops.higher_order.flex_attention`，且输出与 eager 对齐。不以设备相关 golden 字符串做门禁。

### 4.3 设备能力以 runtime 白名单为准，与测试入口分开

`flex_attention` 入口会 `_validate_device`，当前允许 `cpu/cuda/xpu/hpu/mps`，不含 npu。因此：

- **测试解耦**：NPU 应出现在 instantiate 矩阵里（入口正常）；
- **功能验收**：NPU 上 Flex 应明确 skip（运行时拒绝），而非 Error 或假绿。

这是「测试入口 / 算子运行时」两层分离的典型场景。

### 4.4 用 BlockMask 套件做对照

只测 `create_block_mask` / 闭包 pytree 的用例（`TestExperimentFlex`）不跑 Flex 本体，NPU 上可全过。与 Flex 本体 skip 形成对照，快速区分「框架入口问题」还是「算子未支持」。

### 4.5 不做的范围

- 不改 `common_device_type` 全局 Flex 注册表；
- 不动 `test_aot_inductor*.py`（等 `HAS_GPU`/`GPU_TYPE`/`copy_tests`）；
- 测例内不写死 `import torch_npu._inductor`；
- 不为多设备维护 CUDA/CPU 两套 IR golden。

---

## 五、实施步骤

### 步骤 1：改造 `test/export/test_export.py` — 拆出 `TestExportFlexAttention`

**改前**（原 `TestExport` 大类内）：

```python
    @requires_gpu
    def test_flex_attention_export(self):
        ...
        model = MixedFakeModeModel(use_inductor=False)
        x = torch.randn(2, 128, 64)
        ...
        self.assertExpectedInline(
            str(exported_mod.code).strip(),
            """\
def forward(self, x):
    ...
    arange = torch.ops.aten.arange.start(0, 2, device = device(type='cpu'), ...)
    ...
""",
        )
        exported_out = exported_mod(x)
        self.assertEqual(exported_out, eager_out)
```

问题：`@requires_gpu` 使 NPU 直接 skip（`HAS_GPU` 不含 NPU）；张量在 CPU 但闸卡 GPU；golden 写死 `device(type='cpu')`，换设备即碎。

**改后**：

```python
class TestExportFlexAttention(TestCase):
    def test_flex_attention_export(self, device):
        device_type = torch.device(device).type
        if device_type not in ("cpu", "cuda", "xpu", "hpu", "mps"):
            self.skipTest(f"FlexAttention runtime does not support {device_type}")

        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        class MixedFakeModeModel(torch.nn.Module):
            ...  # 模型定义不变

        model = MixedFakeModeModel(use_inductor=False).to(device)
        x = torch.randn(2, 128, 64, device=device)
        eager_out = model(x)
        model.use_inductor = True
        with self.assertWarnsRegex(
            UserWarning,
            "torch.compile is ignored when called inside torch.export region",
        ):
            exported_mod = torch.export.export(model, (x,), strict=False).module()
        has_flex = any(
            n.op == "call_function"
            and n.target is torch.ops.higher_order.flex_attention
            for n in exported_mod.graph.nodes
        )
        self.assertTrue(has_flex, "exported graph should retain flex_attention HOP")
        exported_out = exported_mod(x)
        self.assertEqual(exported_out, eager_out)


instantiate_device_type_tests(TestExportFlexAttention, globals())
```

**逐行对照**：

| 改前 | 改后 | 原因 |
|------|------|------|
| `@requires_gpu` | 删除 | `HAS_GPU` 不含 NPU，整条 skip |
| `def test_...(self):` | `def test_...(self, device):` | 接收 instantiate 注入的设备 |
| `model = MixedFakeModeModel(...)` | `...to(device)` | 模型跟设备 |
| `x = torch.randn(2, 128, 64)` | `..., device=device` | 输入跟设备 |
| 无 skip 闸 | `if device_type not in (...): self.skipTest(...)` | 对齐 `_validate_device` 白名单 |
| `self.assertExpectedInline(...)` | `has_flex = any(...); assertTrue; assertEqual` | golden 写死设备字符串，换设备即碎；改为 HOP 存在性 + 数值 |
| 无 instantiate | `instantiate_device_type_tests(...)` | 按 cpu/cuda/npu 等复制用例 |

**同时**：删除文件顶部不再使用的 `requires_gpu` import。

### 步骤 2：改造 `test/export/test_experimental.py` — 拆出 `TestExperimentFlex`

**改前**（原 `TestExperiment` 大类内，多条 BlockMask 用例）：

```python
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_export_blockmask(self):
        ...
        self._test_export_blockmask_with_mask_fn(make_mask_fn)

    def _test_export_blockmask_with_mask_fn(self, make_mask_fn):
        ...
        x = torch.randn(2, 128, device="cuda")
        ...
```

**改后**：将这些用例移到新类 `TestExperimentFlex`，签名加 `device`，去掉 `TEST_CUDA` skip，`device="cuda"` 改为 `device=device`：

```python
@unittest.skipIf(not torch._dynamo.is_dynamo_supported(), "dynamo isn't supported")
class TestExperimentFlex(TestCase):
    def _test_export_blockmask_with_mask_fn(self, make_mask_fn, device):
        ...
        x = torch.randn(2, 128, device=device)
        ...

    def test_export_blockmask(self, device):
        ...
        self._test_export_blockmask_with_mask_fn(make_mask_fn, device)

    # 其余 blockmask_closure_* 同理：加 device，device="cuda" 改为 device=device

    # 以下两条保持 CUDA-SM 门控，不改：
    @unittest.skipUnless(
        IS_FLEX_ATTENTION_CUDA_PLATFORM_SUPPORTED and not torch.version.hip,
        "Requires CUDA with SM >= 8.0, Triton, and not ROCm",
    )
    def test_aot_export_flex_attention_callable_mask_mod(self):
        ...  # 不加 device，保持原样

    @unittest.skipUnless(
        IS_FLEX_ATTENTION_CUDA_PLATFORM_SUPPORTED and not torch.version.hip,
        "Requires CUDA with SM >= 8.0, Triton, and not ROCm",
    )
    def test_aot_export_flex_attention_with_blockmask_placeholders(self):
        ...  # 不加 device，保持原样
```

**逐行对照**：

| 改前 | 改后 | 原因 |
|------|------|------|
| `@unittest.skipIf(not TEST_CUDA, ...)` | 删除 | NPU 不是 CUDA，但 BlockMask 不依赖 Flex runtime |
| `def test_...(self):` | `def test_...(self, device):` | 接收注入设备 |
| `x = torch.randn(..., device="cuda")` | `..., device=device` | 跟设备 |
| `_test_..._with_mask_fn(make_mask_fn)` | `...(make_mask_fn, device)` | 传递设备 |
| `test_aot_export_flex_attention_*` | **不动** | CUDA-SM 门控，刻意保留 |

**同时**：文件顶部去掉不再使用的 `TEST_CUDA` import；末尾加 `instantiate_device_type_tests(TestExperimentFlex, globals())`。

### 步骤 3：抽取共享 `ensure_triton` 到 `inductor_utils.py`

**改前**（`test_export.py` / `test_serialize.py` / `test_nativert.py` / `test_aoti_torchbind_constants.py` 各自重复定义）：

```python
def _ensure_triton_for_device(test_case, device):
    if device == "cpu":
        if not HAS_TRITON:
            test_case.skipTest("Requires triton, which is not available")
    else:
        if not HAS_GPU:
            test_case.skipTest("Requires GPU, which is not available")
        ensure_triton(test_case)
```

**改后**（`inductor_utils.py` 中统一一份）：

```python
def ensure_triton(test_case, device=None, *, required_on_cpu=True):
    if not HAS_TRITON:
        if required_on_cpu or device != "cpu":
            test_case.skipTest("Requires triton, which is not available")
        return
    if device == "cpu":
        return
    ensure_triton(test_case)
```

各测例文件改为 `from torch.testing._internal.inductor_utils import ensure_triton`，调用处 `_ensure_triton_for_device(self, device)` 改为 `ensure_triton(self, device)`。去掉四处重复定义。

### 步骤 4：刻意不做的事

- 不改 `common_device_type` 全局 Flex 注册表；
- 不动 `test_aot_inductor*.py`（仍等 `HAS_GPU`/`GPU_TYPE`/`copy_tests`）；
- 测例内不写死 `import torch_npu._inductor`；
- 不为多设备维护 CUDA/CPU 两套 IR golden。

---

## 六、验收和结果

### 6.1 运行命令

```bash
SETUP=/path/to/npu-pytorch-setup
FORK=/path/to/pytorch

$SETUP/run_test.sh cpu $FORK/test/export/test_export.py -v -k test_flex_attention_export
$SETUP/run_test.sh npu $FORK/test/export/test_export.py -v -k test_flex_attention_export
$SETUP/run_test.sh cpu $FORK/test/export/test_experimental.py -v -k TestExperimentFlex
$SETUP/run_test.sh npu $FORK/test/export/test_experimental.py -v -k TestExperimentFlex
```

`npu` 模式设 `PYTORCH_TESTING_DEVICE_ONLY_FOR=npu`；instantiate 后类名为 `*PRIVATEUSE1`，用例名带 `_npu_`，**不要**再加 `-k NPU`。

### 6.2 验收表

| 项 | 通过标准 | 实际结果 |
|----|----------|----------|
| CPU Flex export | OK：有 compile-ignored 告警、图含 flex HOP、数值对齐 | 通过 |
| NPU Flex export | skipped，原因含 runtime 不支持 npu | 通过（正确 skip） |
| CPU BlockMask 套件 | 全 OK | 通过 |
| NPU BlockMask 套件 | 全 OK | 通过 |

### 6.3 排障对照表

| 现象 | 优先怀疑 |
|------|----------|
| NPU 上 0 条用例 | `ONLY_FOR`/instantiate/入口脚本问题 |
| Flex NPU skip，BlockMask NPU OK | Flex runtime 白名单，非框架入口问题 |
| Flex CPU 无 HOP | 导出路径或断言写错，先修契约 |
| 告警未出现 | export 未真正进入，或 compile 未被调用 |

---

## 七、结论

1. **FlexAttention + Export 的工程核心**是：导出时 compile 降级为恒等（保证图只有一层、稳定可复现），图中保留 `higher_order.flex_attention`（后端才能按 HOP 做融合实现）。两条契约缺一不可，验收时分别用告警正则和 HOP 节点存在性锁定。

2. **多设备验收**必须拆开两层——测试矩阵能否注入 `npu`（入口），以及算子 runtime 是否允许 `npu`（白名单）。本案例中 NPU 上 Flex skip + BlockMask 全过，即一次干净的分层证明：框架入口正常，Flex 本体属算子未支持。

3. **环境钉扎**是复现的一部分。版本错位会把「契约问题」伪装成「NPU 问题」——例如 2.14 fork 引入新 API 导致 2.13 `torch_npu` import 失败，表面看像 NPU 不支持，实则是版本不一致。

可交接清单：概念速查表、钉扎版本表、改前改后逐行对照、固定验收命令、验收门禁表、排障对照表。
