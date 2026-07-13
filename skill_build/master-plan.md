# Export / AOTI 测试设备无关改造设计

本文覆盖下列 **Export / AOTI 测试文件** 的设备无关改造：改造手段（含原 `docs/graph-plan.md`）+ 耦合分类 + 逐文件改法。任务主体是改这些测试文件；基础设施（RFC、`common_device_type` Flex 注册、NpuRunner 等）标为依赖项，可并行推进但不替代本清单文件改动。

基础设施类改动（`GPU_TYPE` / `HAS_GPU` / `HAS_TRITON` / `requires_*`）依赖 RFC（Issue #189138）。按 RFC 定稿实现，不做临时替代。手段 7（Flex）依赖先做基础设施，再回改本清单中的 Flex 相关用例。

**范围：**

```
test/export/test_converter.py
test/export/test_cpp_serdes.py
test/export/test_db.py
test/export/test_draft_export.py
test/export/test_dynamic_shapes.py
test/export/test_experimental.py
test/export/test_export_opinfo.py
test/export/test_export.py
test/export/test_export_strict.py
test/export/test_export_training_ir_to_run_decomp.py
test/export/test_functionalized_assertions.py
test/export/test_hop.py
test/export/test_lift_unlift.py
test/export/test_nativert.py
test/export/test_package.py
test/export/test_passes.py
test/export/test_pass_infra.py
test/export/test_retraceability.py
test/export/test_schema.py
test/export/test_serdes.py
test/export/test_serialize.py
test/export/test_sparse.py
test/export/test_strict_export_v2.py
test/export/test_swap.py
test/export/test_tools.py
test/export/test_unflatten.py
test/export/test_unflatten_training_ir.py
test/export/test_upgrader.py
test/export/test_verifier.py
test/test_model_exports_to_core_aten.py
test/test_prims.py
test/export/testing.py
test/inductor/test_aoti_cross_compile_windows.py
test/inductor/test_aot_inductor.py
test/inductor/test_aot_inductor_arrayref.py
test/inductor/test_aot_inductor_custom_ops.py
test/inductor/test_aot_inductor_package.py
test/inductor/test_aot_inductor_utils.py
test/inductor/test_aoti_torchbind_constants.py
```

**原则：** 先看测试要验证什么。通用逻辑跟当前设备；后端专属能力保留专属入口，不强行泛化。

**判断顺序：**

1. 只测 Python / 图 / schema / 序列化 → 保持 CPU / 普通 `TestCase`。
2. 在加速器上跑 export / compile / AOTI，逻辑通用 → 用 `device_type` / `GPU_TYPE` / `self.device`，不写死 `"cuda"`。
3. 专属能力（CUDA Graph、FakeCuda、PTXAS/CUBIN、Windows 交叉编译等）→ 保留专属 class / skip。FlexAttention 走手段 7（先基础设施依赖，再改用例）。

**依赖项（可并行推进，不替代本清单测试文件改动）：**

| 依赖 | 挡住什么 | 说明 |
|------|----------|------|
| **RFC #189138** | 手段 2、3，以及手段 5/7 的正确门禁 | `GPU_TYPE` / `HAS_GPU` / `HAS_TRITON` / `requires_*` 按定稿实现 |
| **手段 7 基础设施** | `test_experimental.py` / `test_export.py` 的 Flex 段 | `common_device_type` Flex 注册表 + `skip_device_if`；后端 Flex 产品能力 |
| **`AOTIModelContainerRunnerNpu` + pybind** | 手段 6 在 NPU 上真正跑通 | 测试侧只改动态查找；类由后端提供 |
| **Inductor `register_backend_for_device` + Dynamo interface** | AOTI privateuse1 入口 | copy_tests 挂上后仍需后端可编译 |

---

## 改造手段

### 背景：`instantiate_device_type_tests` 是什么

PyTorch 测试里写「同一套逻辑、多种设备各跑一遍」的标准方式。你写一个**通用测试模板类**，框架按设备自动复制出多套可发现的 TestCase。

```python
# 你写的是模板（本身不会被直接跑）
class TestHOP(TestCase):
    def test_foo(self, device):
        x = torch.randn(2, 2, device=device)
        ...

# 框架展开：删掉模板名，生成 TestHOPCPU / TestHOPCUDA / ...
instantiate_device_type_tests(TestHOP, globals())
```

大致过程（`common_device_type.py`）：

1. 从 `get_desired_device_type_test_bases(...)` 取出要跑的设备基类（如 `CPUTestBase`、`CUDATestBase`、`PrivateUse1TestBase`）
2. 对每个基类生成 `TestHOP` + 设备名 的新类，并把每个 `test_*` 实例化成带 `device` 参数的用例
3. 原模板类从 `globals()` 里删掉，避免重复发现

常用参数：

| 参数 | 作用 |
|------|------|
| `only_for="cpu"` / `"cuda"` | 只生成这些设备 |
| `except_for=...` | 排除某些设备 |
| `allow_mps` / `allow_xpu` | 显式打开 MPS/XPU（默认不一定进） |

和手工 `@parameterized.expand([("cpu",), ("cuda",)])` 的差别：后者设备列表写死在用例里；前者跟**框架当前有哪些设备基类**走，新后端注册后有机会自动进来（再配合下面的 CI 收窄）。

本清单里：`test_hop.py`、`test_export_opinfo.py`、`test_prims.py` 等已在用。AOTI 主路径多用 `copy_tests`（另一套「模板复制到 CPU/GPU 类」的机制），不是这套。

---

### 1. PrivateUse1 纳入设备调度

上面这套调度的设备基类需覆盖 PrivateUse1。树内已有：

```python
# common_device_type.py
class PrivateUse1TestBase(DeviceTypeTestBase):
    device_type = "privateuse1"
    ...

if _is_privateuse1_backend_available():
    test_bases.append(PrivateUse1TestBase)
```

含义拆开看：

- **`PrivateUse1TestBase`**：给第三方加速器预留的测试基类（不是单独的 `NpuTestBase`）。父类 `DeviceTypeTestBase` 负责注入 `device`、setup 等公共逻辑；`device_type` 先标成槽位名 `"privateuse1"`，跑起来时 `setUpClass` 会改成后端 rename 后的名字（如 `"npu"`）。
- **`test_bases`**：`instantiate_device_type_tests` 会为哪些设备生成 TestCase 的名单。常见有 `CPUTestBase`、`CUDATestBase`；若 `_is_privateuse1_backend_available()` 为真（后端已 rename + 注册设备模块等），再把 `PrivateUse1TestBase` 加进去。
- **串起来**：对 `TestHOP` 做 `instantiate` 时，名单里有谁就生成谁，例如 `TestHOPCPU` / `TestHOPCUDA` / `TestHOPNPU`。每个 `test_*(self, device)` 里的 `device` 会变成对应设备字符串。

因此：后端注册 privateuse1 后，**没有** `only_for` 限制的文件会自动多出一套 PrivateUse1/NPU TestCase。不另造 `NpuTestBase` / `allow_npu`。

**CI 收窄设备**：`instantiate` 可能生成很多设备组合；CI 用环境变量限制「这次只跑谁」，避免全设备都跑：

```
PYTORCH_TESTING_DEVICE_ONLY_FOR=<backend>    # 只跑列出的设备，如 npu
# 或
PYTORCH_TESTING_DEVICE_FOR_CUSTOM=privateuse1  # 把 custom/privateuse1 纳入要跑的集合
```

本清单用法：`test_hop.py`、通用 `test_prims` 等已 `instantiate` 的文件，代码侧通常不用改，靠后端注册 + CI 收窄。

---

### 2. `GPU_TYPE` / `HAS_GPU` / Triton 能力泛化（依赖 RFC #189138）

用例大量使用：

```python
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
```

现状实现写死设备：

```python
# torch/_inductor/utils.py
GPU_TYPES = ["cuda", "mps", "xpu", "mtia"]

def get_gpu_type() -> str:
    avail_gpus = [x for x in GPU_TYPES if getattr(torch, x).is_available()]
    gpu_type = "cuda" if len(avail_gpus) == 0 else avail_gpus.pop()  # 无设备回退 cuda
    return gpu_type

# torch/testing/_internal/inductor_utils.py
HAS_CUDA_AND_TRITON = torch.cuda.is_available() and HAS_TRITON
HAS_XPU_AND_TRITON = torch.xpu.is_available() and HAS_TRITON
HAS_GPU = HAS_CUDA_AND_TRITON or HAS_XPU_AND_TRITON  # 无 privateuse1
```

RFC 目标：

- `get_gpu_type()` 走 `torch.accelerator` 或已注册 `device_interface`；无设备返回 `"cpu"`，禁止回退 `"cuda"`
- `torch.cuda.is_available()` / `requires_gpu_with_enough_memory` 等硬编码改用 `torch.accelerator`
- `HAS_TRITON` 经 device interface 探测，例如：

```python
@staticmethod
def is_triton_capable(device: torch.types.Device = None) -> bool:
    """设备是否具备 Triton 能力（即使 backend 尚未装好也可返回 False）。"""
    return False
```

统一入口示例：

```python
def get_accelerator_type(*, require_available: bool = False) -> str:
    acc = torch.accelerator.current_accelerator(check_available=require_available)
    return acc.type if acc is not None else "cpu"

requires_accelerator = unittest.skipUnless(
    torch.accelerator.is_available(), "requires accelerator"
)
```

`GPU_TYPE` 与上述语义对齐。本手段是后续硬编码替换、skip 改写、AOTI GPU 入口的前提。

---

### 3. `requires_cuda_and_triton` 等 skip 改写（依赖手段 2 / RFC）

现状：

```python
# triton_utils.py
requires_cuda_and_triton = unittest.skipUnless(
    HAS_CUDA_AND_TRITON, "requires cuda and triton"
)

# 用例
@requires_cuda_and_triton
def test_...(self): ...

@unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
def test_...(self): ...

@unittest.skipIf(not TEST_CUDA, "CUDA not available")
def test_...(self): ...
```

`HAS_CUDA_AND_TRITON = torch.cuda.is_available() and HAS_TRITON`，只认 CUDA。

改法：通用用例改为 `requires_gpu` / RFC 定名的 `requires_accelerator_and_triton`，或在 `HAS_TRITON` 已设备无关后直接依赖它。真 CUDA 专属继续保留 CUDA skip。

---

### 4. 补 `instantiate_device_type_tests`

仅有 `run_tests()`、确需多设备参数化的文件，补上：

```python
instantiate_device_type_tests(TestFoo, globals())
```

本清单多数 Export 文件是普通 `TestCase` / CPU 逻辑，**不强制全开**。只对已参数化或明确要多设备的文件使用（并依赖手段 1 的 PrivateUse1 调度）。

---

### 5. 替换 `"cuda"` 硬编码

现状：

```python
torch.randn(4, 4, 4, device="cuda")
model.cuda()
```

改法：文件顶部取当前设备（已有则复用）：

```python
device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
# 或
from torch.testing._internal.inductor_utils import GPU_TYPE

torch.randn(4, 4, 4, device=device_type)  # / GPU_TYPE / self.device
```

同步等 runtime 优先 `torch.accelerator`：

```python
torch.accelerator.synchronize()
torch.accelerator.device_count()
```

依赖手段 2，否则 `GPU_TYPE` 在纯 privateuse1 机器上仍可能错。

---

### 6. AOTI runner 动态加载

现状（`test/inductor/test_aot_inductor_utils.py`）else 默认 Cuda：

```python
if device == "cpu":
    return torch._C._aoti.AOTIModelContainerRunnerCpu(so_path, 1)
elif device == "xpu":
    return torch._C._aoti.AOTIModelContainerRunnerXpu(so_path, 1, device)
elif device == "mps":
    return torch._C._aoti.AOTIModelContainerRunnerMps(so_path, 1)
else:
    return torch._C._aoti.AOTIModelContainerRunnerCuda(so_path, 1, device)
```

改法：按设备名拼接类名动态加载：

```python
@staticmethod
def legacy_load_runner(device, so_path: str) -> "AOTIModelContainerRunner":
    if IS_FBCODE:
        # ... 保持 FBCODE 逻辑 ...
        pass

    runner_cls_name = f"AOTIModelContainerRunner{device.capitalize()}"
    runner_cls = getattr(torch._C._aoti, runner_cls_name, None)
    if runner_cls is None:
        raise RuntimeError(
            f"Unsupported device '{device}': expected {runner_cls_name}"
        )
    if device in ("cpu", "mps"):
        return runner_cls(so_path, 1)
    return runner_cls(so_path, 1, device)
```

找不到类则报错，禁止静默落 Cuda。

---

### 7. FlexAttention 专属泛化（纳入考虑；先做基础设施依赖）

**本清单中的落点：** `test_experimental.py`、`test_export.py`（及 `test_hop.py` 中 flex 相关 xfail）。AOTI 文件无 Flex。

**分两阶段，不要一上来改测试文件里的 Flex 段。**

#### 阶段 A — 基础设施依赖（不在「只改测试文件」的最小交付里，但要排期）

问题根源（`common_device_type.py`）：

- `IS_FLEX_ATTENTION_CPU/CUDA/XPU/MPS_PLATFORM_SUPPORTED` 分散定义，每加设备加变量
- 平台门禁与 skip 按设备各写一套

目标形态（与 graph-plan 方案7对齐，落地时注意能力探测不能过粗）：

```python
# common_device_type.py（依赖项伪代码）
_flex_attention_device_support = {
    "cpu": not IS_MACOS and torch.cpu._is_avx2_supported()
           and os.getenv("ATEN_CPU_CAPABILITY") != "default",
    "cuda": torch.cuda.is_available() and has_triton()
            and torch.cuda.get_device_capability() >= (8, 0),
    "xpu": torch.xpu.is_available() and has_triton(),
    "mps": torch.mps.is_available(),
}

def is_flex_attention_supported(device: str) -> bool:
    if device in _flex_attention_device_support:
        return _flex_attention_device_support[device]
    # privateuse1：须有真实能力探测，禁止仅 is_available() and has_triton()
    try:
        interface = get_interface_for_device(device)
        return interface.is_available() and interface_flex_capable(interface)
    except (RuntimeError, ImportError):
        return False

def skip_device_if(skip_condition: bool, device: str, reason: str):
    # 须保持 skipCUDAIf 等「仅匹配该 device_type 才 skip」的语义；
    # privateuse1 兜底不要误用成无条件的 unittest.skipIf
    ...
```

**依赖前提：**

1. 手段 2 / RFC：Triton、加速器探测语义正确
2. 后端：NPU（或目标设备）上 FlexAttention / BlockMask **产品能力**可用，否则测试基础设施改完也会红
3. skip 工厂与 `instantiate_device_type_tests` 的 `device` 注入语义对齐

#### 阶段 B — 本清单测试文件（基础设施就绪后）

**就绪前（当前默认）：** Flex 段保持 CUDA 专属，先改同文件非 Flex 部分。

```python
# 暂时保留
@unittest.skipUnless(
    IS_FLEX_ATTENTION_CUDA_PLATFORM_SUPPORTED and not torch.version.hip,
    "Requires CUDA with SM >= 8.0, Triton, and not ROCm",
)
def test_aot_export_flex_attention_...(self):
    model = FlexAttentionModel(...).cuda()
    x = torch.randn(..., device="cuda")
```

**就绪后（手段7落地到用例）：**

```python
# 示意：门禁与设备跟 is_flex_attention_supported / GPU_TYPE 走
@unittest.skipUnless(
    is_flex_attention_supported(GPU_TYPE),
    f"FlexAttention not supported on {GPU_TYPE}",
)
def test_aot_export_flex_attention_...(self):
    model = FlexAttentionModel(...).to(GPU_TYPE)
    x = torch.randn(..., device=GPU_TYPE)
```

**注意：** 阶段 A 未完成前，不要把 experimental/export 里 Flex 段强行改成 `GPU_TYPE`，避免「门禁仍是 CUDA、body 已是 npu」的半吊子状态。

---

### 8. `TEST_*` / `dtypesIf*` 的 PrivateUse1 对应物

用例里常见 `TEST_CUDA`、`dtypesIfCUDA` 等。PrivateUse1 侧已有：

```python
# common_utils.py
TEST_PRIVATEUSE1 = _is_privateuse1_backend_available()

# common_device_type.py
class dtypesIfPRIVATEUSE1(dtypes):
    def __init__(self, *args):
        super().__init__(*args, device_type=torch._C._get_privateuse1_backend_name())
```

优先用手段 2、3 的通用装饰器；必须按 PrivateUse1 单独声明可用性 / dtype 时再用本手段，不要新造 `TEST_NPU` / `dtypesIfNPU`。

---

### 9. `dist.init_process_group` 硬编码 — 本清单范围外

与 Export/AOTI 清单无关，不处理。

---

## 耦合类型 → 用哪些手段

### 一：设备创建 / CUDA runtime 硬编码

**手段 5**（替换硬编码）+ **手段 2**（`GPU_TYPE`/`HAS_GPU` 正确）。

现状（`test_converter.py`）：

```python
requires_cuda = unittest.skipUnless(torch.cuda.is_available(), "requires cuda")

@requires_cuda
def test_prim_device_cuda(self):
    inp = (torch.rand((3, 4), device="cuda:0"),)
```

现状（`test_nativert.py`）：

```python
for device in ["cpu", "cuda"]:
    if device == "cuda" and not HAS_CUDA_AND_TRITON:
        continue
```

改后：

```python
@requires_gpu
def test_prim_device_accelerator(self):
    inp = (torch.rand((3, 4), device=GPU_TYPE),)

for device in ["cpu", GPU_TYPE]:
    if device == GPU_TYPE and not HAS_GPU:
        continue
```

已有正确写法（`test_export.py`）：

```python
device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
```

**不改：** CUDA 显存峰值、`CUDAGraph` 等专属 API。

---

### 二：skip / 参数化绑死 CUDA

| 手段 | 管什么 | 现状例子 |
|------|--------|----------|
| **手段 3** | 通用 skip 改写 | `@requires_cuda_and_triton`；`skipIf(not TEST_CUDA)` |
| **手段 2** | 手段 3 的前提：`HAS_GPU` 含当前后端 | 否则换装饰器后纯 NPU 仍 skip |
| **手段 1** | 已 `instantiate` 的文件让 PrivateUse1 进调度 | `test_hop.py`：`instantiate_device_type_tests(TestHOP, globals())` |
| **手段 8** | 必须按 PrivateUse1 写可用性 / dtype | `TEST_PRIVATEUSE1`、`dtypesIfPRIVATEUSE1` |

手段 3 现状定义与用法：

```python
requires_cuda_and_triton = unittest.skipUnless(HAS_CUDA_AND_TRITON, "...")

@requires_cuda_and_triton
def test_export_associative_scan_symbol_dim(self):
    device = torch.device("cuda")

@unittest.skipIf(not TEST_CUDA, "CUDA not available")
def test_export_blockmask(self):
    x = torch.randn(2, 128, device="cuda")
```

改法：通用用例 → `requires_gpu` / RFC 通用装饰器；已参数化文件靠手段 1，一般不改调用签名。

**保留：** FakeCuda 等真 CUDA 专属：

```python
@onlyCUDA
class TestExportOnFakeCuda(TestCase): ...
instantiate_device_type_tests(TestExportOnFakeCuda, globals(), only_for="cuda")
```

---

### 三：AOTI 入口 / 编译后端绑定（仅 AOTI；Export 无）

| 手段 | 管什么 | 现状例子 |
|------|--------|----------|
| **手段 2** | `HAS_GPU`/`GPU_TYPE` 决定 GPU 模板能否跑 | `AOTInductorTestABICompatibleGpu` + `if HAS_GPU: run_tests` |
| **手段 6** | runner 不写死 Cuda | `legacy_load_runner` else → Cuda |
| **copy_tests 入口** | 按现有 Gpu 模式挂 privateuse1 | 见下 |

现状入口：

```python
class AOTInductorTestABICompatibleGpu(TestCase):
    device = GPU_TYPE
    device_type = GPU_TYPE

copy_tests(AOTInductorTestsTemplate, AOTInductorTestABICompatibleGpu, GPU_TYPE, GPU_TEST_FAILURES)
```

手段 2、6 就绪后增加 privateuse1 入口；后端需已 `register_backend_for_device` 且 Dynamo interface 可用。

**不改：** CUDA Graph、PTXAS/CUBIN、Windows 交叉编译。

---

### 四：golden / backend 字符串写死

**手段 5** 的断言侧。

通用 golden：设备名跟运行设备走：

```python
self.assertExpectedInline(log, f"""... device(type='{device_type}', index=0) ...""")
```

专属不改——测的就是搬到指定设备且 golden 绑死设备名（`test_passes.py` 末尾 4 个 move 用例在 CUDA 上保留原语义；另见下文局部改）：

```python
ep = move_to_device_pass(ep, "cuda")
# golden 含 to.device(..., 'cuda', ...)
```

`backend="inductor"` 在目标后端已支持时保持不动。

---

### 五：OpInfo / allowlist / 算子差异

| 手段 | 管什么 |
|------|--------|
| **手段 1** | 哪些设备会生成测试（`only_for` / PrivateUse1 调度） |
| **手段 8** + `set_test_configs` | 生成后哪些算子/用例能跑 |

```python
# 调度范围
instantiate_device_type_tests(TestExportOpInfo, globals(), only_for="cpu")  # 保持
instantiate_device_type_tests(TestExportOnFakeCuda, globals(), only_for="cuda")  # 保持

# 算子收窄
DeviceTypeTestBase.set_test_configs(
    op_allowlist={...},
    test_exclusions={"SomeClass": "*", "OtherClass": ["test_a"]},
)
```

不一次全开 accelerator；先 allowlist / exclusions。

---

## 逐文件解耦

动作：`改` / `部分改` / `可选重构` / `不改`。

### 不改（无耦合或专属，共 26 个）

| 文件 | 原因 |
|------|------|
| `test_cpp_serdes.py` 等无耦合 Export 文件* | 无设备硬编码 |
| `testing.py` | 共享工具，无硬编码 |
| `test_model_exports_to_core_aten.py` | 无设备耦合 |
| `test_aot_inductor_arrayref.py` | CPU ArrayRef 专项 |
| `test_aoti_cross_compile_windows.py` | Windows×CUDA 交叉编译专属 |
| `test_package.py` | AOTI cuda device key 校验专属 |
| `test_export_opinfo.py` | CPU OpInfo + FakeCuda 分工保留 |
| `test_hop.py` | 已 `instantiate`；PrivateUse1 注册后即调度；ROCm skip 保留 |
| `test_prims.py` | `@onlyCUDA` / CUDA RNG 专属保留；通用类靠 PrivateUse1 |

\*无耦合 Export：`test_cpp_serdes`、`test_dynamic_shapes`、`test_export_strict`、`test_export_training_ir_to_run_decomp`、`test_functionalized_assertions`、`test_lift_unlift`、`test_pass_infra`、`test_retraceability`、`test_schema`、`test_serdes`、`test_sparse`、`test_strict_export_v2`、`test_swap`、`test_tools`、`test_unflatten`、`test_unflatten_training_ir`、`test_upgrader`、`test_verifier`。

---

### Export：改 / 部分改

#### `test_converter.py` — 部分改｜类型一、二

- **手段：** 5（`device=`）、3（skip）
- **现状：** `requires_cuda` + `device="cuda:0"`
- **改法：** `@requires_gpu` + `device=GPU_TYPE`；可改名 `test_prim_device_accelerator`
- **保留：** 无

#### `test_draft_export.py` — 部分改｜类型一

- **手段：** 5、3
- **现状：** mismatch 用 `torch.device("cuda")`；`test_cuda_memory_usage` 用 `torch.cuda.memory_*`
- **改法：** mismatch 的 bad device → `GPU_TYPE`；相关 skip → 通用装饰器
- **保留：** `test_cuda_memory_usage` 整测

#### `test_experimental.py` — 部分改｜类型一、二、四

- **手段：** 5、3；Flex 段见手段 7（先基础设施，再改用例）
- **现状：** 大量 `TEST_CUDA` + `device="cuda"`；Flex 用 `IS_FLEX_ATTENTION_CUDA_*`；部分 golden 含 cuda
- **改法：** 非 Flex：`TEST_CUDA` → `requires_gpu`；`device="cuda"` → `GPU_TYPE`；通用 golden 参数化。Flex：阶段 A 依赖就绪前保持 CUDA 专属；就绪后按手段 7 阶段 B 改
- **保留（依赖未就绪时）：** FlexAttention / BlockMask 段暂不改

#### `test_export.py` — 部分改｜类型一、二、四

- **手段：** 5、3；Flex 段见手段 7（先基础设施，再改用例）
- **现状：** 已有 `device_type` / `GPU_TYPE` / `requires_gpu`；仍有残留 `device="cuda"`、`@requires_cuda_and_triton`、`skipIf(not torch.cuda.is_available())`、部分 golden、`move_to_device_pass("cuda")`
- **改法：** 通用残留 → `GPU_TYPE` / `device_type`；`@requires_cuda_and_triton` → 通用装饰器
- **保留：** `move_to_device_pass` 及 golden；FakeCuda/`fake_device`；CUDA autocast hop；Flex 段（手段 7 依赖就绪前）

#### `test_nativert.py` — 改｜类型一

- **手段：** 5、2
- **现状：** `["cpu", "cuda"]` + `HAS_CUDA_AND_TRITON`
- **改法：** `["cpu", GPU_TYPE]` + `HAS_GPU`（手段 2 落地后已含 privateuse1）
- **保留：** 无

#### `test_serialize.py` — 部分改｜类型一、二

- **手段：** 5、3
- **现状：** Triton serialize 写死 `"cuda"`；部分 save/load `.cuda()`；`test_weight_sharing_gpu` 已用 `requires_gpu`+`GPU_TYPE`
- **改法：** 可泛化路径统一 `GPU_TYPE` + `requires_gpu`；Triton skip → 通用装饰器
- **保留：** 确认仅 CUDA 的 Triton 路径可保留专属 skip，并注释原因

---

### Export：可选重构

#### `test_db.py` — 可选重构｜扩覆盖（非清 CUDA 耦合）

- **手段：** 1 + 4 + 8（按需）
- **现状：** 仅 `instantiate_parametrized_tests` 按 ExportDB case 名展开；`torch/_export/db` 示例无 device 字段，测试隐式在 CPU 上跑
- **动机：** 多设备批量回归 ExportDB 支持情况；**不是**修 `device="cuda"` 硬编码（本文件无 cuda 耦合）
- **改法：**
  1. 保留 case 参数化，**叠加** `instantiate_device_type_tests`（case × device 双参数化）
  2. `export` 前将 `model`、`example_args`、`example_kwargs` 搬到 `device`（测试侧 `.to(device)`）
  3. NPU 等新后端用 `set_test_configs` 的 **op_allowlist / test_exclusions** 渐进放开，勿一次跑全库示例
- **注意：**
  - 示例数量大，全设备展开会显著拉长 CI
  - 首波失败面宽，优先级低于 converter / export / nativert 等有 cuda 硬编码的文件
  - **可不纳入首波交付**；与 master plan 主路径并行、低优先级

```python
# 示意
class ExampleTests(TestCase):
    def test_exportdb_supported(self, device, name, case):
        model = case.model.to(device)
        args, kwargs = tree_map_tensors_to_device(case.example_args, case.example_kwargs, device)
        exported_program = export(model, args, kwargs, ...)

instantiate_device_type_tests(ExampleTests, globals())
instantiate_parametrized_tests(ExampleTests)
```

---

### Export：局部改（`test_passes` move 用例）

#### `test_passes.py` — 局部改｜类型四（仅 4 个 move 用例）

- **手段：** 4 + 5（**不**整文件 instantiate）
- **现状：** ~23 个 Pass 基础设施测试已在 CPU，无 cuda 耦合；末尾 4 个 `test_move_device_*` 绑 `TEST_CUDA` + `"cuda"` + `assertExpectedInline` golden 字面量 `'cuda'`
- **改法：**
  1. **前 ~23 个用例不动**（无 device 参数化收益）
  2. 将 4 个 move 用例拆到单独类（如 `TestMoveToDevicePass`），`instantiate_device_type_tests(..., except_for="cpu")`
  3. `"cuda"` / `"cuda:0"` → `device` / `f"{device}:0"`；`move_to_device_pass(ep, "cuda")` → `move_to_device_pass(ep, device)`
  4. `{"cpu": "cuda:0"}` → `{"cpu": f"{device}:0"}`；模型内 `autocast(device_type="cuda")` 随 `device` 改
  5. **golden：** 按 device 分支维护多套 `assertExpectedInline`，或改为断言图中 `device` metadata，避免单套 `'cuda'` 字符串硬套到 NPU
  6. skip：`@skipIf(not TEST_CUDA)` → `@requires_gpu` 或 `skipCUDAIf` 等等价物（RFC 落地后）
- **保留：** CUDA 设备类上继续覆盖现有 cpu↔cuda 往返语义；NPU 适配为增量，不要求与 cuda golden 逐字相同
- **不做：** 整文件 `instantiate_device_type_tests(TestPasses, ...)`（会在 CPU/NPU 上重复跑 23 个纯 CPU Pass 测试，ROI 极低）

| 用例 | 处理 |
|------|------|
| `test_move_device_to` | 拆到 move 类；golden 按 device |
| `test_move_device_submod` | 同上；autocast `device_type` 跟 device |
| `test_move_to_device_pass` | 同上；cpu↔accelerator 往返映射泛化 |
| `test_move_device_example_inputs` | 同上；`example_inputs` 搬到 `f"{device}:0"` |
| 其余 ~23 个 Pass 测试 | 不改 |

---

### AOTI：改 / 部分改

#### `test_aot_inductor_utils.py` — 改｜类型三

- **手段：** 6
- **现状：** `legacy_load_runner` else → Cuda
- **改法：** 按手段 6 动态加载；`cpu`/`mps` 无 device 参数；缺失则报错
- **保留：** FBCODE 临时目录逻辑

#### `test_aoti_torchbind_constants.py` — 改｜类型一、二

- **手段：** 5、2
- **现状：** 本地 `HAS_CUDA`；`device = "cuda" if HAS_CUDA else "cpu"`
- **改法：** `GPU_TYPE if HAS_GPU else "cpu"`；skip 对齐 `HAS_GPU` / Triton
- **保留：** 无

#### `test_aot_inductor.py` — 部分改｜类型一、三、四

- **手段：** 5、2、6 + copy_tests 入口
- **现状：** 模板多用 `self.device`；约 11 处裸 `"cuda"` / `torch.cuda.*`；`copy_tests` 挂 CPU/`GPU_TYPE`/MPS
- **改法：**
  1. 可泛化残留 → `self.device` / `GPU_TYPE`
  2. 增加 privateuse1 `copy_tests` 入口 + failures 表
  3. FileCheck 按 `self.device` 分支扩展
  4. `fail_gpu` 后缀跟 `GPU_TYPE` 收敛
- **保留：** CUDAGraph、pinned async、multigpu、ptxas、Flash/SDPA CUDA、`torch.cuda.memory_*`

| 用例 | 处理 |
|------|------|
| `test_loaded_modules_tracking` | 跟 `self.device` / 按设备分支 |
| `test_small_constant_pinned_async_copy` | 保留 CUDA-only |
| `test_scaled_grouped_mm` | 跟 `self.device`（若后端支持） |
| `test_cond_cpu_predicate_cuda_operands` | 泛化为 accelerator operands，或保留并注释 |
| `test_repeated_calling` | `torch.accelerator.synchronize()` |
| `test_runtime_checks_fp8` | 按后端能力 skip |
| `test_cuda_to_cuda_device_copy` | 保留或泛化为同设备 copy |
| weight caching allocator | 保留 CUDA allocator |
| `test_with_cudagraphs` | 保留 |

#### `test_aot_inductor_custom_ops.py` — 部分改｜类型三

- **手段：** 2、5 + copy_tests 入口
- **现状：** 模板用 `self.device`；`impl`/`shim` 写死 CPU/CUDA/XPU
- **改法：** 增加 privateuse1 入口；`fail_gpu` 动态化；新后端 `impl`/shim 走注册扩展
- **保留：** 现有 CUDA/XPU shim 字符串

#### `test_aot_inductor_package.py` — 部分改｜类型三、四

- **手段：** 2、5
- **现状：** `parameterized_class`：`cpu` + `GPU_TYPE`；少量 `!= "cuda"`；cubin/ptxas
- **改法：** 参数化含 privateuse1；可泛化门控用 `GPU_TYPE`
- **保留：** PTXAS/CUBIN 相关

---

## 文件总表

| 文件 | 类型 | 手段 | 动作 |
|------|------|------|------|
| `test_converter.py` | 一、二 | 5, 3 | 部分改 |
| `test_draft_export.py` | 一 | 5, 3 | 部分改 |
| `test_experimental.py` | 一、二、四 | 5, 3, 7 | 部分改（Flex 跟手段7依赖） |
| `test_export.py` | 一、二、四 | 5, 3, 7 | 部分改（Flex 跟手段7依赖） |
| `test_nativert.py` | 一 | 5, 2 | 改 |
| `test_serialize.py` | 一、二 | 5, 3 | 部分改 |
| `test_db.py` | — | 1, 4, 8 | 可选重构（扩 ExportDB 多设备覆盖） |
| `test_passes.py` | 四 | 4, 5 | 局部改（仅 4 个 move 用例） |
| `test_hop.py` | 二 | 1 | 不改 |
| `test_prims.py` | 二、五 | 1, 8 | 不改（专属保留） |
| `test_export_opinfo.py` | 二、五 | 1 | 不改 |
| `test_package.py` | — | — | 不改 |
| `test_aot_inductor_utils.py` | 三 | 6 | 改 |
| `test_aoti_torchbind_constants.py` | 一、二 | 5, 2 | 改 |
| `test_aot_inductor.py` | 一、三、四 | 5, 2, 6 | 部分改 |
| `test_aot_inductor_custom_ops.py` | 三 | 2, 5 | 部分改 |
| `test_aot_inductor_package.py` | 三、四 | 2, 5 | 部分改 |
| `test_aot_inductor_arrayref.py` | — | — | 不改 |
| `test_aoti_cross_compile_windows.py` | — | — | 不改 |
| 其余无耦合文件 | — | — | 不改 |

手段编号对照：1 PrivateUse1 调度 · 2 GPU_TYPE/HAS_GPU 泛化 · 3 skip 改写 · 4 补 instantiate · 5 替换 cuda 硬编码 · 6 AOTI runner · 7 Flex（先基础设施依赖，再改用例） · 8 TEST/dtypes PrivateUse1 · 9 dist（范围外）

---

## 实施顺序

```
1. 依赖：RFC #189138（手段 2、3）
   get_gpu_type / HAS_GPU / HAS_TRITON / requires_* 设备无关化

2. 手段 6
   legacy_load_runner 动态加载
   （NpuRunner 类仍为后端依赖）

3. 手段 5、3 — 本清单测试文件主改动
   清通用硬编码与 skip
   converter / nativert / torchbind / export / serialize / draft_export / experimental(非 Flex)
   aot_inductor 模板可泛化残留 → self.device

4. 手段 1 + AOTI 入口
   PrivateUse1 调度确认（hop / prims）
   AOTI privateuse1 copy_tests + failures
   custom_ops / package 接入

5. CI
   PYTORCH_TESTING_DEVICE_ONLY_FOR=<backend>
   或 PYTORCH_TESTING_DEVICE_FOR_CUSTOM=privateuse1

6. 手段 7（可选后续波次）
   6a. 依赖：common_device_type Flex 注册表 + skip 工厂 + 后端 Flex 能力
   6b. 再改 test_experimental.py / test_export.py 中 Flex 段

7. 可选 / 局部（低优先级或独立波次）
   test_db.py：ExportDB case × device 双参数化 + allowlist
   test_passes.py：仅 4 个 move_to_device 用例拆类 + device 泛化
```

---

## 验收

1. 纯 privateuse1 机器上 `GPU_TYPE` 为实际后端名，不回退 `"cuda"`
2. 可改段无裸 `device="cuda"` / `.cuda()`（专属除外；Flex 在手段7依赖未就绪前可仍为 CUDA）
3. `legacy_load_runner(<backend>, ...)` 不落入 Cuda runner
4. `test_hop` / 通用 `test_prims` 在 PrivateUse1 + CI env 下可调度
5. AOTI privateuse1 入口最小用例通过
6. FakeCuda、CUDA Graph、Windows 交叉编译、`test_cuda_memory_usage` 不要求在非 CUDA 上通过
7. `test_passes` 4 个 move 用例：局部改完成前可保持 CUDA 专属；改完后在目标加速器上 `move_to_device_pass(ep, device)` 通过，不要求与 cuda golden 逐字相同
8. 手段7依赖就绪后：experimental/export 中已纳入泛化的 Flex 用例可在支持 Flex 的当前设备上调度；未支持设备正确 skip
9. （可选）`test_db` 多设备 ExportDB：按 allowlist 渐进验收，不要求首波全绿

---

## 总结

1. 任务主体是改本文列出的 Export/AOTI **测试文件**；RFC、NpuRunner、Flex 基础设施等为依赖项。
2. 手段 2、3 依赖 RFC #189138；手段 7 纳入考虑，但须先基础设施再改 Flex 用例；手段 9 范围外。
3. 调度用已有 PrivateUse1。
4. 通用跟当前设备；CUDA Graph / FakeCuda 等专属保留；Flex 按手段 7 两阶段推进。
5. `test_db.py` 为可选重构（ExportDB 多设备扩覆盖）；`test_passes.py` 仅末尾 4 个 move 用例局部改，不整文件 instantiate。
