# Export / AOTI 测试设备无关改造设计

本文面向下列 Export / AOTI 测试用例的解耦改造。目标是把测试里不必要的 CUDA 写死逻辑，改成跟随当前测试设备或当前后端能力的写法。

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

**核心原则：** 先看测试真正想验证什么，再替换硬编码设备；通用逻辑跟随当前设备，后端专属逻辑保留专属入口。

**参考判断顺序：**

1. 这个测试是否只验证 Python 逻辑、图规则、schema、序列化格式，不真的依赖 GPU 或 AOTI 编译？如果是，保持普通 `TestCase` / CPU 写法，不要强行改成 accelerator 测试。
2. 这个测试是否只是在加速器上跑 export / compile / AOTI，逻辑本身通用？如果是，优先使用当前测试设备（`device_type` / `GPU_TYPE` / `self.device`），而不是写死 `"cuda"`。
3. 这个测试是不是专门验证某个后端独有能力，比如 CUDA Graph、FlexAttention、FakeCuda、PTXAS/CUBIN、Windows 交叉编译、Triton CUDA 路径？如果是，保留专属 class 或专属 skip。

---

## 常见基础设施

### `device_type` / `GPU_TYPE` / `self.device`

作用：表示当前测试应该使用的设备类型字符串，例如 `"cuda"`、`"xpu"`、`"npu"`、`"cpu"`。

适用场景：创建 tensor、module 时替换硬编码 `"cuda"`。

Export 文件里已有写法：

```python
# test/export/test_export.py
device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
```

AOTI 文件里常用：

```python
# torch/testing/_internal/inductor_utils.py
GPU_TYPE = get_gpu_type()

# test/inductor/test_aot_inductor.py
class AOTInductorTestABICompatibleGpu(TestCase):
    device = GPU_TYPE
    device_type = GPU_TYPE
```

模板方法里用 `self.device`：

```python
example_inputs = (
    torch.randn(10, 10, device=self.device),
    torch.randn(10, 10, device=self.device),
)
```

### `torch.accelerator`

作用：统一 accelerator 入口，获取当前加速设备、设备数量、同步等。

适用场景：把 `torch.cuda.is_available()`、`torch.cuda.synchronize()` 这类 CUDA 专属 runtime API 改成通用写法。

```python
torch.accelerator.current_accelerator().type   # "cuda" / "xpu" / "npu" ...
torch.accelerator.is_available()
torch.accelerator.device_count()
torch.accelerator.synchronize()
```

NPU 注册成 PrivateUse1 后端后，`current_accelerator().type` 可返回 `"npu"`。

### `instantiate_device_type_tests` / `copy_tests`

作用：按设备类型自动生成多套测试，或把同一套模板挂到不同设备入口。

Export 示例：

```python
# test/export/test_hop.py
instantiate_device_type_tests(TestHOP, globals())
```

AOTI 示例：

```python
copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestABICompatibleGpu,
    GPU_TYPE,
    GPU_TEST_FAILURES,
)
```

### 当前基础设施缺口（改造前必须修）

```python
# torch/_inductor/utils.py
GPU_TYPES = ["cuda", "mps", "xpu", "mtia"]  # 不含 npu

def get_gpu_type() -> str:
    avail_gpus = [x for x in GPU_TYPES if getattr(torch, x).is_available()]
    gpu_type = "cuda" if len(avail_gpus) == 0 else avail_gpus.pop()  # 无设备时回退 cuda
    return gpu_type
```

```python
# torch/testing/_internal/inductor_utils.py
HAS_GPU = HAS_CUDA_AND_TRITON or HAS_XPU_AND_TRITON  # 不含 NPU
```

改造要求：

1. `get_gpu_type()` 优先走 `torch.accelerator.current_accelerator()`；无设备时返回 `"cpu"`，禁止回退 `"cuda"`
2. `GPU_TYPES` / `HAS_GPU` 纳入 NPU（fork 中增加 `HAS_NPU_AND_TRITON`）
3. 新建统一装饰器，例如 `requires_accelerator`、`requires_accelerator_and_triton`
4. `instantiate_device_type_tests` 支持 `allow_npu=True`，并增加 `NpuTestBase`

建议统一入口（新建 `torch/testing/_internal/common_accel_utils.py`）：

```python
def get_accelerator_type(*, require_available: bool = False) -> str:
    acc = torch.accelerator.current_accelerator(check_available=require_available)
    return acc.type if acc is not None else "cpu"

ACCELERATOR_TYPE = get_accelerator_type()

requires_accelerator = unittest.skipUnless(
    torch.accelerator.is_available(), "requires accelerator"
)
```

过渡期：`GPU_TYPE` 继续导出，语义对齐当前加速器类型；新代码优先用 `ACCELERATOR_TYPE`。

---

## 一、耦合类型一：设备创建和 CUDA runtime API 耦合

### 核心问题

测试逻辑本身是通用 export / AOTI 行为，但 `device="cuda"`、`.cuda()`、`torch.cuda.*` 写死 CUDA，导致非 CUDA accelerator 无法复用。

### 改造建议

- tensor / module 创建优先使用 `device_type`、`GPU_TYPE`、`ACCELERATOR_TYPE` 或 `self.device`
- 同步、设设备等 runtime 操作优先使用 `torch.accelerator` 或 `torch.get_device_module(device_type)`
- 后端没有的 API（如某些 `memory_stats`）按能力 skip，不要整测绑死 CUDA

### 示例 1：`test_converter.py`

```python
# 改前
requires_cuda = unittest.skipUnless(torch.cuda.is_available(), "requires cuda")

@requires_cuda
def test_prim_device_cuda(self):
    ...
    inp = (torch.rand((3, 4), device="cuda:0"),)

# 改后
from torch.testing._internal.common_accel_utils import (
    ACCELERATOR_TYPE,
    requires_accelerator,
)

@requires_accelerator
def test_prim_device_accelerator(self):
    class Module(torch.nn.Module):
        def forward(self, x):
            device = x.device
            return torch.ones(2, 3, device=device)

    inp = (torch.rand((3, 4), device=ACCELERATOR_TYPE),)
    self._check_equal_ts_ep_converter(Module(), inp)
```

修改点：

- 不再用 `torch.cuda.is_available()` 决定能不能跑
- 不再写死 `device="cuda:0"`，改为跟随当前加速器

### 示例 2：`test_aoti_torchbind_constants.py`

```python
# 改前
HAS_CUDA = torch.cuda.is_available()

@unittest.skipIf(HAS_CUDA and not HAS_TRITON, "requires triton on CUDA builds")
def test_custom_objs_exposed_through_loader(self):
    device = "cuda" if HAS_CUDA else "cpu"
    m = self._make_model().to(device)
    x = torch.randn(2, 3, device=device)

# 改后
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU, HAS_TRITON

@unittest.skipIf(HAS_GPU and not HAS_TRITON, "requires triton on GPU builds")
def test_custom_objs_exposed_through_loader(self):
    device = GPU_TYPE if HAS_GPU else "cpu"
    m = self._make_model().to(device)
    x = torch.randn(2, 3, device=device)
```

同文件另外两处 `"cuda" if HAS_CUDA else "cpu"` 一并改。

### 示例 3：`test_nativert.py`

```python
# 改前
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON

for device in ["cpu", "cuda"]:
    if device == "cuda" and not HAS_CUDA_AND_TRITON:
        continue
    ...
    (get_module.__func__().to(device), (torch.randn(4, 4).to(device),)),

# 改后
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU_AND_TRITON

for device in ["cpu", GPU_TYPE]:
    if device == GPU_TYPE and not HAS_GPU_AND_TRITON:
        continue
    ...
```

### 示例 4：`test_serialize.py` / `test_export.py` 零星硬编码

```python
# 改前
@unittest.skipIf(not torch.cuda.is_available(), "Requires cuda")
def test_...(self):
    model = MyModule().eval().cuda()
    inp = (torch.randn(1, 64, device="cuda"),)

# 改后
@requires_accelerator
def test_...(self):
    model = MyModule().eval().to(ACCELERATOR_TYPE)
    inp = (torch.randn(1, 64, device=ACCELERATOR_TYPE),)
```

`test_export.py` 顶部已有 `device_type`，且大量使用 `GPU_TYPE`；只需清理剩余约 7 处裸 `device="cuda"`。

### 本类型涉及文件

| 文件 | 动作 |
|------|------|
| `test_converter.py` | 改 |
| `test_aoti_torchbind_constants.py` | 改 |
| `test_nativert.py` | 改 |
| `test_serialize.py` | 改零星硬编码 |
| `test_export.py` | 改零星硬编码 |
| `test_draft_export.py` | 改 mismatch 用例中的 `device=cuda`；`test_cuda_memory_usage` 不改 |
| `test_experimental.py` | 改非 FlexAttention 段；Flex 段不改 |
| `test_aot_inductor.py` | 模板内裸 `device="cuda"` 改为 `self.device`；CUDA Graph / memory 专属不改 |

### 本类型明确不改

```python
# test/export/test_draft_export.py — CUDA 显存峰值，专属
torch.cuda.reset_peak_memory_stats()
base_usage = torch.cuda.memory_allocated(device)
peak_mem_usage = torch.cuda.memory_stats(device)["allocated_bytes.all.peak"]

# test/inductor/test_aot_inductor.py — CUDAGraph 专属
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    ...
```

---

## 二、耦合类型二：测试准入、skip 装饰器和 device-type 参数化耦合 CUDA

### 核心问题

测试主体可以被多个 accelerator 复用，但准入层使用 `@requires_cuda`、`@onlyCUDA`、`TEST_CUDA`、`only_for=("cuda",)` 或整类 skip，导致非 CUDA 后端无法实例化或运行。

### 改造建议

- 通用测试去掉 `@requires_cuda` / `@onlyCUDA`，改成 `requires_accelerator` 或框架注入的 `device`
- 已参数化文件打开 NPU：`instantiate_device_type_tests(..., allow_npu=True)`
- 真 CUDA 专属继续保留专属 skip / `only_for=("cuda",)`
- 后端差异用 `test_exclusions` 精确跳过，不要把粗粒度 CUDA 条件写进通用测试本体

### 示例 1：`test_experimental.py` 非 Flex 段

```python
# 改前
@unittest.skipIf(not TEST_CUDA, "CUDA not available")
def test_export_blockmask(self):
    ...
    x = torch.randn(2, 128, device="cuda")

# 改后
@requires_accelerator
def test_export_blockmask(self):
    ...
    x = torch.randn(2, 128, device=ACCELERATOR_TYPE)
```

FlexAttention 专属段保留：

```python
@unittest.skipIf(
    not (IS_FLEX_ATTENTION_CUDA_PLATFORM_SUPPORTED and not torch.version.hip),
    ...
)
def test_aot_export_flex_attention_callable_mask_mod(self):
    ...
```

### 示例 2：`test_hop.py` / `test_prims.py` 打开 NPU 调度

```python
# 改前
instantiate_device_type_tests(TestHOP, globals())

# 改后
instantiate_device_type_tests(TestHOP, globals(), allow_npu=True)
```

前提：`common_device_type.py` 增加 `NpuTestBase`，并在 `get_desired_device_type_test_bases(..., allow_npu=True)` 中挂上。

### 示例 3：`test_export_opinfo.py` — 专属保留

```python
# 不改：FakeCuda 场景就是测 “有 cuda build 但无 GPU”
@onlyCUDA
class TestExportOnFakeCuda(TestCase):
    ...
instantiate_device_type_tests(TestExportOnFakeCuda, globals(), only_for="cuda")
```

CPU 侧 `TestExportOpInfo` 保持 `only_for="cpu"` 即可，不要一次性全开 accelerator。

### 本类型涉及文件

| 文件 | 动作 |
|------|------|
| `test_experimental.py` | 非 Flex 段改 skip；Flex 段保留 |
| `test_hop.py` | `allow_npu=True` |
| `test_prims.py` | `allow_npu=True` |
| `test_export.py` | `@requires_cuda_and_triton` 在基础设施就绪后改为 `requires_accelerator_and_triton` |
| `test_serialize.py` | 同上 |
| `test_export_opinfo.py` | FakeCuda 保留；CPU OpInfo 不盲目全开 |

---

## 三、耦合类型三：AOTI / Inductor 测试入口和编译后端绑定特定设备

> **适用范围：仅 AOTI 文件。**  
> 范围内 Export 文件（`test/export/*`、`test/test_prims.py`、`test/test_model_exports_to_core_aten.py`）**没有类型三**。  
> Export 测的是图导出、序列化、pass、schema 等，不依赖 AOTI `copy_tests` 设备入口，也不依赖 Inductor 按设备注册的编译后端。  
> 对应 DTensor 文档里的「通信 backend 写死 NCCL」；在本清单里，同类问题只出现在 AOTI 侧。

### 核心问题

AOTI 测试有这类绑定：

- AOTI 测试入口只挂了 CPU / `GPU_TYPE` / MPS，没有 NPU
- `HAS_GPU` 不含 NPU，GPU 模板在纯 NPU 机器上整组 skip
- Inductor 未注册 NPU 后端时，即使改了设备名也无法编译

通用 AOTI 测试应跟随当前设备入口；CUDA 驱动 / 工具链专属测试保留专属入口。

### 改造建议

- 按现有 `AOTInductorTestABICompatibleGpu` 模式增加 NPU `copy_tests` 入口
- `HAS_GPU` 纳入 `HAS_NPU_AND_TRITON`
- NPU fork 提供 Inductor `register_backend_for_device("npu", ...)` 与 Dynamo `NpuInterface`
- CUDA Graph、PTXAS、Windows 交叉编译保留专属，不强行泛化

### 示例 1：现有 GPU 入口（参考，已存在）

```python
# test/inductor/test_aot_inductor.py
class AOTInductorTestABICompatibleGpu(TestCase):
    device = GPU_TYPE
    device_type = GPU_TYPE
    ...

copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestABICompatibleGpu,
    GPU_TYPE,
    GPU_TEST_FAILURES,
)
```

模板内已用 `self.device`，这是正确方向。

### 示例 2：新增 NPU 入口

```python
from torch.testing._internal.inductor_utils import HAS_NPU_AND_TRITON

NPU_TEST_FAILURES = {
    # 从 GPU_TEST_FAILURES 拷贝后，按 NPU 实际能力收敛
}

@unittest.skipIf(not HAS_NPU_AND_TRITON, "requires npu and triton")
class AOTInductorTestABICompatibleNpu(TestCase):
    device = "npu"
    device_type = "npu"
    check_model = check_model
    check_model_with_multiple_inputs = check_model_with_multiple_inputs
    code_check_count = code_check_count
    allow_stack_allocation = False
    use_minimal_arrayref_interface = False

copy_tests(
    AOTInductorTestsTemplate,
    AOTInductorTestABICompatibleNpu,
    "npu",
    NPU_TEST_FAILURES,
)
```

模板内设备分支扩展：

```python
if self.device == "mps":
    FileCheck().check("aoti_torch_mps_get_kernel_function(").run(code)
elif self.device == "npu":
    FileCheck().check("<npu_launch_symbol>")  # 按 NPU codegen 实际符号填写
elif self.device == GPU_TYPE:
    FileCheck().check("launchKernel(").run(code)
```

### 示例 3：Inductor 后端注册前提

`torch/_inductor/codegen/common.py` 已有 privateuse1 扩展点：

```python
private_backend = torch._C._get_privateuse1_backend_name()
if (
    private_backend != "privateuseone"
    and get_scheduling_for_device(private_backend) is None
):
    device_scheduling = _get_custom_mod_func("Scheduling")
    wrapper_codegen = _get_custom_mod_func("PythonWrapperCodegen")
    cpp_wrapper_codegen = _get_custom_mod_func("CppWrapperCodegen")
    fx_wrapper_codegen = _get_custom_mod_func("WrapperFxCodegen")
    if device_scheduling and wrapper_codegen and cpp_wrapper_codegen:
        register_backend_for_device(
            private_backend,
            device_scheduling,
            wrapper_codegen,
            cpp_wrapper_codegen,
            fx_wrapper_codegen,
        )
```

NPU 侧需提供 Scheduling / PythonWrapper / CppWrapper，并保证最小 `aoti_compile_and_package` 可跑通。没有这一步，只改测试设备名不够。

### 本类型涉及文件

| 文件 | 动作 |
|------|------|
| `test_aot_inductor.py` | 新增 NPU `copy_tests`；CUDA 专属子测试保留 |
| `test_aot_inductor_arrayref.py` | 增加 NPU 入口或复用 `self.device` |
| `test_aot_inductor_custom_ops.py` | 跟随 `GPU_TYPE` / `HAS_GPU` 扩展 |
| `test_aot_inductor_package.py` | 跟随 `GPU_TYPE`；PTXAS/CUBIN 专属保留 |
| `test_aot_inductor_utils.py` | 不改 |
| `test_aoti_cross_compile_windows.py` | 全文件不改（平台专属） |

### 本类型明确不改

```python
# CUDA 驱动 / 架构专属
if self.device != "cuda":
    raise unittest.SkipTest("requires CUDA/HIP")

# Windows 交叉编译专属
"aot_inductor.cross_target_platform": "windows"
```

---

## 四、耦合类型四：compile / export / debug 的预期字符串或 backend 写死特定后端

### 核心问题

Export / AOTI 测试常用 `assertExpectedInline()`、FileCheck 校验 graph / log / 生成代码。如果 golden 写死 `'cuda'`，或 compile backend / pass 目标写死 CUDA 专属能力，非 CUDA 后端会失败。这里要区分通用图语义与 backend-specific 能力。

### 改造建议

- 通用日志 / graph 字符串里的设备名改成 `f'{device_type}'` 或 `self.device`
- 测的是「搬到 cuda」的 pass / IR，保留 `"cuda"` 和对应 golden
- `torch.compile(..., backend="inductor")`：若 NPU 已支持 Inductor，不要改成 `aot_eager`；只有测通用图语义且 NPU 不支持 Inductor 时才考虑替换

### 示例 1：`test_passes.py` — 专属，不改

```python
@unittest.skipIf(not TEST_CUDA, "requires cuda")
def test_move_device_to(self):
    ...
    ep = move_to_device_pass(ep, "cuda")
    self.assertExpectedInline(
        ep.graph_module.code.strip("\n"),
        """\
def forward(self, x):
    ...
    to = torch.ops.aten.to.device(x, 'cuda', torch.float32);  x = None
    ...
""",
    )
```

这里 `"cuda"` 是 pass 目标设备，golden 也断言 `'cuda'`，属于专属测试。

### 示例 2：通用 golden 应参数化（若范围内出现）

```python
# 改前
self.assertExpectedInline(
    log_string(),
    """... DeviceMesh((2,), 'cuda', stride=(1,)) ...""",
)

# 改后
self.assertExpectedInline(
    log_string(),
    f"""... DeviceMesh((2,), '{device_type}', stride=(1,)) ...""",
)
```

### 示例 3：`backend="inductor"`

范围内 AOTI / export+compile 测试若目标是覆盖 Inductor 路径，且 NPU 支持 Inductor：

```python
opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
```

保持不动。不要为了“看起来通用”改成 `aot_eager`，否则测不到目标路径。

### 本类型涉及文件

| 文件 | 动作 |
|------|------|
| `test_passes.py` | `move_to_device_pass(..., "cuda")` 保留 |
| `test_experimental.py` | FlexAttention 相关保留 |
| `test_aot_inductor.py` | FileCheck 按 `self.device` 分支；CUDA Graph 保留 |
| `test_aot_inductor_package.py` | PTXAS / CUBIN 相关保留 |

---

## 五、耦合类型五：OpInfo、operator 支持、skip/xfail 和 allowlist 耦合

### 核心问题

`test_export_opinfo.py`、`test_prims.py` 等通过 OpInfo / device-type 批量生成测试。不同后端算子支持不同；不能一刀切全开 accelerator。

### 改造建议

- 算子支持差异通过 `op_overrides`、`op_allowlist`、`test_exclusions`、`set_test_configs(...)` 表达
- 暂时只能 CPU 跑的 OpInfo，不要一次性打开 accelerator；先建立失败列表 / allowlist，再逐步放开
- FakeCuda 这类场景测试继续 `only_for="cuda"`

### 示例 1：统一配置入口（框架已有）

```python
# torch/testing/_internal/common_device_type.py
@classmethod
def set_test_configs(
    cls,
    *,
    op_overrides=None,
    op_allowlist=None,
    test_exclusions=None,
):
    cls.op_overrides = op_overrides
    cls.op_allowlist = op_allowlist
    cls.test_exclusions = test_exclusions
```

```python
@classmethod
def _apply_op_allowlist(cls, ops):
    if cls.op_allowlist is None:
        return
    supported_set = set(cls.op_allowlist)
    ops.op_list = [op for op in ops.op_list if op.full_name in supported_set]
```

NPU 侧可在测试基类上：

```python
DeviceTypeTestBase.set_test_configs(
    op_allowlist={...},          # 仅生成支持的算子
    test_exclusions={
        "SomeClass": "*",        # 整类跳过
        "OtherClass": ["test_a"],
    },
)
```

### 示例 2：`test_export_opinfo.py`

```python
# CPU OpInfo：保持
instantiate_device_type_tests(TestExportOpInfo, globals(), only_for="cpu")

# FakeCuda：保持 CUDA-only
instantiate_device_type_tests(TestExportOnFakeCuda, globals(), only_for="cuda")
```

若后续要在 NPU 上跑部分 OpInfo：先 allowlist，再 `allow_npu=True`，不要直接全开。

### 示例 3：`test_prims.py`

已 `instantiate_device_type_tests`。改造动作：

1. 基础设施支持 `allow_npu=True`
2. 用 `op_allowlist` / `test_exclusions` 收敛 NPU 不支持的算子
3. 再打开 NPU 调度

### 本类型涉及文件

| 文件 | 动作 |
|------|------|
| `test_export_opinfo.py` | CPU / FakeCuda 保持；NPU 若开放则走 allowlist |
| `test_prims.py` | `allow_npu=True` + exclusions / allowlist |
| `test_hop.py` | 主要是 device-type 调度，算子差异少；打开 `allow_npu` 即可 |

---

## 文件总表

### 无耦合，不改

```
test_cpp_serdes.py
test_db.py
test_dynamic_shapes.py
test_export_strict.py
test_export_training_ir_to_run_decomp.py
test_functionalized_assertions.py
test_lift_unlift.py
test_pass_infra.py
test_retraceability.py
test_schema.py
test_serdes.py
test_sparse.py
test_strict_export_v2.py
test_swap.py
test_tools.py
test_unflatten.py
test_unflatten_training_ir.py
test_upgrader.py
test_verifier.py
test_model_exports_to_core_aten.py
testing.py
test_aot_inductor_utils.py
```

### 按类型改造

| 文件 | 主要耦合类型 | 动作 |
|------|--------------|------|
| `test_converter.py` | 一、二 | 改 device / skip |
| `test_aoti_torchbind_constants.py` | 一、二 | 改 device / HAS_GPU |
| `test_nativert.py` | 一 | 改设备循环 |
| `test_serialize.py` | 一、二 | 改零星硬编码；Triton skip 后续通用化 |
| `test_export.py` | 一、二、四 | 改零星硬编码；Triton skip 后续通用化 |
| `test_draft_export.py` | 一 | 改 mismatch；memory_usage 保留 |
| `test_experimental.py` | 一、二、四 | 非 Flex 改；Flex 保留 |
| `test_passes.py` | 四 | `move_to_device_pass("cuda")` 保留 |
| `test_hop.py` | 二、五 | `allow_npu=True` |
| `test_prims.py` | 二、五 | `allow_npu=True` + allowlist |
| `test_export_opinfo.py` | 二、五 | FakeCuda 保留；CPU 保持 |
| `test_package.py` | 二 | 视具体用例；CUDA 不可用语义保留 |
| `test_aot_inductor.py` | 一、三、四 | 加 NPU 入口；专属子测试保留 |
| `test_aot_inductor_arrayref.py` | 三 | 跟 NPU 入口 |
| `test_aot_inductor_custom_ops.py` | 三 | 跟 `GPU_TYPE` / `HAS_GPU` |
| `test_aot_inductor_package.py` | 三、四 | 跟 GPU 入口；PTXAS/CUBIN 保留 |
| `test_aoti_cross_compile_windows.py` | 三、四 | 全文件不改 |
| `test_aoti_torchbind_constants.py` | 一、二 | 改 device / HAS_GPU（无类型三入口问题） |

---

## 实施顺序

```
1. 修基础设施
   - get_gpu_type 禁止回退 "cuda"
   - HAS_NPU_AND_TRITON 并入 HAS_GPU
   - common_accel_utils（ACCELERATOR_TYPE / requires_accelerator）
   - NpuTestBase + allow_npu

2. 类型一、二：改可泛化硬编码与 skip
   - converter / torchbind / nativert / serialize / export 零星
   - draft_export mismatch / experimental 非 Flex

3. 类型二、五：打开已参数化文件的 NPU 调度
   - hop / prims：allow_npu=True
   - prims / export_opinfo：按需 allowlist

4. 类型三：AOTI NPU 入口
   - 确认 NPU Inductor/AOTI 可编译
   - AOTInductorTestABICompatibleNpu + NPU_TEST_FAILURES
   - arrayref / custom_ops / package 接入

5. CI
   - export-npu / aoti-npu 独立 job
   - PYTORCH_TESTING_DEVICE_ONLY_FOR=npu
```

---

## 验收

1. 纯 NPU 机器上 `get_accelerator_type()` / `GPU_TYPE` 为 `"npu"`，不再静默变成 `"cuda"`
2. 类型一可改段无裸 `device="cuda"` / `.cuda()`（专属段除外）
3. `test_hop.py` / `test_prims.py` 在 NPU job 可被调度
4. `AOTInductorTestABICompatibleNpu.test_simple` 通过
5. 类型三/四列出的专属测试不要求在 NPU 通过

---

## 总结

对本清单内 Export / AOTI 测试：

1. **先看测试想验证什么**：通用逻辑跟当前设备；专属能力保留专属入口
2. **按耦合类型改**：
   - Export：主要是类型一、二、四、五（**无类型三**）
   - AOTI：类型一、二、三、四（类型三是 AOTI 入口 / Inductor 后端）
3. **先修 `GPU_TYPE` / `HAS_GPU` / NPU 调度基类**，再改测试文件，再加 AOTI NPU 模板
4. **CUDA Graph、FlexAttention、FakeCuda、`move_to_device_pass("cuda")`、Windows 交叉编译等不改**
