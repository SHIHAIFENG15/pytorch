# Export / AOTI 测试用例硬件耦合解耦改造指南

本文档面向 **Export**（`test/export/`）与 **AOTI**（`test/inductor/test_aot_*.py`）测试的硬件耦合分析，沿用 [code-plan.md](code-plan.md) 与 [dynamo-test-decoupling-plan.md](dynamo-test-decoupling-plan.md) 的分类框架。

**参考提交**：`85f319fa`（`skip_if_lt_x_devices` + `torch.accelerator` 动态推导）

---

## 1. 文件清单与扫描摘要

### 1.1 Export（31 个）

| 文件 | 耦合命中 | 分类 |
|------|----------|------|
| `test_converter.py` | 5 | P1 小改 |
| `test_cpp_serdes.py` | 0 | 无耦合 |
| `test_db.py` | 0 | 无耦合 |
| `test_draft_export.py` | 12 | 混合（1 专属 + 1 可改） |
| `test_dynamic_shapes.py` | 0 | 无耦合 |
| `test_experimental.py` | 62 | 混合（FlexAttention 专属 + 可泛化） |
| `test_export_opinfo.py` | 19 | 混合（FakeCuda 专属 + 已参数化） |
| `test_export.py` | 56 | 混合（大部分已用 `GPU_TYPE`，少量硬编码） |
| `test_export_strict.py` | 0 | 无耦合 |
| `test_export_training_ir_to_run_decomp.py` | 0 | 无耦合 |
| `test_functionalized_assertions.py` | 0 | 无耦合 |
| `test_hop.py` | 5 | 已参数化 |
| `test_lift_unlift.py` | 0 | 无耦合 |
| `test_nativert.py` | 12 | P2 可改（`"cuda"` 硬编码循环） |
| `test_package.py` | 12 | 混合（AOTI delegate 相关） |
| `test_passes.py` | 31 | 混合（`move_to_device_pass` 测 cuda 目标） |
| `test_pass_infra.py` | 0 | 无耦合 |
| `test_retraceability.py` | 0 | 无耦合 |
| `test_schema.py` | 0 | 无耦合 |
| `test_serdes.py` | 0 | 无耦合 |
| `test_serialize.py` | 20 | P1/P2 混合 |
| `test_sparse.py` | 0 | 无耦合 |
| `test_strict_export_v2.py` | 0 | 无耦合 |
| `test_swap.py` | 0 | 无耦合 |
| `test_tools.py` | 0 | 无耦合 |
| `test_unflatten.py` | 0 | 无耦合 |
| `test_unflatten_training_ir.py` | 0 | 无耦合 |
| `test_upgrader.py` | 0 | 无耦合 |
| `test_verifier.py` | 0 | 无耦合 |
| `test_model_exports_to_core_aten.py` | 0 | 无耦合 |
| `test_prims.py` | 10 | 已参数化 |
| `testing.py` | 0 | 无耦合（测试辅助） |

### 1.2 AOTI（7 个）

| 文件 | 耦合命中 | 分类 |
|------|----------|------|
| `test_aot_inductor.py` | 457 | **基础设施已解耦**；内含 CUDA 专属子测试 |
| `test_aot_inductor_arrayref.py` | 30 | 继承模板，`self.device` 已参数化 |
| `test_aot_inductor_custom_ops.py` | 72 | 已用 `GPU_TYPE` |
| `test_aot_inductor_package.py` | 86 | 已用 `GPU_TYPE` |
| `test_aot_inductor_utils.py` | 12 | 辅助/基类，无直接设备硬编码 |
| `test_aoti_cross_compile_windows.py` | 22 | 平台专属（非纯 CUDA） |
| `test_aoti_torchbind_constants.py` | 10 | P1 可改 |

### 1.3 统计

| 分类 | Export | AOTI | 合计 |
|------|--------|------|------|
| 无耦合 | 22 | 1 | 23 |
| 已参数化 / 基础设施已解耦 | 3 | 5 | 8 |
| CUDA/平台专属（不改） | — | — | 见第 3 节 |
| 需要改造 | 6 | 1 | 7 |

---

## 2. Export / AOTI 与 Dynamo 的差异

| 维度 | Dynamo | Export | AOTI |
|------|--------|--------|------|
| 测试焦点 | bytecode 追踪、guard | FX 图导出、序列化 | C++ 编译产物、runtime |
| 主流参数化 | `instantiate_device_type_tests` | 同上 + 模块级 `device_type` | `GPU_TYPE` + `copy_tests` 模板 |
| 多卡 skip | 无 | 无 | 无 |
| 天然 GPU 绑定 | 低 | 中（flex_attention 等） | **高**（Triton kernel → C++） |

**AOTI 特别说明**：`test_aot_inductor.py` 通过 `AOTInductorTestsTemplate` + `copy_tests(..., GPU_TYPE)` 已将 **绝大多数** 测试绑定到 `GPU_TYPE`。**但 `GPU_TYPE` 不是「任意加速器」的抽象**，见 [第 10 节](#10-gpu_type-与-npu-等设备的真实边界)。

```python
# torch/_inductor/utils.py
GPU_TYPES = ["cuda", "mps", "xpu", "mtia"]  # 不含 npu / hpu / privateuse1

def get_gpu_type() -> str:
    avail_gpus = [x for x in GPU_TYPES if getattr(torch, x).is_available()]
    gpu_type = "cuda" if len(avail_gpus) == 0 else avail_gpus.pop()  # 无可用时回退 "cuda"
    return gpu_type

# torch/testing/_internal/inductor_utils.py
HAS_GPU = HAS_CUDA_AND_TRITON or HAS_XPU_AND_TRITON  # 仅 CUDA+XPU 且需 Triton
GPU_TYPE = get_gpu_type()
```

```python
# test/inductor/test_aot_inductor.py L9215–9230 — 已解耦的 GPU 测试入口
@unittest.skipIf(sys.platform == "darwin", "No CUDA on MacOS")
class AOTInductorTestABICompatibleGpu(TestCase):
    device = GPU_TYPE
    device_type = GPU_TYPE
    ...

copy_tests(AOTInductorTestsTemplate, AOTInductorTestABICompatibleGpu, GPU_TYPE, GPU_TEST_FAILURES)
```

---

## 3. CUDA/GPU 专属判断（Export / AOTI 特化版）

通用判断流程见 [dynamo-test-decoupling-plan.md 第 3 节](dynamo-test-decoupling-plan.md#3-cudagpu-专属判断指南详细)。本节补充 **Export/AOTI 场景下的专属信号**。

### 3.1 一句话原则

> **测的是「导出到 CUDA 图 IR」「CUDA 驱动/runtime」「CUDA 专属算子路径」→ 专属。  
> 只是在 GPU 上跑 export/AOTI 编译流程，设备名写死 → 可泛化。**

### 3.2 Export 专属信号表

| 信号 | 为何专属 | 文件/位置 |
|------|----------|-----------|
| `flex_attention` / `BlockMask` / `create_block_mask` + `IS_FLEX_ATTENTION_CUDA_PLATFORM_SUPPORTED` | FlexAttention 当前 CUDA 平台实现 | `test_experimental.py` |
| `move_to_device_pass(ep, "cuda")` + golden string 含 `device = 'cuda'` | 测的是 **搬到 cuda 设备** 的 pass 输出格式 | `test_passes.py` L1327+ |
| `TestExportOnFakeCuda` + `only_for="cuda"` + `CUDA_VISIBLE_DEVICES=""` | 模拟「有 cuda build 但无 GPU」的 FakeTensor 场景 | `test_export_opinfo.py` |
| golden string / `assertExpectedInline` 含 `'cuda'` 且断言图 IR | IR 文本格式绑定设备名 | `test_passes.py`、`test_after_aot` 类 repro |
| `test_cuda_memory_usage` + `torch.cuda.memory_stats` peak | CUDA 显存峰值监控 | `test_draft_export.py` L770+ |
| `package_nativert_with_aoti_delegate` + fbcode only | 集成路径，非通用开源 CI | `test_nativert.py` |
| `requires_cuda_and_triton` | Triton 路径当前 CUDA 为主 | `test_export.py`、`test_serialize.py` |
| `torch.cuda.get_device_capability()` 作为 **测试逻辑** | 测 export 对 device capability 的追踪 | `test_export_opinfo.py` L247 |

### 3.3 AOTI 专属信号表

| 信号 | 为何专属 | 文件/位置 |
|------|----------|-----------|
| `self.device != "cuda"` + CUmodule / `loaded_modules_` | CUDA/HIP 驱动层 kernel 泄漏检测 | `test_aot_inductor.py` L292 |
| `torch.cuda.CUDAGraph()` + AOTI runtime | CUDAGraph 与 AOTI 多线程冲突 | `test_aot_inductor.py` L8257+ |
| `torch.cuda.memory_allocated()` 断言 | CUDA 显存泄漏检测 | `test_aot_inductor.py` L3992+ |
| `GPU_TYPE != "cuda"` / `TEST_WITH_ROCM` skip | CUDA PTX / nvcc 专属 codegen | `test_aot_inductor.py` L416+ |
| `@requires_triton_ptxas_compat` | ptxas 版本与 Triton 兼容性 | `test_aot_inductor.py` |
| `SM80OrLater` / `SM90OrLater` / `PLATFORM_SUPPORTS_FP8` | NVIDIA 架构能力门控 | `test_aot_inductor.py` |
| `PLATFORM_SUPPORTS_FLASH_ATTENTION` | Flash SDPA 硬件支持 | `test_aot_inductor.py` |
| `aot_inductor.cross_target_platform": "windows"` | 交叉编译到 Windows（平台专属） | `test_aoti_cross_compile_windows.py` |
| `TRITON_PTXAS_VERSION` / embed_kernel_binary CUBIN | CUDA 二进制嵌入 | `test_aot_inductor_package.py` L379 |
| `@skipCUDAIf(True, "Test for x86 backend")` | x86 CPU 后端专属 | `test_aot_inductor.py` L3364 |
| `launchKernel(` / `aoti_torch_mps_get_kernel_function(` FileCheck | 按 **实际 device** 检查 codegen，已参数化 | 模板内 `if self.device == ...` |

### 3.4 可泛化信号（Export / AOTI 常见误判）

| 看起来像专属 | 实际可改 | 改法 |
|-------------|----------|------|
| `device="cuda"` 在普通 `export(model, (tensor,))` | 是 | `device=GPU_TYPE` 或 `device_type` |
| `@unittest.skipIf(not TEST_CUDA)` 且方法体只有普通 tensor op | 是 | `torch.accelerator.is_available()` |
| `test_nativert.py` 中 `for device in ["cpu", "cuda"]` | 是 | `["cpu", GPU_TYPE]` + `HAS_GPU` 判断 |
| `device = "cuda" if HAS_CUDA else "cpu"` | 是 | `device = GPU_TYPE if HAS_GPU else "cpu"` |
| `test_aot_inductor.py` 里 175 处 `GPU_TYPE` | **已解耦** | 不改 |
| `aot_inductor.*` 配置字符串 | **不是设备名** | 不改 |
| `self.device` 来自 `copy_tests` 模板 | **已解耦** | 不改 |

---

## 4. 文件分类详情

### 4.1 无耦合 — 无需改动（23 个）

**Export（22 个）**：

```
test_cpp_serdes.py, test_db.py, test_dynamic_shapes.py, test_export_strict.py,
test_export_training_ir_to_run_decomp.py, test_functionalized_assertions.py,
test_lift_unlift.py, test_pass_infra.py, test_retraceability.py, test_schema.py,
test_serdes.py, test_sparse.py, test_strict_export_v2.py, test_swap.py,
test_tools.py, test_unflatten.py, test_unflatten_training_ir.py, test_upgrader.py,
test_verifier.py, test_model_exports_to_core_aten.py, testing.py
```

**AOTI（1 个）**：`test_aot_inductor_utils.py`（runner 辅助，无直接 `device="cuda"`）

---

### 4.2 已参数化 / 基础设施已解耦（8 个）

| 文件 | 机制 | 处理 |
|------|------|------|
| `test_export.py` | 模块级 `device_type`；大量 `GPU_TYPE`；部分 `instantiate_device_type_tests` | 仅清理零星 `device="cuda"`（约 7 处） |
| `test_hop.py` | `instantiate_device_type_tests` | 保持 |
| `test_prims.py` | `instantiate_device_type_tests` | 保持 |
| `test_export_opinfo.py` | `TestExportOpInfo` 仅 CPU；`TestExportOnFakeCuda` 仅 CUDA | CPU 部分保持；FakeCuda 部分专属 |
| `test_aot_inductor.py` | `AOTInductorTestsTemplate` + `GPU_TYPE`/`cpu`/`mps` 三套 `copy_tests` | 保持框架；仅清理模板内零星硬编码 |
| `test_aot_inductor_arrayref.py` | 继承 `AOTInductorTestsTemplate`，`self.device` | 保持 |
| `test_aot_inductor_custom_ops.py` | `GPU_TYPE`、`HAS_GPU_AND_TRITON` | 保持 |
| `test_aot_inductor_package.py` | `GPU_TYPE`、`@parametrize("device", ...)` | 保持 |

---

### 4.3 CUDA / 平台专属 — 保持不动

#### Export

| 文件 | 专属范围 | 理由 |
|------|----------|------|
| `test_experimental.py` | FlexAttention / BlockMask 相关测试（`IS_FLEX_ATTENTION_CUDA_PLATFORM_SUPPORTED`） | FlexAttention CUDA 实现；含 `not torch.version.hip` |
| `test_passes.py` | `test_move_device_to`、`test_move_device_submod` 等（L1327+） | 断言 `move_to_device_pass(ep, "cuda")` 的 IR 含 `'cuda'` |
| `test_export_opinfo.py` | `TestExportOnFakeCuda` 整个类 | `only_for="cuda"`；测 FakeTensor on fake cuda:0 |
| `test_draft_export.py` | `test_cuda_memory_usage`（L770+） | `torch.cuda.memory_stats` peak 监控 |
| `test_package.py` | CUDA 不可用断言相关用例 | 测「无 CUDA 环境」行为 |
| `test_export.py` | `@requires_cuda_and_triton` 方法；`test_module_to_with_shared_weights` 等含 `"cuda" in str(x.device)` 逻辑 | Triton / 设备字符串分支语义 |
| `test_serialize.py` | `@requires_cuda_and_triton`；部分 `.cuda()` 序列化路径 | Triton + CUDA 序列化 |
| `test_nativert.py` | fbcode-only 测试类 | 开源 CI 不跑，暂不改造 |

#### AOTI

| 文件 | 专属范围 | 理由 |
|------|----------|------|
| `test_aot_inductor.py` | `test_loaded_modules_tracking`（`self.device != "cuda"`） | CUmodule 泄漏 |
| `test_aot_inductor.py` | CUDAGraph + AOTI 测试（L8257+） | `torch.cuda.CUDAGraph` |
| `test_aot_inductor.py` | FP8 / SM80 / SM90 / FlashAttention 门控测试 | 硬件架构能力 |
| `test_aot_inductor.py` | `GPU_TYPE != "cuda"` / ROCm skip 的 codegen 测试 | PTX/nvcc 路径 |
| `test_aot_inductor.py` | `torch.cuda.memory_allocated` 泄漏测试 | CUDA 显存 API |
| `test_aot_inductor_package.py` | `TRITON_PTXAS_VERSION` / CUBIN embed | CUDA 工具链版本 |
| `test_aoti_cross_compile_windows.py` | **全文件** | Linux→Windows 交叉编译，平台专属 |

---

### 4.4 需要改造（7 个）

| 优先级 | 文件 | 改动点 | 类型 |
|--------|------|--------|------|
| P1 | `test_converter.py` | `device="cuda:0"` → `GPU_TYPE`；`requires_cuda` → `accelerator`（若语义允许） | 类型三 |
| P1 | `test_aoti_torchbind_constants.py` | `"cuda" if HAS_CUDA` → `GPU_TYPE if HAS_GPU` | 类型三 |
| P1 | `test_getitem` 同类：`test_serialize.py` 零星 | `device="cuda"` / `.cuda()` → `GPU_TYPE` | 类型三 |
| P2 | `test_experimental.py` | 非 FlexAttention 测试中的 `device="cuda"`（约 10+ 处） | 类型三 |
| P2 | `test_nativert.py` | `for device in ["cpu", "cuda"]` → `["cpu", GPU_TYPE]` | 类型三 |
| P2 | `test_export.py` | 清理剩余 `device="cuda"`（约 7 处）；`@requires_cuda_and_triton` 不改 | 类型三 |
| P2 | `test_draft_export.py` | `test_missing_meta_kernel_guard` 中 `device=torch.device("cuda")` 用于 mismatch 测试 → 可用 `GPU_TYPE` | 类型三 |

---

## 5. 逐文件改造示例

### 5.1 `test_converter.py`

```python
# 改前
requires_cuda = unittest.skipUnless(torch.cuda.is_available(), "requires cuda")
inp = (torch.rand((3, 4), device="cuda:0"),)

# 改后
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
requires_accelerator = unittest.skipUnless(torch.accelerator.is_available(), "requires accelerator")
inp = (torch.rand((3, 4), device=GPU_TYPE),)
```

### 5.2 `test_aoti_torchbind_constants.py`

```python
# 改前
HAS_CUDA = torch.cuda.is_available()
device = "cuda" if HAS_CUDA else "cpu"

# 改后
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
device = GPU_TYPE if HAS_GPU else "cpu"
```

### 5.3 `test_nativert.py`

```python
# 改前
for device in ["cpu", "cuda"]:
    if device == "cuda" and not HAS_CUDA_AND_TRITON:
        continue

# 改后
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU_AND_TRITON
for device in ["cpu", GPU_TYPE]:
    if device == GPU_TYPE and not HAS_GPU_AND_TRITON:
        continue
```

### 5.4 `test_experimental.py`（仅非 FlexAttention 部分）

```python
# 文件顶部
device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"

# 改前（普通 blockmask 测试，非 flex_attention CUDA 平台门控）
x = torch.randn(2, 128, device="cuda")
@unittest.skipIf(not TEST_CUDA, "CUDA not available")

# 改后
x = torch.randn(2, 128, device=device_type)
@unittest.skipIf(not torch.accelerator.is_available(), "requires accelerator")
```

**不改**：带 `IS_FLEX_ATTENTION_CUDA_PLATFORM_SUPPORTED` 的 `test_aot_export_flex_attention_*`。

### 5.5 `test_passes.py` — 为何整体不改

```python
# 专属：测 move_to_device_pass 搬到 cuda 后的 IR
ep = move_to_device_pass(ep, "cuda")
self.assertExpectedInline(..., """device = 'cuda' ...""")
```

此处 `"cuda"` 是 **pass 的目标设备参数** 和 **期望 IR 输出**，不是「随便哪个 GPU」。若改为 `device_type`，需同时改 pass 语义和 golden string，且该 pass 本身可能只支持 cuda 目标。

---

## 6. AOTI 解耦现状说明（重点）

`test_aot_inductor.py` 是 AOTI 测试的核心，**不需要大规模重写**。其架构：

```
AOTInductorTestsTemplate          # 所有测试方法用 self.device
    ├── copy_tests → AOTInductorTestABICompatibleCpu   (device="cpu")
    ├── copy_tests → AOTInductorTestABICompatibleGpu   (device=GPU_TYPE)  ← 自动跟平台走
    ├── copy_tests → AOTInductorTestDualWrapper        (device=GPU_TYPE)
    └── copy_tests → AOTInductorTestABICompatibleMps   (device="mps")
```

**模板内按设备分支是正确设计**，不是耦合：

```python
if self.device == "mps":
    FileCheck().check("aoti_torch_mps_get_kernel_function(").run(code)
elif self.device == GPU_TYPE:
    FileCheck().check("launchKernel(").run(code)
```

**仅需关注的零星硬编码**（模板内，约 <10 处）：

| 模式 | 处理 |
|------|------|
| `if self.device != "cuda": raise SkipTest` | **保留** — 测 CUDA driver |
| `torch.cuda.memory_allocated()` | **保留** — CUDA API |
| `device="cuda"` 在个别测试方法内 | 改为 `device=self.device` 或 `GPU_TYPE` |

`test_aot_inductor_arrayref.py`、`test_aot_inductor_custom_ops.py`、`test_aot_inductor_package.py` 均继承或复用上述模板，**无独立解耦工作**。

---

## 7. 改造优先级

```
P0  无耦合 23 个 + 已参数化 8 个 + CUDA/平台专属段 → 不改
 |
P1  test_converter.py, test_aoti_torchbind_constants.py, test_serialize.py（零星）
 |
P2  test_experimental.py（非 Flex 部分）, test_nativert.py, test_export.py（零星）, test_draft_export.py（1 处）
 |
完成（AOTI 主文件仅做零星清理，不重写模板）
```

### 建议 Test Plan

```bash
# Export P1
python test/export/test_converter.py
python test/export/test_serialize.py
python test/inductor/test_aoti_torchbind_constants.py

# Export P2
python test/export/test_experimental.py
python test/export/test_nativert.py
python test/export/test_export.py
python test/export/test_draft_export.py

# AOTI（改后回归 GPU 模板）
python test/inductor/test_aot_inductor.py -k AOTInductorTestABICompatibleGpu
python test/inductor/test_aot_inductor_arrayref.py
python test/inductor/test_aot_inductor_custom_ops.py
python test/inductor/test_aot_inductor_package.py
```

---

## 8. 与 Dynamo 文档的对照

| 主题 | Dynamo 文档 | 本文档 |
|------|-------------|--------|
| 无耦合占比 | ~62% | Export ~71%（22/31） |
| 主战场 | `device="cuda"` 硬编码 | Export 同上；AOTI 已用 `GPU_TYPE` |
| CUDA 专属最大头 | `test_cudagraphs.py` | `test_aot_inductor.py` 内 FP8/CUDAGraph/驱动测试 |
| 特殊专属 | NCCL 日志 | `move_to_device_pass(..., "cuda")` golden IR |
| 不建议改 | CUDA Stream 遗留 API | FlexAttention、FakeCuda、cross_compile_windows |

---

## 9. 自检清单（Export / AOTI 版）

准备修改前，**任一项为「是」则先停下来确认**：

- [ ] 是否测 `flex_attention` / `BlockMask` / `IS_FLEX_ATTENTION_CUDA_PLATFORM_SUPPORTED`？
- [ ] golden string / FileCheck 是否断言 IR 或 C++ 代码中含字面量 `'cuda'`？
- [ ] 是否调用 `move_to_device_pass(..., "cuda")` 或类似「目标设备=cuda」的 pass？
- [ ] 是否 `TestExportOnFakeCuda` / `only_for="cuda"`？
- [ ] 是否 `torch.cuda.CUDAGraph` / `memory_allocated` / CUmodule？
- [ ] 是否 `SM80`/`SM90`/`FP8`/`PLATFORM_SUPPORTS_FLASH_ATTENTION` 门控？
- [ ] 是否 `cross_target_platform: windows`？
- [ ] 方法是否在 `AOTInductorTestsTemplate` 内且已用 `self.device`？（是 → 通常不改）

全部「否」且仅有 `device="cuda"` → 放心改为 `GPU_TYPE` / `device_type`。

---

## 10. `GPU_TYPE` 与 NPU 等设备的真实边界

### 10.1 直接回答：NPU 上 `GPU_TYPE` 是什么？

**上游 PyTorch 开源代码中，NPU 不在 `GPU_TYPES` 列表里。**

| 环境 | `GPU_TYPE` 取值 | `HAS_GPU` | 实际行为 |
|------|-----------------|-----------|----------|
| 仅 CUDA + Triton | `"cuda"` | `True` | GPU 模板测试运行 |
| 仅 XPU + Triton | `"xpu"` | `True` | GPU 模板测试运行 |
| 仅 MPS | `"mps"` | `False` | 走独立 `AOTInductorTestABICompatibleMps` 模板 |
| 仅 MTIA | `"mtia"` | `False` | `GPU_TYPE` 为 `"mtia"`，但 `HAS_GPU` 仍为 False（需 CUDA/XPU+Triton） |
| **仅 NPU（privateuse1 注册为 `npu`）** | **`"cuda"`（回退默认值）** | **`False`** | **GPU 模板整体 skip；用 `GPU_TYPE` 会指向错误设备** |
| 无加速器 | `"cuda"`（回退） | `False` | 测试 skip 或误用 cpu/cuda 字符串 |

NPU 在 PyTorch 中通常通过 `privateuse1` 后端扩展（`rename_privateuse1_backend("npu")`），与 `torch.cuda` / `torch.xpu` 并列的第一公民模块 **`torch.npu` 在上游不存在**。

### 10.2 「设备无关」实际分三层

```
第 1 层：字符串去硬编码（device="cuda" → GPU_TYPE / device_type）
         ↓ 仅保证「在已支持的加速器集合内」不写死 cuda
第 2 层：测试基础设施识别设备（HAS_GPU、instantiate_device_type_tests、torch.accelerator）
         ↓ 决定测试是否被调度到该设备
第 3 层：Inductor/AOTI 后端是否实现（register_backend_for_device、Triton、C++ wrapper）
         ↓ 决定测试能否真正编译运行
```

**解耦改造主要做第 1 层；能否在 NPU 上跑取决于第 2、3 层是否就绪。**

### 10.3 各类文件在 NPU 上的真实可达性

| 类别 | 文件示例 | NPU 上能否跑 | 原因 |
|------|----------|-------------|------|
| 纯 CPU Export | `test_schema.py`、`test_verifier.py` 等 22 个 | **能** | 不涉及加速器 |
| Export + `instantiate_device_type_tests` | `test_hop.py`、`test_prims.py` | **可能** | 需 `PrivateUse1TestBase` 注册 + `PYTORCH_TESTING_DEVICE_FOR_CUSTOM` |
| Export + `GPU_TYPE` / `device_type` | `test_export.py` 部分 | **部分** | 纯 export 不经过 Inductor 的用例可以；`@requires_cuda_and_triton` 不行 |
| Export CUDA 专属 | FlexAttention、`move_to_device_pass(cuda)` | **不能** | 语义绑定 CUDA |
| AOTI `copy_tests(..., GPU_TYPE)` | `test_aot_inductor.py` 主体 | **不能**（默认） | `HAS_GPU=False`；且需 NPU 的 Inductor Scheduling + CppWrapper |
| AOTI CPU/MPS 模板 | `AOTInductorTestABICompatibleCpu/Mps` | CPU 能；MPS 需 Mac | 与 NPU 无关 |
| AOTI CUDA 驱动专属 | CUDAGraph、CUmodule、FP8 | **不能** | CUDA 特有 |
| AOTI cross_compile_windows | `test_aoti_cross_compile_windows.py` | **不能** | 平台专属 |

### 10.4 NPU 要能跑这些测试，需要什么（超出「改字符串」）

1. **测试基础设施**
   - 扩展 `GPU_TYPES` 或新增 `ACCELERATOR_TYPE`，纳入 `npu`
   - 或新增 `HAS_NPU_AND_TRITON` / `requires_npu` 等等价物
   - `get_gpu_type()` 在无 cuda/xpu/mps/mtia 时不应静默回退 `"cuda"`

2. **Inductor 后端注册**（`torch/_inductor/codegen/common.py` 已有扩展点）

```python
# privateuse1 改名后，若厂商提供 Scheduling / PythonWrapperCodegen / CppWrapperCodegen
register_backend_for_device(private_backend, device_scheduling, wrapper_codegen, cpp_wrapper_codegen, ...)
```

3. **AOTI 测试模板**
   - 新增 `copy_tests(..., "npu", NPU_TEST_FAILURES)` 或让 `GPU_TYPE` 覆盖 npu
   - 为 NPU 维护独立的 `GPU_TEST_FAILURES` / xfail 表

4. **Export 测试**
   - 对不经过 Inductor 的用例：用 `torch.accelerator.current_accelerator().type` 比 `GPU_TYPE` 更准
   - 对 Triton 用例：需 NPU 版 `requires_*_and_triton`

### 10.5 结论：除了 GPU 专属，能否「真正做到设备无关」？

| 子集 | 能否设备无关 | 说明 |
|------|-------------|------|
| Export 纯 CPU 测试（22 个） | **已经是** | 与设备无关 |
| Export 可改造段（`device="cuda"` → 动态） | **在 cuda/xpu 间是** | 跨到 NPU 需第 2 层基础设施 |
| Export CUDA 专属段 | **故意不是** | 不应泛化 |
| AOTI 模板主体（`self.device`） | **在 cuda/xpu 间是** | 架构上已参数化 |
| AOTI 整体在 NPU 上 | **当前不是** | 缺 HAS_GPU、后端注册、测试模板 |
| 用 `torch.accelerator` 替代 `GPU_TYPE` | **更接近「任意加速器」** | 但仍需 Inductor/AOTI 后端支持才能跑通 |

**务实结论**：

- 文档中的「可泛化」= **消除 cuda 硬编码，使测试跟随 PyTorch 已纳入 CI 的加速器集合（CUDA、XPU、MPS、MTIA）**，不是「写一次在所有未来硬件上自动通过」。
- **NPU 上要跑 Export/AOTI 测试**，需要厂商侧完成 Inductor 后端 + 测试基础设施扩展；仅把 `device="cuda"` 改成 `GPU_TYPE` **在纯 NPU 机器上不够**，因为 `GPU_TYPE` 会错误地变成 `"cuda"` 且 `HAS_GPU` 为 False。
- 若 NPU 环境**同时有 CUDA**，改 `GPU_TYPE` 的测试会在 CUDA 上跑，**不会在 NPU 上跑**——这是当前设计的有意行为，不是解耦遗漏。

### 10.6 NPU 场景的推荐做法

```python
# 比 GPU_TYPE 更通用（第 2 层），但仍需后端支持（第 3 层）
device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"

@unittest.skipIf(not torch.accelerator.is_available(), "requires accelerator")
def test_foo(self):
    x = torch.randn(4, device=device_type)
    ep = torch.export.export(model, (x,))
```

```python
# NPU 专用测试入口（需厂商自建，类似 AOTInductorTestABICompatibleGpu）
class AOTInductorTestABICompatibleNpu(TestCase):
    device = "npu"  # 或 torch.accelerator.current_accelerator().type
    ...

copy_tests(AOTInductorTestsTemplate, AOTInductorTestABICompatibleNpu, "npu", NPU_TEST_FAILURES)
```

**不建议**在 NPU 未纳入 `GPU_TYPES` / `HAS_GPU` 前，仅做字符串替换并声称「已设备无关」。
