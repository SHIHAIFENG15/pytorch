---
name: npu-device-agnostic-export-aoti
description: >-
  Export/AOTI device-agnostic TESTS for NPU (privateuse1). Use when editing
  Export or AOTI test files for accelerator decoupling, GPU_TYPE/HAS_GPU/requires_*
  test gates, instantiate_device_type_tests, or test-side has_triton skip.
  Remembers which files must wait for means 2/3. Do not hardcode
  torch_npu._inductor in tests. Not for Dynamo/Inductor DeviceInterface source
  (RFC #189138) — that is rfc-189138-compile-device-interface.
---

# NPU Export/AOTI Device-Agnostic Tests

**Tests only.** Dynamo/Inductor `DeviceInterface` / `GPU_TYPES` / `has_triton()`
source patches belong to `rfc-189138-compile-device-interface` (RFC #189138).
This skill consumes those predicates in test files; it does not implement them.

## Hard rules (do not forget)

1. **`ce628a0` / shallow `GPU_TYPE` PRs are not the final scheme.** Prefer local
   `instantiate_device_type_tests` + injected `device` for Export cases that can
   generalize without Triton/`HAS_GPU`.
2. **Do not revive shallow `GPU_TYPE` / `@requires_gpu` as the NPU/device
   entry path.** Prefer instantiate + injected `device`. Exception follow-up:
   `test_aoti_torchbind_constants.py` already uses instantiate; after means 2/3
   only swap the Triton helper to shared decorators — still do not pick device
   via `GPU_TYPE`/`HAS_GPU`. Main AOTI suite (`test_aot_inductor*.py`) waits
   for means 2 + `copy_tests`.
3. **Do not put designer nicknames** ("方案 N", "手段 N") in code comments.
4. Prefer `except_for="cpu"` for accelerator-only instantiate classes; use
   in-body `skipTest` for Triton/capability gates.

## Means cheat sheet

| # | What | Status for NPU work |
|---|------|---------------------|
| 1 | PrivateUse1 in `instantiate_device_type_tests` | Ready when `import torch_npu` + `PYTORCH_TESTING_DEVICE_ONLY_FOR=npu` |
| 2 | `GPU_TYPE` / `HAS_GPU` / Triton include privateuse1; no silent `"cuda"` fallback | Infra in flight: [#190324](https://github.com/pytorch/pytorch/pull/190324) `has_triton()` via DeviceInterface registry; [#190326](https://github.com/pytorch/pytorch/pull/190326) `is_gpu()`/`device_need_guard()` via DeviceInterface. Full `GPU_TYPE`/`HAS_GPU` test-gate follow-ups still open |
| 3 | Rewrite `requires_cuda_and_triton` → generic `requires_gpu` / accelerator+triton | **Blocked on means 2** naming/semantics |
| 5 | Replace hardcoded `device="cuda"` / `.cuda()` | OK only with instantiate/`device`, or after means 2 for `GPU_TYPE` paths |
| 6 | AOTI `legacy_load_runner` dynamic by device name; never else→Cuda | Done in `test_aot_inductor_utils.py` |
| 7 | FlexAttention | **Not landing infra** in this cont (keep Export-only instantiate). Export Flex/BlockMask Done via instantiate + `device`; do not touch `common_device_type` / inductor flex suite for now |

Without means 2, upstream still effectively:

```python
HAS_GPU = HAS_CUDA_AND_TRITON or HAS_XPU_AND_TRITON  # NPU not included
GPU_TYPE  # often stays "cuda" on pure NPU
```

So `@requires_gpu` / module-level `HAS_GPU` / `GPU_TYPE` **must not** be treated as the NPU entry path.

## Local verification (copy PRs onto pin torch; do not upstream)

Pinned torch used by venvs: `/home/y00839695/npu-pytorch-setup/pytorch/`.

| Artifact | Role |
|----------|------|
| `agent_space/pr_patches/190324.diff` / `190326.diff` | Saved upstream diffs |
| Pin tree edits to `_triton.py` / `device_interface.py` / `inductor/utils.py` | Runtime half of #190324/#190326 applied locally |
| `agent_space/local_device_interface_pr_compat.py` | NPU `is_triton_capable`/`is_gpu` + register without `_inductor` |
| `agent_space/local_npu_test_runner.py` + `run_test.sh` NPU path | Loads compat before tests |

After this, upstream tests gate with `has_triton()` only — **no** hardcoded
`import torch_npu._inductor` in test bodies.

## Triton / GPU classification: community PRs (source, not this skill)

Implementing these diffs is `rfc-189138-compile-device-interface`. This skill
only cares that tests **consume** the predicates and do not monkeypatch them.

| PR | What | Relation |
|----|------|----------|
| [#190324](https://github.com/pytorch/pytorch/pull/190324) | `has_triton()` loops `get_registered_device_interfaces()` instead of a hardcoded device dict | PrivateUse1 Triton visible once DeviceInterface is registered — **no** `import torch_npu._inductor` monkeypatch |
| [#190326](https://github.com/pytorch/pytorch/pull/190326) | `is_gpu()` / `device_need_guard()` from DeviceInterface (`is_gpu()`, `exposes_streams()`) | NPU classified as GPU without mutating `GPU_TYPES` |

Verify merge status on GitHub. Follow-ups still needed for full **test-gate**
means 2/3 (`GPU_TYPE`/`HAS_GPU`/`@requires_gpu` / `requires_*_and_triton` rewrite).

**Hard rule for upstream tests:** do **not** hardcode `import torch_npu._inductor`
(or other out-of-tree backend modules). Gate with:

```python
from torch.utils._triton import has_triton, has_triton_package
if not has_triton_package() or not has_triton():
    self.skipTest("requires triton")
```

Backend registration belongs to the out-of-tree package / DeviceInterface
contract (and test harness `import torch_npu` when running NPU), not test bodies.

## Triton: layers (do not conflate)

```
1) pip install triton-ascend                 # package on disk
2) DeviceInterface registered (e.g. import torch_npu) + #190324 has_triton()
3) test gates: HAS_GPU / @requires_gpu / requires_*_and_triton (means 2/3)
```

- **Avoid:** per-test `import torch_npu._inductor` to monkeypatch `has_triton`.
- Without merged #190324, NPU may still see `has_triton()==False` even with
  triton-ascend + registered `npu` interface (current upstream hardcoded dict).
- Do **not** rely on module-level `HAS_TRITON`/`HAS_GPU` as the NPU entry path
  until means 2 lands.
- Shared runtime gate: `torch.testing._internal.inductor_utils.ensure_triton`
  (not per-file `_ensure_triton_*` copies). Use `required_on_cpu=False` for
  AOTI tests that include CPU.

## Wait vs can-do-now (with instantiate + has_triton)

**Literal means 3** (`@requires_cuda_*` → `@requires_gpu`) still fails on NPU without
means 2 (`HAS_GPU` stays False).
**Practical substitute:** instantiate + `device` + in-body `has_triton()` /
`skipTest` (not module-level `HAS_GPU` / `@requires_gpu`, and **not**
hardcoded `torch_npu._inductor`).

### Can proceed NOW (no means 2): Export only

| File | How |
|------|-----|
| `test/export/test_export.py` | Split done: `TestExportAccelerator` / `TestExportRNN` / `TestExportTriton` / `TestExportFlexAttention`. Keep CUDA-only raw triton / shared-weights goldens. |
| `test/export/test_serialize.py` | Split: `TestSerializeAccelerator`, `TestDeserializeAccelerator`, `TestSaveLoadAccelerator`, `TestSerializeTriton` with same gating. |

Prior shallow `GPU_TYPE`/`@requires_gpu` edits on these were reverted; use instantiate
path instead, not revive GPU_TYPE.

### AOTI: instantiate OK for small files; main suite still waits

| File | Status |
|------|--------|
| `test/inductor/test_aoti_torchbind_constants.py` | **Done via means 4 + local Triton gate** (`instantiate` + `device`). Do **not** use `GPU_TYPE`/`HAS_GPU` to pick device. After RFC #189138 (means 2/3), replace `_ensure_triton_for_accelerator` with shared `requires_*` decorators — keep instantiate/`device` as the device path. |
| `test/inductor/test_aot_inductor.py` | **WAIT** — designer: `copy_tests` + means 2; privateuse1 entry |
| `test/inductor/test_aot_inductor_custom_ops.py` | **WAIT** — same |
| `test/inductor/test_aot_inductor_package.py` | **WAIT** — `GPU_TYPE` gates; keep PTXAS/CUBIN |
| `test/inductor/test_aot_inductor_arrayref.py` | **Keep** — CPU ArrayRef path; not NPU device-decouple target |
| `test/inductor/test_aoti_cache_dir.py` | **Out of scope** — not in current NPU export/AOTI decouple plan |
| `test/inductor/test_aoti_cross_compile_windows.py` | **Keep CUDA/Windows-only** — Linux→Windows AOTI cross-compile; `GPU_TYPE`/`@requires_gpu`; MinGW `cudart` / `WINDOWS_CUDA_HOME`. Not privateuse1 work. |

Also need NPU `AOTIModelContainerRunner*` / Inductor backend for real AOTI runs on NPU.

## Files OK without means 2/3 (instantiate path)

Keep / extend this style only:

| File | Notes |
|------|-------|
| `test/export/test_db.py` | Full case×device instantiate; ExportDB examples device-safe |
| `test/export/test_draft_export.py` | Device-guard class + instantiate; keep CUDA memory test CUDA-only |
| `test/export/test_converter.py` | Small device class + instantiate |
| `test/export/test_passes.py` | `TestMoveToDevicePass` + instantiate; CPU `skipTest` |
| `test/export/test_nativert.py` | AOTI-related split + instantiate (not GPU_TYPE loop) |
| `test/export/test_experimental.py` | Non-Flex annotate_overlap done. **Flex/BlockMask:** `TestExperimentFlex` + instantiate. CUDA-SM `test_aot_export_flex_attention_*` remain gated. |
| `test/inductor/test_aot_inductor_utils.py` | Means 6 dynamic runner (explicit fail if no Npu runner) |
| `test/inductor/test_aoti_torchbind_constants.py` | instantiate + `device`; in-body Triton for non-CPU; **follow-up: swap helper → means 2/3 decorators after RFC** |
| `torch/_export/db/examples/optional_input.py` | Device-safe tensors |
| `torch/_export/db/examples/static_if.py` | Device-safe tensors |

## Keep CUDA-specific (never "generalize")

FakeCuda (`test_export_opinfo` `only_for="cuda"`), CUDA memory / PTXAS / CUBIN /
CUDAGraph, raw CUDA Triton metadata maps, `aoti_torch_cuda_*` shim names,
`test_aot_inductor_arrayref.py` (CPU ArrayRef),
`test_aoti_cross_compile_windows.py` (Windows+CUDA cross-compile / MinGW cudart),
`test_model_exports_to_core_aten.py` (no device coupling).

## FlexAttention

### Infra (方案7 registry / `skip_device_if`) — **reverted / not in cont**

Prefer only changing Export/AOTI **test cases**. Do not land `_flex_attention_device_support`
/ `is_flex_attention_supported` / `skip_device_if` or rewrite inductor
`test_flex_attention.py` / `test_flex_decoding.py` unless explicitly requested.

### Export Flex / BlockMask (Done on cont)

### 1) `test_export.py` — `TestExportFlexAttention`

`instantiate_device_type_tests` includes CPU. Uses `.to(device)` / `device=device`.
Skips if device not in `_validate_device` allowlist (`cpu`/`cuda`/`xpu`/`hpu`/`mps`).
Asserts flex HOP + numerics (no device-specific golden). NPU skips.

### 2) `test_experimental.py` — `TestExperimentFlex`

BlockMask export suite via instantiate + `device`. Left CUDA-platform-gated:
`test_aot_export_flex_attention_*` under `IS_FLEX_ATTENTION_CUDA_PLATFORM_SUPPORTED`.
NPU: BlockMask cases pass.

## Export checklist: accelerator-needed but NPU skipped (audit)

### Must do (user confirmed)
| Item | Status |
|------|--------|
| `test_export.py::test_flex_attention_export` | **Done** — `TestExportFlexAttention`; NPU skips (runtime allowlist) |
| `test_experimental.py` Flex/BlockMask suite | **Done** — `TestExperimentFlex`; CUDA-SM flex AOT remain gated |

### Possibly missed — status
| Item | Status |
|------|--------|
| `associative_scan` symbol_dim/scandim/lifted_buffers | **Not changed** — keep `@requires_cuda_and_triton` + CUDA. Baseline **2.13**: pointwise still requires CUDA/XPU in eager validate; NPU cannot pass. Do not device-agnosticize until torch_npu/2.13 stack drops that gate. |
| `assert_tensor_metadata_device_index` | **Done** on cont: `TestExportMoveToDeviceIndex` + `except_for="cpu"`; NPU verified |
| `test_module_to_with_shared_weights` | **Not changed** — keep CUDA-only (`skipIf` + `"cuda" in str`); NPU `Module.to` not dynamo-traceable |
| `test_exception` | **Done** on cont: `TestExportAutocastException`; NPU verified |

## Env notes (this machine / docker `y00839695_pytorch`)

- Setup tree: `/home/y00839695/npu-pytorch-setup/` (`venv_npu`, `run_test.sh cpu|npu`).
- Fork under test: `/home/y00839695/pytorch` (often **newer than** venv torch).
- **Version lock:** `venv_npu` must stay on **PyTorch 2.13.x** because `torch_npu`
  `ci/build.sh` only allows `SUPPORTED_TORCH_VERSION=... 2.13.0`. Fork tip may be
  **2.14.0a0** with `register_custom_class` / `CustomClassBase`. Do **not**
  `pip install -e` the 2.14 fork into `venv_npu` or NPU breaks.
- Consequence: fork tests that require 2.14-only APIs (e.g. `test_serialize.py`
  opaque custom class imports) **cannot import** on this NPU venv until torch_npu
  supports 2.14. Export accelerator/triton instantiate tests that avoid those APIs
  still run.
- **Local-only serialize smoke on 2.13:** use gitignored helpers
  `agent_space/compat_torch213_opaque.py` +
  `agent_space/run_serialize_npu_compat.sh` (shim maps `register_custom_class` →
  `register_opaque_type`). Do not put this shim in PR test code.
- `triton-ascend==3.2.1` in `venv_npu`. After #190324, `has_triton()` should
  see registered NPU DeviceInterface; until then NPU may still get False
  (do not work around via hardcoded `torch_npu._inductor` in upstream tests).
- Official triton-ascend matrix prefers CANN 9.0.0; host may be CANN 9.1 beta + <!-- codespell:ignore -->
  newer `torch_npu` → Cond header stubs / launcher string bugs may need local
  patches (not upstream).
- Build PyTorch only via `pip install -e . -v --no-build-isolation` when user asks;
  use `USE_CUDA=0 USE_ROCM=0 MAX_JOBS=$(nproc)` as in `setup_pytorch_envs.sh`.
  Ask before switching the editable install target away from
  `/home/y00839695/npu-pytorch-setup/pytorch`.

### Intentionally CUDA-only (do not treat as missed)
| Item | Reason |
|------|--------|
| `test_export_raw_triton_kernel_*` | Raw CUDA Triton kernel |
| `test_draft_export.py::test_cuda_memory_usage` | `torch.cuda` memory APIs |
| `test_export_opinfo.py` `@onlyCUDA` / FakeCuda | FakeCuda surface |
| `TestExportRNN` `only_for=("cuda","xpu")` | NPU export lacks LSTM/GRU |
| `test_module_to_with_shared_weights` | CUDA-only; NPU Module.to not dynamo-traceable |
| `test_aoti_cross_compile_windows.py` | Windows+CUDA cross-compile |
| `test_aot_export_flex_attention_*` (experimental) | `IS_FLEX_ATTENTION_CUDA_PLATFORM_SUPPORTED` (SM>=8.0) |
