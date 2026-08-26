---
name: rfc-189138-compile-device-interface
description: >-
  Compiles PrivateUse1 into torch.compile via the DeviceInterface registry
  (RFC #189138). Use when editing Dynamo/Inductor SOURCE, C1-C8 / R0-R2 graph-mode
  特性解耦, 分布式图模式 spreadsheet, device_interface.py, GPU_TYPES, has_triton(),
  is_gpu(), DeviceOpOverrides, or PRs #190324 #190326 #191314 #191317. Not for
  Export/AOTI test instantiate or HAS_GPU test gates.
---

# RFC #189138: compile-stack DeviceInterface decoupling

**This skill is source-first.** It covers making Dynamo / Inductor / `has_triton()`
ask the existing `DeviceInterface` registry instead of hardcoded `cuda`/`xpu`/`mps`
lists. It does **not** cover Export/AOTI test refactors.

Umbrella: [pytorch#189138](https://github.com/pytorch/pytorch/issues/189138)
(DeviceInterface registry for PrivateUse1 in `torch.compile`). Eager already has
the contract; compile still hardcodes in-tree accelerators.

If the task is a **test file** under `test/export/` or AOTI suites, stop and use
`npu-device-agnostic-export-aoti` / `pytorch-npu-device-agnostic-testing`.

Details: [schemes.md](schemes.md) (C1-C8), [examples.md](examples.md)
(`has_triton` walk), [reference.md](reference.md) (RFC / files).

## Classify the site (C1-C8)

Spreadsheet schemes are **fix patterns** for `torch/` CUDA literals. R0 = edit
in-tree; R1 = also a public register hook; R2 = RFC first.

| See | Scheme | Mechanism |
|-----|--------|-----------|
| Name list / `is_cuda` / `has_triton` dict | **C1** | Walk registry + capability bit |
| `device="cuda"` in framework | **C2** | Device from tensor/args |
| Emit Triton/CUDA kernel text | **C3** | `DeviceOpOverrides` |
| `torch.cuda.Stream/Event/synchronize` | **C5** | `DeviceInterface` methods |
| Dynamo `isinstance(torch.cuda.*)` | **C7** | RFC `register_variable_builder` |
| `backend == "nccl"` | **C8** | RFC ProcessGroup registry |
| nn CUDA fast path / DLPack | **C4 / C6** | v1.0 sheet has 0 rows; skip |

Walking `get_registered_device_interfaces()` is **small** (table already
exists). 110 C1 rows are many call sites, not one registry rewrite.
pytorch never writes `"npu"`; torch_npu registers `NpuInterface` and sets bits.
`NpuInterface` today does not override `is_triton_capable` (stays False).

CUDA Graph / NCCL / nvtx are often **features**, not identity. Do not fold them
into `is_gpu()`.

## Which skill

| Task | Skill |
|------|--------|
| `torch/_dynamo/device_interface.py`, `torch/utils/_triton.py`, `torch/_inductor/utils.py` `GPU_TYPES`/`is_gpu`/`device_need_guard`, `DeviceOpOverrides` | **this skill** |
| Dynamo hardcoded `["cuda", "xpu"]` tables, inductor codegen device strings | **this skill** |
| `instantiate_device_type_tests`, injected `device`, `@requires_gpu`, `HAS_GPU` test gates | test skills, not this |
| AOTI C-shim `aoti_torch_{device}_{op}` | **out of scope** of RFC #189138 |

## Hard rules

1. **One registry.** Extend `DeviceInterface` / `register_interface_for_device` /
   `get_registered_device_interfaces` and existing `DeviceOpOverrides`. Do not add
   a parallel PrivateUse1 registration path.
2. **Capability bits default False.** New predicates (`is_gpu`, `is_triton_capable`,
   `exposes_streams`, later `BackendFeature`) default off. In-tree backends that
   already behave as GPU/Triton **opt in** so cuda/xpu/mps behavior is unchanged.
3. **Do not monkeypatch** `GPU_TYPES`, `has_triton()`, or `is_gpu()` from
   `torch_npu` / tests as the product fix. Register a `DeviceInterface` and make
   in-tree predicates walk the registry.
4. **Do not hardcode `"npu"`** (or any PrivateUse1 name) into in-tree device lists.
   In-tree code asks the registry; the OOT package registers the interface.
5. **Do not import `torch._dynamo` from the OOT eager import path.** Lazy convention
   registration is RFC [#191314](https://github.com/pytorch/pytorch/issues/191314)
   (`torch_<backend>._dynamo.device_interface`).
6. **AOTI C-shim is out of scope.** Do not fold `aoti_torch_{device}_{op}` into
   these patches.
7. **Incremental patches.** De-hardcode one predicate or one call-site family per
   PR (jansel: small de-hardcoding patches, not a second registration system).
8. **Comments:** no designer nicknames ("手段 N", "方案 N") in product code.
9. **Check the tree before reimplementing.** This workspace's `has_triton()` may
   still use a hardcoded `triton_supported_devices` dict even if GitHub #190324
   is merged elsewhere. Read the file; do not assume.

## Current in-tree snapshot (re-read before editing)

| Site | Typical state until RFC PRs land |
|------|----------------------------------|
| `torch/_dynamo/device_interface.py` | Registry exists; `is_triton_capable` defaults False; **no** `is_gpu` / `exposes_streams` until #190326 |
| `torch/utils/_triton.py` `has_triton()` | Often still a hardcoded `{cuda, xpu, cpu, mtia}` dict; #190324 walks the registry |
| `torch/_inductor/utils.py` | `GPU_TYPES = ["cuda", "mps", "xpu", "mtia"]`; `is_gpu` is `device in GPU_TYPES`; `device_need_guard` special-cases MPS |
| `torch/_inductor/codegen/common.py` | `DeviceOpOverrides` + `register_device_op_overrides` — extend, do not replace |

`GPU_TYPES` stays a **list shim** derived from the registry after #190326. Do not
invent a new public `get_gpu_types()` unless a later RFC asks for it.

## Patch workflow

1. Classify with [schemes.md](schemes.md) (C1-C8), then the sub-RFC map in [reference.md](reference.md).
2. Prefer an existing `DeviceInterface` method. If missing, add a capability bit
   with default `False` and opt in in-tree backends that already have the behavior.
3. Replace the list/dict with a walk of `get_registered_device_interfaces()` (or a
   helper that does). Keep `GPU_TYPES` as a derived list if callers need a list.
4. Zero behavior change for cuda/xpu/mps/cpu: add or extend unit tests of the
   **predicate** (e.g. inductor `test_utils` / `TestHasTriton`), not Export suites.
5. Grep remaining copies of the old list in `torch/_dynamo` and `torch/_inductor`.
6. OOT (`torch_npu`) only: implement/override the bits and `register_interface_for_device("npu", ...)`.
   Never teach pytorch source the string `"npu"`.

## Tests that belong here vs not

**In scope:** unit tests of the new query (`has_triton` sees a registered fake
interface; `is_gpu("npu")` after register; `device_need_guard` uses `exposes_streams`).

**Out of scope:** `test/export/test_*.py`, AOTI `copy_tests` / instantiate /
`hw_classification`. Those wait on the registry but are owned by the test skills.

## Common mistakes

| Excuse | Reality |
|--------|---------|
| "Append `"npu"` to `GPU_TYPES` in pytorch" | PrivateUse1 name is OOT. Registry + `is_gpu()` bit. |
| "Monkeypatch `GPU_TYPES` / `has_triton` in torch_npu" | Breaks the contract; jansel: extend DeviceInterface. |
| "This RFC is the Export instantiate PRs" | Those are test entry. This RFC is compile runtime. |
| "Also generate AOTI C-shim" | Explicitly out of scope of #189138. |
| "Import dynamo from torch_npu `__init__`" | Cycle / eager cost; use #191314 convention. |
| "One PR to replace every cuda string" | Rejected style. Incremental de-hardcoding. |
| "Walking the registry is a huge refactor" | Registry exists. `has_triton()` walk is small; C7/C8 are the large RFCs. |
| "pytorch must import torch_npu for has_triton" | OOT registers; in-tree only asks the table. |
| "C1 cudagraph_* file is the same as has_triton" | Filename cuda often means CUDA-only feature. |

## Red flags — stop

- Editing `test/export/` or `test/inductor/test_aot_inductor*` under this skill
- `GPU_TYPES.append("npu")` or `if device == "npu"` in `torch/`
- `import torch_npu._inductor` inside pytorch source or pytorch tests as the fix
- New parallel `register_npu_backend()` next to DeviceInterface
- Product comments with "手段 2/3"
