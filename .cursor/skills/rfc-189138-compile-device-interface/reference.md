# RFC #189138 reference

Umbrella issue (open): https://github.com/pytorch/pytorch/issues/189138

Eager PrivateUse1 already goes through `torch.utils.backend_registration` /
accelerator APIs. Dynamo and Inductor still branch on in-tree type strings.
jansel: extend the existing OOT/`DeviceInterface` path; no second registration
system; prefer small de-hardcoding patches.

## Spreadsheet C* vs RFC

Graph-mode sheet `特性解耦人员分工_分布式图模式_v1.0.xlsx` on `origin/docs2`.
Full scheme text: [schemes.md](schemes.md). Example walk: [examples.md](examples.md).

| Sheet | RFC / PR |
|-------|----------|
| C1 capability / name lists | #189135, `has_triton` #190324 |
| C2 device from context | #189136 Dynamo tables |
| C3 Inductor codegen | #189137 `DeviceOpOverrides` |
| C5 Stream/Event runtime | DeviceInterface methods; load hook #191314 |
| C7 VariableBuilder | RFC before code; not a drive-by |
| C8 ProcessGroup | RFC; **not** DeviceInterface |
| C4 / C6 | Empty in v1.0 |

`origin_plan.md` 方案 1-9 (instantiate / `HAS_GPU`) is **test** work. Different skill.

## Sub-RFC map

| Issue | Layer | In-tree work |
|-------|--------|----------------|
| [#189135](https://github.com/pytorch/pytorch/issues/189135) | Identity / capability | Bits on `DeviceInterface` (`is_triton_capable`, later `is_gpu`, `exposes_streams`) |
| [#189136](https://github.com/pytorch/pytorch/issues/189136) | Dynamo tables | Replace `["cuda", "xpu"]` (and similar) with registry queries / capability bits |
| [#189137](https://github.com/pytorch/pytorch/issues/189137) | Inductor codegen | `GPU_TYPES` / `is_gpu` / `device_need_guard` / `DeviceOpOverrides` |
| [#191314](https://github.com/pytorch/pytorch/issues/191314) | OOT load | Lazy `torch_<backend>._dynamo.device_interface` — do not import dynamo from eager `__init__` |
| [#191317](https://github.com/pytorch/pytorch/issues/191317) | Features | `BackendFeature` (or equivalent) for optional compile features |

**Out of scope:** AOTI C-shim `aoti_torch_{device}_{op}`. That is a separate
codegen/ABI problem, not DeviceInterface membership.

## Related PRs (verify status on GitHub; do not assume)

| PR | Intent |
|----|--------|
| [#190324](https://github.com/pytorch/pytorch/pull/190324) | `has_triton()` iterates `get_registered_device_interfaces()` and uses `is_triton_capable` instead of a hardcoded device dict |
| [#190326](https://github.com/pytorch/pytorch/pull/190326) | `is_gpu()` / `device_need_guard()` / `GPU_TYPES` derived from registry (`is_gpu()`, `exposes_streams()`) |

Test-harness follow-ups (`HAS_GPU`, `@requires_cuda_and_triton` rewrite) are
**not** this skill. They consume the predicates once they exist.

## Files to grep first

```
torch/_dynamo/device_interface.py
torch/_dynamo/utils.py
torch/_dynamo/graph_utils.py
torch/utils/_triton.py
torch/_inductor/utils.py          # GPU_TYPES, get_gpu_type, is_gpu, device_need_guard
torch/_inductor/codegen/common.py # DeviceOpOverrides
torch/_inductor/codegen/*/device_op_overrides.py
torch/_inductor/fx_passes/freezing_patterns.py
```

High-signal leftover patterns:

- `GPU_TYPES = [`
- `device in ["cuda", "xpu"`
- `triton_supported_devices = {`
- `device != "mps" and is_gpu`
- `getattr(torch, x).is_available()` over a hardcoded GPU name list

## Capability bits

Add on `DeviceInterface` with **default False**. In-tree classes that already
match today's behavior override True.

| Bit | Meaning | Typical in-tree |
|-----|---------|-----------------|
| `is_triton_capable()` | Device type can have a Triton backend (even if the package is missing) | cuda, xpu, cpu (with extra check), mtia |
| `is_gpu()` | Counts as GPU for inductor helpers / `GPU_TYPES` shim | cuda, mps, xpu, mtia — not cpu |
| `exposes_streams()` | Needs stream device guards (`device_need_guard`) | cuda/xpu/mtia yes; mps historically no |

Do not special-case `"mps"` in `device_need_guard` once `exposes_streams` exists.

`has_triton()` extra checks that are truly CUDA-arch-specific (e.g. sm >= 7)
stay as interface-specific logic or `raise_if_triton_unavailable`, not as a
new in-tree device-name dict.

## Registration contract (OOT)

1. Subclass `DeviceInterface` (and `DeviceOpOverrides` if codegen needs it).
2. Call `register_interface_for_device("<privateuse1-name>", Interface)`.
3. Load that module without importing `torch._dynamo` from the backend's eager
   `__init__` — follow #191314 convention when implementing the load hook.
4. Never require pytorch to `import torch_npu` or `import torch_npu._inductor`.

## Dynamo leftovers (#189136)

Replace type-string tables with "ask the interface" or "ask is_gpu /
is_triton_capable". Keep CUDA-only semantics that are **not** identity
(e.g. CUDA graphs, nvtx) behind CUDA-specific APIs, not behind a widened
`is_gpu()`.

## Inductor leftovers (#189137)

- `is_gpu(device)` → interface `is_gpu()` (string may be `"cuda:0"`; match
  today's type-only check).
- `GPU_TYPES` → list of registered names where `is_gpu()` is true (cache if
  needed; `get_gpu_type()` must keep "at most one available GPU type").
- Stream guards → `exposes_streams()`.
- Kernel wrappers / headers → `DeviceOpOverrides`, not new if/elif on `"npu"`.

## What "zero behavior change" means

For a machine with only CUDA: `has_triton()`, `is_gpu("cuda")`,
`device_need_guard("cuda")`, `get_gpu_type()` unchanged.
Same for xpu-only and mps-only.
PrivateUse1 becomes visible **only after** its OOT package registers.

## Tests for source patches

Prefer existing inductor/dynamo unit tests of the helper:

- Register a throwaway `DeviceInterface` in the test process.
- Assert the predicate sees it.
- Unregister / isolate so other tests do not pick up `"npu"`.

Do not land Export `instantiate_device_type_tests` changes in the same PR
unless they are a one-line consumer of the new predicate and the test skill
agrees they are unblocked.
