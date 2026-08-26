# C1-C8 source decoupling schemes

Source: `origin/docs2` spreadsheet
`特性解耦人员分工_分布式图模式_v1.0.xlsx` (graph-mode **source**, not tests).

These are **how to fix a hardcoded CUDA site**, not eight products. Classify the
site first, then pick one scheme. Do not mix with Export `instantiate` work.

Counts in the v1.0 sheet (338 sites): C1 110, C3 102, C5 82, C2 36, C8 5, C7 3,
C4/C6/C0 0.

## R0 / R1 / R2 (depth)

| Tag | Meaning | pytorch | torch_npu |
|-----|---------|---------|-----------|
| **R0** | Replace a literal / ask an existing API | Small local edit | Usually none for that site |
| **R1** | Same, plus a **public registration hook** OOT can call | May add a default-False bit or document an existing register_* | Implement + register |
| **R2** | Changes Dynamo/c10d extension shape | **RFC first** | Wait for the RFC |

`C3-R0/R1` means some sites are R0 (ask existing `DeviceOpOverrides`) and some
need R1 (new override method).

## Who changes what

| Layer | In `pytorch/` | In `torch_npu` |
|-------|----------------|----------------|
| Ask the registry | Walk `get_registered_device_interfaces()`; never write `"npu"` | Already calls `register_interface_for_device("npu", NpuInterface)` on import |
| Capability | Add bit default **False**; cuda/xpu/mps opt in | Override True (e.g. `NpuInterface.is_triton_capable`) |
| Codegen strings | Call `get_device_op_overrides(device.type)` | `register_device_op_overrides("npu", ...)` |
| Walking the registry itself | **Small** (registry already exists) | Not in pytorch |

`NpuInterface` today does **not** override `is_triton_capable`, so it inherits
`DeviceInterface` → `False`. A pytorch-only `has_triton()` walk does not make
NPU true until torch_npu sets the bit.

## CUDA-only vs identity

Filename or API containing `cudagraph` / `nccl` / `nvtx` is often **CUDA
feature**, not "device identity". Do not widen `is_gpu()` to hide CUDA Graph.
Keep those as CUDA-only or map to a named feature bit later (`BackendFeature`).

---

## C1 — drop device-name gates; ask capability / dispatch

**One line:** Code asks "is this cuda/xpu?" Should ask "does this device have
the capability?"

**Looks like:** `"cuda" in ...`, `is_cuda`, `device in ["cuda", "xpu", "mtia"]`,
`triton_supported_devices = {cuda, xpu, cpu, mtia}`.

**If unchanged:** NPU never matches the list. Appending `"npu"` in pytorch is
the wrong fix.

**Fix:** `get_registered_device_interfaces()` + `is_triton_capable()` /
`is_gpu()` / `is_available()`. Skip `cuda:0` duplicate keys.

**Size:** Per site small. 110 sites is many PRs, not one mega-diff. The
**canonical** first site is `torch/utils/_triton.py` `has_triton()` (RFC
#190324). See [examples.md](examples.md).

**torch_npu:** Register interface (done) + set the bit the predicate reads.

**RFC:** #189135 / #190324. Dynamo leftover lists are #189136.

---

## C2 — device flows from context; delete literals

**One line:** The graph or argument already has a device; do not `to("cuda")`
or scan `["cuda", "xpu"]` to guess it.

**Looks like:** `node.target == "cuda"`, `device="cuda"` in framework code,
`.cuda()` on values that already have `.device`.

**Example:** `torch/_dynamo/graph_utils.py` loops `for gpu in ["cuda", "xpu"]`
to detect `.to("cuda")`. `.npu()` / `.to("npu")` is invisible.

**Fix:** Use fake-tensor `device.type`, the incoming `device` argument, or
`torch.accelerator` **current device of the tensor**, not a name list.

**Size:** Small per site. Do not use module-level
`torch.accelerator.current_accelerator()` as a silent `"cuda"` fallback.

**torch_npu:** None for the in-tree site.

**RFC:** #189136 (Dynamo tables). Same *idea* as test-side injected `device`,
but this scheme is **source**.

---

## C3 — Inductor codegen backend registration (triton / cutlass)

**One line:** Generated wrapper/kernel text must come from a per-device plugin,
not `if device.type == "cuda": emit cuda strings`.

**Looks like:** `if device.type != "cuda": return`,
`torch.cuda.get_device_properties` inside a pass, Triton heuristic branching
on `"cuda"` vs `"xpu"`.

**Example:** `torch/_inductor/fx_passes/replace_random.py` returns 0 for every
non-CUDA device, then uses CUDA properties.

**Fix:** `DeviceOpOverrides` / `register_device_op_overrides` /
scheduling hooks. Add an override method if missing (R1), default no-op.

**Size:** Medium-high. Wrong strings break compiled kernels. Need Inductor
lowering context.

**torch_npu:** Register NPU overrides. pytorch still must not mention `"npu"`.

**RFC:** #189137. CUDA Graph utils named `cudagraph_*` may be CUDA-only — do
not auto-apply C3.

---

## C4 — nn forward CUDA behavior → registrable op/hook

**One line:** LSTM / SyncBN / similar keep a CUDA fast path in `nn` modules.

**Sheet v1.0:** 0 sites. Skip this round unless a new scan finds hits.

**Fix (when it appears):** registrable kernel or hook, not `if tensor.is_cuda`.

**RFC:** not #189138 core; separate nn dispatch work.

---

## C5 — `torch.cuda.Stream` / `synchronize` / `Event` → DeviceInterface

**One line:** Runtime Python should call the interface, not `torch.cuda.*`.

**Looks like:** `torch.cuda.current_stream()`, `torch.cuda.Event`,
`torch.cuda.synchronize()`, `isinstance(value, torch.cuda.StreamContext)`.

**Example:** `torch/jit/_trace.py` skips timing unless `torch.cuda.is_available()`,
then uses CUDA Event/stream.

**vs C3:** C5 is a **Python call now**. C3 **pastes source text** into generated
files.

**vs C1:** C1 is membership ("are you a GPU?"). C5 is operations on the device.

**Fix:** `get_interface_for_device(dev).current_stream()` / `.Event` / `.synchronize()`.
Base `DeviceInterface` already has these methods; in-tree `CudaInterface` binds
them. OOT `NpuInterface` already binds NPU Stream/Event.

**Size:** Medium. Need the interface registered (R1 if the hook is not public
enough). #191314 if registration happens at eager import.

**torch_npu:** `NpuInterface.Stream` / `Event` already exist; in-tree must
**call the interface**.

**RFC:** #189135 + DeviceInterface surface.

---

## C6 — include_paths / serialization / DLPack via dispatch

**One line:** C++ headers, Storage, DLPack branch on device name.

**Sheet v1.0:** 0 sites. Skip.

**Out of scope if it is AOTI C-shim** `aoti_torch_{device}_{op}` — that is not
DeviceInterface membership.

---

## C7 — Dynamo VariableBuilder `register_variable_builder` (RFC)

**One line:** Dynamo traces CUDA types with a long `isinstance` chain. OOT
types graph-break unless they can register a builder.

**Looks like:** `isinstance(value, torch.cuda.StreamContext)`,
`class *Cuda*` in `variables/`.

**Example:** `torch/_dynamo/variables/builder.py` special-cases
`torch.cuda.StreamContext` then generic `torch.Stream`.

**Fix:** Public `register_variable_builder` (or equivalent). One RFC for all
C7 sites (builder / ctx_manager / streams). Do not land three private patches.

**Size:** High. Do not start here.

**torch_npu:** After RFC, register NPU Stream/Event builders. Today some of
this is monkeypatched in `torch_npu.utils._dynamo` — that is not the upstream
fix.

**RFC:** new or #189136 follow-up; sheet marks R1/R2.

---

## C8 — c10d ProcessGroup backend literals → registry (RFC)

**One line:** `if backend == "nccl"` should be "does this backend have X?"

**Looks like:** `"nccl"`, `"gloo"` in `torch/distributed/**`.

**Example:** `torch/distributed/c10d_logger.py` reads `torch.cuda.nccl.version()`
only when backend is `"nccl"`. HCCL never matches.

**Fix:** ProcessGroup/backend capability registry. RFC required.

**Size:** High. Not DeviceInterface. Do not fold into a `has_triton()` PR.

**torch_npu:** HCCL backend registration is OOT; in-tree must not hardcode
`"hccl"` as the only new name if a registry exists.

---

## C0 — other

Manual. Do not invent a ninth register.

## Classifier

| You see | Scheme | First move |
|---------|--------|------------|
| Device name list / `is_cuda` / `has_triton` dict | C1 | Walk registry + capability bit |
| `device="cuda"` / `.cuda()` in **framework** | C2 | Use tensor/arg device |
| Generating Triton/CUDA kernel text | C3 | `DeviceOpOverrides` |
| nn module CUDA fast path | C4 | Skip unless scan finds it |
| `torch.cuda.Stream/Event/synchronize` | C5 | `DeviceInterface` methods |
| DLPack / cpp include | C6 | Skip unless scan finds it |
| Dynamo `isinstance(torch.cuda.*)` | C7 | RFC; do not patch |
| `backend == "nccl"` | C8 | RFC; not this DeviceInterface PR |
| `test/export` instantiate | **not C\*** | Test skills |
