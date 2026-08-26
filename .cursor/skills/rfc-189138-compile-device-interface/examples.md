# Canonical example: walk the DeviceInterface registry

This is **C1-R0** plus a torch_npu **bit**. It is not a rewrite of Dynamo.

## What already exists (do not rebuild)

In-tree `init_device_reg()` registers `CudaInterface`, `XpuInterface`, …
`get_registered_device_interfaces()` returns that dict.

`import torch_npu` already does:

```python
register_interface_for_device("npu", NpuInterface)
```

`DeviceInterface.is_triton_capable()` defaults to **False**.
`CudaInterface` overrides True (sm >= 7 / HIP).
`NpuInterface` currently does **not** override, so NPU stays False even if
registered.

## pytorch (`torch/utils/_triton.py`) — small

**Before:** hardcoded names (NPU invisible):

```python
triton_supported_devices = {
    "cuda": cuda_extra_check,
    "xpu": _return_true,
    "cpu": cpu_extra_check,
    "mtia": _return_true,
}
```

**After (shape, not a paste-ready patch):**

```python
from torch._dynamo.device_interface import get_registered_device_interfaces

def has_triton() -> bool:
    if not has_triton_package():
        return False
    seen: set[str] = set()
    for name, iface in get_registered_device_interfaces():
        if ":" in name or name in seen:
            continue
        seen.add(name)
        if iface.is_available() and iface.is_triton_capable():
            return True
    return False
```

Move CUDA sm>=7 / CPU triton-backend checks onto `CudaInterface` /
`CpuInterface` if they are not already there. Do not add `"npu"` to a dict.
Do not `import torch_npu`.

CUDA-only machines: same True/False as today.
NPU-only: still False until torch_npu sets the bit.

## torch_npu (`NpuInterface`) — required for NPU True

```python
class NpuInterface(DeviceInterface):
    ...
    @staticmethod
    def is_triton_capable(device=None) -> bool:
        return True  # or: triton-ascend actually present
```

Registration is already there; this bit is the missing half.

## What this does not do

- Does not implement C3 codegen or C5 stream calls.
- Does not replace Export test gates (`HAS_GPU`).
- Does not need `GPU_TYPES.append("npu")`.
- 110 spreadsheet C1 rows are **other call sites**, each its own small PR
  after this predicate exists.
