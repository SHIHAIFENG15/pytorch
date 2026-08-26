# NPU decoupling Cursor skills

Project skills for the two **separate** tracks. Do not mix them in one PR.

| Skill | Track | Use when |
|-------|--------|----------|
| `pytorch-npu-device-agnostic-testing` | **Test** (用例) | Export/AOTI device-agnostic tests, fork workflow, CPU/NPU runners |
| `npu-device-agnostic-export-aoti` | **Test** (用例) | Export/AOTI file-level gates: instantiate + `device`, `has_triton()`, wait vs do-now |
| `rfc-189138-compile-device-interface` | **Feature** (特性) | Dynamo/Inductor source, C1–C8 / R0–R2, `DeviceInterface`, `has_triton()` / `is_gpu()` |

Copied from local `~/.cursor/skills/` onto branch `skills/npu-decoupling` (fork `SHIHAIFENG15/pytorch`).

`docs2` still has an older snapshot at `skill_build/`; prefer the copies under `.cursor/skills/`.
