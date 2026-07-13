---
name: pytorch-npu-device-agnostic-testing
description: >-
  在昇腾 NPU（privateuse1）上把 PyTorch Export/AOTI 上游测试改造为设备无关，并在 CPU/NPU 双环境跑用例、编译、走 SHIHAIFENG15 私仓 fork 工作流的完整上下文。
  Use when working on PyTorch Export/AOTI device-agnostic test decoupling, npu-pytorch-setup, instantiate_device_type_tests / PrivateUse1 / npu tests, AOTI runners, torch_npu, or the SHIHAIFENG15 pytorch fork.
disable-model-invocation: true
---

# PyTorch 用例设备无关（NPU/privateuse1）改造与运行

把 PyTorch **Export / AOTI 上游测试**改造成设备无关（让昇腾 NPU 作为 `privateuse1` 后端能跑），并在 CPU / NPU 双环境下编译、运行、走 fork 提交流程。

## 目标一句话

先看测试要验证什么：**通用逻辑跟当前设备（不写死 `"cuda"`），后端专属能力（CUDA Graph、FakeCuda、Windows 交叉编译等）保留专属入口或 skip，不强行泛化。**

---

## 环境速览（路径/名字是本机的，换机后按实际调整）

- **容器**：`pytorch-npu-yxz-dev`（CANN 8.5.0 + Python 3.11 + 8×910B3）。所有 venv/编译/测试**必须在容器里跑**，宿主机看不到容器内解释器。
- **工作根目录**：`/home/y00839695/npu-pytorch-setup/`
  - `pytorch/`：**已编译**的 PyTorch 源码树（`torch 2.13.0a0+gitfad7424`，pin commit `fad74248e78716152917a729adb2b44ba2bab16e`）
  - `torch_npu/`：已编译安装（`torch_npu 2.13.0+git45fbeae`）
  - `venv_cpu/`：CPU 版 torch
  - `venv_npu/`：NPU 版 torch + torch_npu
  - `setup_pytorch_envs.sh`：一键搭建脚本（`--only-build` 只重编）
  - `run_test.sh`、`init_fork.sh`：辅助脚本
- **改用例的 fork 副本**：`/home/y00839695/pytorch/`（remote `origin` = `git@github.com:SHIHAIFENG15/pytorch.git`）。改测试文件在这里改、这里提交。

### 进入环境

```bash
sudo su root -c "docker exec -it pytorch-npu-yxz-dev bash"
```

### Git 拉取约定（脚本已固化，换机需复刻）

- PyTorch 走 **GitHub SSH**（密钥须注册到对应 GitHub 账号）。
- torch_npu 走 **GitCode HTTPS + 令牌**（存 `~/.git-credentials`）。
- 子模块修复：`git config --global url."git@github.com:".insteadOf "https://github.com/"`（GitHub HTTPS pack 传输在本网络会卡死，全部改走 SSH）。
- CANN `set_env.sh` 在 `set -u` 下有 unbound variable，`source` 前后用 `set +eu` / `set -eu` 包住。

---

## 跑测试（CPU / NPU）

用 `run_test.sh`（在容器内、工作根目录下）：

```bash
./run_test.sh cpu test/export/test_hop.py -v
./run_test.sh npu test/export/test_hop.py -v
./run_test.sh npu test/export/test_export.py -v -k test_basic
```

也可直接用 venv 的 python 跑 fork 副本里的任意测试文件（torch 已装进 venv，测试文件独立于源码树）：

```bash
/home/y00839695/npu-pytorch-setup/venv_npu/bin/python <某测试文件路径> -v
```

### NPU 跑测试的关键坑

- NPU 侧必须 **`import torch_npu`** 且设 **`PYTORCH_TESTING_DEVICE_ONLY_FOR=npu`**（`run_test.sh npu` 已内置）。
- `instantiate_device_type_tests` 在 NPU 下生成的类名是 **`*PRIVATEUSE1`**、用例名带 **`_npu_`**。
- **不要**再用 `-k NPU` 过滤（`ONLY_FOR=npu` 已限定设备，`-k NPU` 会匹配 0 个用例）。
- privateuse1 后端名 = `"npu"`。

### 重新编译（CPU / NPU 各一套）

```bash
# 都在容器内、工作根目录下
ONLY_BUILD=1 ./setup_pytorch_envs.sh            # 或脚本支持的 --only-build
```

CPU 版编译进 `venv_cpu`，NPU 版（torch_npu）编译进 `venv_npu`。编译前需 `source` CANN 的 `set_env.sh`（脚本已处理）。

---

## Fork 工作流（SHIHAIFENG15 私仓）

在 `/home/y00839695/pytorch/`（fork 副本）里改用例：

```bash
git checkout -b npu-device-agnostic-test    # 从 pin commit 建分支
# 改测试文件
git add -A && git commit -m "..."
git push -u origin <分支名>                  # 首推含全量历史，较慢
```

`init_fork.sh` 可自动加远程 + 从 pin commit `fad74248...` 建分支。

---

## 解耦总纲（9 类改造手段摘要）

完整设计（逐文件改法、耦合分类、伪代码）见 **[master-plan.md](master-plan.md)**。摘要：

1. **PrivateUse1 纳入设备调度**：树内已有 `PrivateUse1TestBase`；后端注册后，没有 `only_for` 限制的文件自动多出一套 NPU 用例。不另造 `NpuTestBase`。
2. **`GPU_TYPE`/`HAS_GPU`/Triton 泛化**（依赖 RFC #189138）：`get_gpu_type()` 无设备时禁止回退 `"cuda"`，改走 `torch.accelerator` / device interface。
3. **`requires_cuda_and_triton` 等 skip 改写**（依赖手段 2）：通用用例改 `requires_gpu` / `requires_accelerator_and_triton`；真 CUDA 专属保留 CUDA skip。
4. **补 `instantiate_device_type_tests`**：仅对确需多设备参数化的文件补，不强制全开。
5. **替换 `"cuda"` 硬编码**：`device=GPU_TYPE`/`self.device`；同步用 `torch.accelerator.synchronize()` 等。
6. **AOTI runner 动态加载**：按 `f"AOTIModelContainerRunner{device.capitalize()}"` 用 `getattr(torch._C._aoti, ...)`，找不到就报错，禁止静默落 Cuda。
7. **FlexAttention 泛化**（先做基础设施依赖）：就绪前 Flex 段保持 CUDA 专属，先改同文件非 Flex 部分。
8. **`TEST_*`/`dtypesIf*` 的 PrivateUse1 对应物**：用 `TEST_PRIVATEUSE1`、`dtypesIfPRIVATEUSE1`，不新造 `TEST_NPU`。
9. **`dist.init_process_group` 硬编码**：本清单范围外，不处理。

### 依赖项（可并行，但不替代改测试文件）

- **RFC #189138**：挡住手段 2、3 及 5/7 的门禁。
- **手段 7 基础设施**：挡住 `test_experimental.py`/`test_export.py` 的 Flex 段。
- **`AOTIModelContainerRunnerNpu` + pybind**：挡住手段 6 在 NPU 真正跑通。

---

## 已定结论 / 坑（避免重复调查）

- **`AOTIModelContainerRunnerNpu`**：torch_npu 里 C++ 侧存在，但**未**作为 `torch._C._aoti` 的 pybind 属性暴露，走的是 registry（键 `"npu"`）。所以手段 6 的 `getattr(torch._C._aoti, "AOTIModelContainerRunnerNpu")` 在 NPU 上当前拿不到——这是待补的基础设施缺口。
- **`test/export/test_export_opinfo.py`**：保持 CUDA 专属（FakeCuda 相关），**不改**。
- **`test/inductor/test_aoti_cross_compile_windows.py`**：Windows 交叉编译是 GPU/CUDA 专属，**不改**。
- **保留不动**：CUDA 显存峰值、`CUDAGraph`、`FakeCuda`（`@onlyCUDA` + `only_for="cuda"`）等真专属能力。
- `setup_pytorch_envs.sh` 跑完偶尔在结尾报 `unexpected EOF`：是运行中改脚本导致行号错位的表象错误，磁盘上脚本语法正确，可忽略。
- `tail -f` / `run_test.sh` 在**宿主机**跑会因为找不到容器内解释器而报 `No such file or directory`——一律进容器再跑。

---

## 术语

- **privateuse1 / npu**：PyTorch 给第三方加速器的通用后端槽位；本项目 rename 成 `"npu"`。
- **`instantiate_device_type_tests`**：把一个通用模板 TestCase 按设备自动复制成 `*CPU`/`*CUDA`/`*PRIVATEUSE1` 多套。
- **`copy_tests`**：AOTI 主路径用的另一套「模板复制到 CPU/GPU 类」机制。
