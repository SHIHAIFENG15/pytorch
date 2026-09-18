# PyTorch 硬件解耦综合分析与修改报告

分析审核报告 + 修改交付报告 · 生成日期：2026-09-08 · 基线：main@1859e8b4716e

**2**已更新并推送分支

**通过**语法解析 / diff check

**已推送**origin 同名分支

[综合结论](#summary)[判定准则](#criteria)[10 文件审核](#scan)[Serialization 修改](#serialization)[Wrapper 修改](#wrapper)[TokenSwitch 方案](#token-switch)[验证与交付](#verification)

## 一、执行结论

| 分支                                                         | 修改结果                                                     | 当前状态     |
| :----------------------------------------------------------- | :----------------------------------------------------------- | :----------- |
| `decouple_serialization` [`d7bad3e0641`](https://github.com/zouliuchangsong-debug/pytorch/commit/d7bad3e0641) | 保留未注册设备 module 的直通行为；测试改为直接验证不会调用 `_validate_device`。 | 已提交并推送 |
| `decouple_wrapper_benchmark` [`064468c0e92`](https://github.com/zouliuchangsong-debug/pytorch/commit/064468c0e92) | CPU-only 测试迁移到独立文件；旧 CLI 参数隐藏；异常时关闭 memory history。 | 已提交并推送 |
| `TokenSwitch`                                                | 本轮只提供注册机制设计，不修改代码。                         | 符合用户范围 |

用户已按仓库 AI_POLICY 审核并批准精确代码。本地 PyTorch 缺少 `libtorch_global_deps.dylib`，因此测试在用户提供的 NPU 容器执行；两个分支均已提交并推送。

## 二、分析审核：判定准则

本次判定综合以下三份指导文档，并以“是否迫使 out-of-tree 后端 monkey-patch 或 fork PyTorch”作为最终仲裁：

- [特性解耦判定.md（内嵌附录）](#criteria-decision)
- [hardware_decoupling_feature_spec.md（内嵌附录）](#criteria-spec)
- [pytorch_3rd_party_backend_solutions.md（内嵌附录）](#criteria-solutions)

**1. 领地**
是否属于 CUDA/NCCL/CUTLASS 私有实现**2. 通用性**
是否承担设备无关的公共机制**3. 硬编码**
是否绕过已有设备抽象**4. 扩展点**
第三方后端能否公开注册**5. Patch-Free**
NPU 不 patch 核心源码能否运行

**关键约束：**默认行为不变。设备 module 未注册时应保持原有直通或 fallback 语义；不能为了“更早报错”而扩大硬件解耦 PR 的行为变化。

<details style="box-sizing: border-box; margin: 10px 0px; padding: 10px 14px; border: 0.666667px solid rgb(219, 226, 234); border-radius: 9px;"><summary style="box-sizing: border-box; cursor: pointer; font-weight: 600;">严重度与整改方式</summary><ul style="box-sizing: border-box; margin: 8px 0px; padding-left: 22px;"><li style="box-sizing: border-box;"><strong style="box-sizing: border-box;">V0：</strong>没有可用通用路径，只能 patch/fork；优先整改。</li><li style="box-sizing: border-box;"><strong style="box-sizing: border-box;">V1：</strong>存在 fallback，但第三方后端仍不能正确走通。</li><li style="box-sizing: border-box;"><strong style="box-sizing: border-box;">V2：</strong>内部已有机制，仅缺少公开、稳定的注册入口。</li><li style="box-sizing: border-box;"><strong style="box-sizing: border-box;">R0：</strong>让调用点遵从已有抽象；<strong style="box-sizing: border-box;">R1：</strong>公开内部注册入口；<strong style="box-sizing: border-box;">R2：</strong>新增注册机制。</li></ul></details>

## 三、分析审核：10 个文件分类汇总

| 文件                                                         | 判定              | 依据与处置                                                   |
| :----------------------------------------------------------- | :---------------- | :----------------------------------------------------------- |
| [`torch/jit/_serialization.py`](https://github.com/zouliuchangsong-debug/pytorch/blob/decouple_serialization/torch/jit/_serialization.py) | V0 · 已整改       | JIT 序列化是通用基础设施，原实现仅验证 CUDA。改为复用通用 `_validate_device`，并保留未注册 module 的直通行为。 |
| [`torch/onnx/_internal/exporter/_registration.py`](https://github.com/pytorch/pytorch/blob/main/torch/onnx/_internal/exporter/_registration.py) | 不违反            | `device` 类型允许任意 `str`，CUDA/CPU 只是类型提示示例，不构成控制流白名单或第三方后端门禁。 |
| [`torch/_vendor/quack/_compile_worker.py`](https://github.com/pytorch/pytorch/blob/main/torch/_vendor/quack/_compile_worker.py) | 后端私有          | Quack 的 CUDA 编译 worker，硬编码 CUDA 属于实现边界，不应抽象成所有设备共用路径。 |
| [`torch/_vendor/quack/autotuner.py`](https://github.com/pytorch/pytorch/blob/main/torch/_vendor/quack/autotuner.py) | 后端私有          | CUDA Graph、`CUDA_VISIBLE_DEVICES` 和 CUDA L2 benchmark 均是 Quack autotuner 的固有语义。 |
| [`torch/_vendor/quack/trace.py`](https://github.com/pytorch/pytorch/blob/main/torch/_vendor/quack/trace.py) | 后端私有          | 直接依赖 CUTLASS、NVVM、PTX 与 NVIDIA 架构计时器，属于明确的 CUDA DSL 实现。 |
| [`torch/distributed/rpc/server_process_global_profiler.py`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/rpc/server_process_global_profiler.py) | 边界项 · 后续跟踪 | 类刻意保持 legacy autograd profiler 的 `use_cuda` API 与 CUDA event 语义。本轮不单独改变兼容接口；若迁移，应随 profiler 的通用 `use_device` 方案统一设计。 |
| [`dense_blockscaled_gemm_kernel.py`](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/kernel/vendored_templates/cutedsl/wrappers/dense_blockscaled_gemm_kernel.py) | 后端私有          | 位于 vendored CuTeDSL 模板目录，依赖 NVIDIA Blackwell、CUTLASS 与 CUDA stream；不属于设备无关调度层。 |
| [`torch/_inductor/wrapper_benchmark.py`](https://github.com/zouliuchangsong-debug/pytorch/blob/decouple_wrapper_benchmark/torch/_inductor/wrapper_benchmark.py) | V0 · 已整改       | 通用 compiled-module benchmark 原来以 `torch.cuda` 门禁峰值内存与 snapshot。现通过 `torch.accelerator` 和设备 module 分发。 |
| [`torch/distributed/_shard/sharded_tensor/api.py`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/_shard/sharded_tensor/api.py) | 不违反            | NCCL 分支选择 CUDA 是通信后端的固有配对；其余迁移路径已使用 `torch.accelerator` 与 `_get_device_module`。命名为 `cuda()` 的 API 也不应泛化。 |
| [`torch/distributed/_token_switch.py`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/_token_switch.py) | V0 候选 · 暂缓    | 通用 autograd 层依赖 NCCL 子类私有状态，且没有 HCCL 等 out-of-tree 通信后端的注册工厂。本轮按用户要求只给方案，不改代码。 |

汇总：2 项完成整改，1 项架构整改暂缓，1 项随上游 legacy profiler 统一跟踪，6 项判定为不违反或后端私有实现。

## 四、修改报告：decouple_serialization

提交链： [`8269c212b8a` 通用设备验证实现](https://github.com/zouliuchangsong-debug/pytorch/commit/8269c212b8af1ecce90ffcd2fef5a6414e1c16ce) → [`d7bad3e0641` 审核后测试修正](https://github.com/zouliuchangsong-debug/pytorch/commit/d7bad3e064131535be5dfb5450dfaecdeafdff83)

### 保留的实现策略

仅当 `torch.<device>` module 已注册时调用通用 `_validate_device`。未注册 module、CPU 和 meta 继续按 main 原行为直通，避免把硬件解耦 PR 扩大为错误语义变更。

```
if (
    map_location is not None
    and map_location.type != "cpu"
    and hasattr(torch, map_location.type)
):
    _validate_device(map_location, map_location.type)
```

### 本轮测试修改

- 测试重命名为 `test_validate_map_location_unregistered_device_module_passthrough`。
- 移除对当前模块布局的脆弱断言 `assertFalse(hasattr(torch, "meta"))`。
- mock JIT 模块中的 `_validate_device`，明确断言未注册 module 时不会调用它。
- 字符串和 `torch.device` 两种 meta 输入均保持原值返回。

## 五、修改报告：decouple_wrapper_benchmark

提交链： [`d7e7f0912b2` accelerator 通用化](https://github.com/zouliuchangsong-debug/pytorch/commit/d7e7f0912b246c57c42495712aaa2fabd0a45278) → [`064468c0e92` 审核后可靠性与测试修正](https://github.com/zouliuchangsong-debug/pytorch/commit/064468c0e921f23c37a40c8804d695744ad2d8b8)

### 实现调整

- `--memory-snapshot` 成为公开参数。
- `--cuda-memory-snapshot` 保持兼容，但通过 `argparse.SUPPRESS` 从帮助信息隐藏。
- memory history 使用 `try/finally` 关闭，benchmark 或 dump 抛异常时也不会遗留全局记录状态。
- 继续通过 `torch.get_device_module(acc).memory` 分发 CUDA、XPU 和 NPU snapshot 实现。

### 测试结构调整

原新增测试位于只有 `HAS_GPU` 时才调用 `run_tests()` 的文件中，CPU-only CI 不会执行。现将其迁移到独立的 `test/inductor/test_wrapper_benchmark.py`，无条件运行。

- 无 accelerator 时跳过峰值内存与 snapshot。
- 验证 reset → benchmark → max memory 的调用顺序。
- 使用 `@parametrize` 同时覆盖新参数和兼容参数。
- 验证旧参数不出现在 `--help`。
- 验证不支持 snapshot 的设备安全跳过。
- 验证支持设备的 record、benchmark、dump、disable 行为。
- 验证 benchmark 抛异常时仍关闭 memory history。

## 六、分析审核：TokenSwitch out-of-tree 后端建议

推荐使用 Python 工厂注册表，不让 PyTorch 直接依赖 HCCL，也暂不增加 C++ ProcessGroup 虚函数。

**PyTorch**
定义 TokenSwitch 契约和注册表**内置 NCCL**
注册 `nccl` factory**torch-npu**
注册 `hccl` factory**调用方**
通过统一 factory 创建实例

```
register_token_switch_backend("nccl", TokenSwitchNCCL)
register_token_switch_backend("hccl", HCCLTokenSwitch)  # torch-npu side

switch = create_token_switch(
    process_group,
    backend="hccl",
    num_experts=num_experts,
    max_dispatch_tokens_per_rank=max_dispatch,
    max_recv_tokens_per_rank=max_recv,
    max_token_bytes=max_token_bytes,
)
```

### 基类必须先消除的隐藏耦合

通用 autograd 当前直接读取 `ctx.ts._max_recv_tokens_per_rank`，但该字段只由 `TokenSwitchNCCL` 初始化。应将其提升为基类属性或抽象 property，否则 HCCL 实现仍需复制 NCCL 私有字段。

### 注册规则

- 按通信后端名 `nccl`/`hccl` 注册，而非按设备名注册。
- 单后端 ProcessGroup 可自动推断；多后端 ProcessGroup 要求显式指定。
- 重复注册默认报错，测试场景可提供显式 override。
- 保留直接构造 `TokenSwitchNCCL` 的兼容路径。

## 七、修改报告：验证与交付记录

| 检查                   | 结果                 | 说明                                                         |
| :--------------------- | :------------------- | :----------------------------------------------------------- |
| `git diff --check`     | 通过                 | 两个工作树均无 whitespace 错误。                             |
| Python AST 解析        | 通过                 | 所有修改的 Python 文件均可解析。                             |
| Serialization 定向单测 | 通过                 | 远端容器运行 4 项 `TestSaveLoad` 测试，全部通过。            |
| Wrapper benchmark 单测 | 通过                 | 远端容器运行 10 项测试，全部通过。                           |
| NPU 真实设备冒烟       | 通过                 | JIT map_location 加载到 `npu:0`；snapshot 文件成功生成且非空。 |
| `lintrunner -a`        | 通过（跳过 PYREFLY） | 两个分支均对精确文件运行并成功应用全部补丁；远端 `lintrunner init` 无法完成，因此跳过 `PYREFLY`。 |
| 远端推送               | 完成                 | 已推送到 `origin/decouple_serialization` 和 `origin/decouple_wrapper_benchmark`。 |

<details open="" style="box-sizing: border-box; margin: 10px 0px; padding: 10px 14px; border: 0.666667px solid rgb(219, 226, 234); border-radius: 9px;"><summary style="box-sizing: border-box; cursor: pointer; font-weight: 600;">远端测试命令与关键输出</summary><pre style="box-sizing: border-box; overflow: auto; padding: 16px; border-radius: 9px; background: rgb(17, 24, 39); color: rgb(229, 237, 247);"><code style="box-sizing: border-box; background: transparent; border-radius: 5px; padding: 0px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;">python test/test_jit.py \
  TestSaveLoad.test_validate_map_location_fast_path \
  TestSaveLoad.test_validate_map_location_unregistered_device_module_passthrough \
  TestSaveLoad.test_validate_map_location_invalid_type \
  TestSaveLoad.test_validate_map_location_non_cuda_device_validation
# Ran 4 tests ... OK

python test/inductor/test_wrapper_benchmark.py
# Ran 10 tests ... OK

python /tmp/remote_npu_smoke.py
# jit_map_location=npu:0
# snapshot_size=5621</code></pre></details>

最终状态：用户审核精确 diff → 远端测试与定向 lint → 分别提交 → 推送同名远端分支，流程已完成。