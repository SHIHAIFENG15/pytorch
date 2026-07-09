FSDP 测试用例硬件耦合解耦改造指南
本文档面向 PyTorch FSDP 测试用例的硬件耦合问题，按 test_coupling_analysis.xlsx 中“耦合类型”sheet 的修改建议进行大类归并，给出每个耦合小类的改造建议、代码示例及社区 PR 参考。

原始分析数据：test_coupling_analysis.xlsx
社区 PR 参考：community_pr_reference.md
类型一：@skip_if_lt_x_gpu(N) 仅识别 cuda/hpu/xpu，新硬件会被误跳过
核心问题：
测试入口的 skip 装饰器或 world_size 计算硬编码了 CUDA/HPU/XPU 三类设备。
新加速器既无法被 @skip_if_lt_x_gpu 识别，也无法通过 torch.cuda.device_count() 获得正确的 world size。
改造原则：
所有设备数量判断都通过 torch.accelerator.current_accelerator() 获取device_type后再通过device_count()获取。
修改建议
@skip_if_lt_x_gpu 在 torch/testing/_internal/common_distributed.py 中硬编码了 CUDA/HPU/XPU 三类设备。推荐在 common_distributed.py 中新增通用的 @skip_if_lt_x_devices(x, *, allow_cpu=False)，供所有分布式测试（包括 FSDP）使用。

推荐做法：
在 torch/testing/_internal/common_distributed.py 中新增 @skip_if_lt_x_devices(x, *, allow_cpu=False)。
装饰器内部通过 torch.accelerator.current_accelerator() 获取当前加速器类型；返回 None 时按 "cpu" 处理。
对加速器调用 device_count() 判断可用设备数。
显式处理 CPU 路径，保留 allow_cpu=True 的退化运行语义。
直接抛 unittest.SkipTest，不再复用 TEST_SKIPS 和 _maybe_handle_skip_if_lt_x_gpu，降低理解成本。作为测试基类的MultiProcessTestCase 的 run_test 会捕获 unittest.SkipTest 并以通用退出码结束子进程；MultiThreadedTestCase 会在线程中直接捕获。
保留现有 @skip_if_lt_x_gpu 不变，新测试逐步迁移到 @skip_if_lt_x_devices，避免一次性改动过大导致行为回归。
原有代码示例
# test/distributed/_composable/fsdp/test_fully_shard_training.py
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu

class TestFullyShardTraining(FSDPTestMultiThread):
    @skip_if_lt_x_gpu(2)
    def test_train_parity_multi_group(self):
        ...
@skip_if_lt_x_gpu 当前实现：

# torch/testing/_internal/common_distributed.py
def skip_if_lt_x_gpu(x, *, allow_cpu=False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available() and torch.cuda.device_count() >= x:
                return func(*args, **kwargs)
            if TEST_HPU and torch.hpu.device_count() >= x:
                return func(*args, **kwargs)
            if TEST_XPU and torch.xpu.device_count() >= x:
                return func(*args, **kwargs)
            if allow_cpu and not (torch.cuda.is_available() or TEST_HPU or TEST_XPU):
                return func(*args, **kwargs)
            test_skip = TEST_SKIPS[f"multi-gpu-{x}"]
            if not _maybe_handle_skip_if_lt_x_gpu(args, test_skip.message):
                sys.exit(test_skip.exit_code)
        return wrapper
    return decorator
修改后代码示例
在 common_distributed.py 中新增通用装饰器：

# torch/testing/_internal/common_distributed.py
import torch
import unittest
from functools import wraps

def skip_if_lt_x_devices(x, *, allow_cpu=False):
    """Skip if fewer than x devices available for the current accelerator.

    Unlike @skip_if_lt_x_gpu, this does not hard-code cuda/hpu/xpu.
    It uses torch.accelerator.current_accelerator() to determine the device type.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"

            # CPU path: allow running a degenerate version if explicitly requested
            if device_type == "cpu":
                if allow_cpu:
                    return func(*args, **kwargs)
                raise unittest.SkipTest("requires accelerator")

            # Accelerator path
            device_module = torch.get_device_module(device_type)
            if device_module.is_available() and device_module.device_count() >= x:
                return func(*args, **kwargs)

            raise unittest.SkipTest(
                f"requires at least {x} {device_type} devices, "
                f"found {mod.device_count()}"
            )

        return wrapper

    return decorator
测试中使用：
# test/distributed/_composable/fsdp/test_fully_shard_training.py
from torch.testing._internal.common_distributed import skip_if_lt_x_devices

class TestFullyShardTraining(FSDPTestMultiThread):
    @skip_if_lt_x_devices(2)
    def test_train_parity_multi_group(self):
        ...
社区 PR 参考
尚无直接对应 PR，因为社区改造多集中在 instantiate_device_type_tests 框架，而 FSDP 分布式测试使用 MultiProcessTestCase。可参考以下思路作为过渡或补充：

PR #176717 (test_unary_ufuncs.py)：引入 @onlyAccelerator 装饰器，提供“跳过 CPU/meta”的通用加速器判断。

URL: https://github.com/pytorch/pytorch/pull/176717
参考方法：新增通用装饰器替代硬编码设备判断。
PR #178135 (bypass_device_restrictions)、PR #180820 (test_exclusions)、PR #176264 (op_overrides)、PR #181703 (op_allowlist)：在无法一次性删除硬编码分支时，提供框架层绕过/排除机制。

URL: https://github.com/pytorch/pytorch/pull/178135
URL: https://github.com/pytorch/pytorch/pull/180820
URL: https://github.com/pytorch/pytorch/pull/176264
URL: https://github.com/pytorch/pytorch/pull/181703
参考方法：作为过渡方案，让新硬件先跑起来，再逐步替换 @skip_if_lt_x_gpu。
类型二：设备特定 API/专用测试解耦
核心问题：
测试代码中硬编码 device="cuda"、torch.cuda.synchronize()、DeviceType.CUDA、torch.cuda.Stream()、torch.device("cuda", rank) 等设备特定 API 调用。
NCCL 环境变量设置、NCCL 日志断言、NCCL 特定功能测试与 CUDA/NCCL 强绑定
改造原则：
通用计算/同步逻辑使用 torch.get_device_module(device_type) 或 torch.accelerator.* 替代 torch.cuda.*。
真正依赖 CUDA 特有机制（CUDA Graph、CUDA multicast、CUDA 硬件能力检查等）的测试保留为 CUDA 专用测试，不作修改。
小类 1：原硬编码 CUDA 的逻辑可以支持通用 device_type
判断标准
把原来写死的 device="cuda"、DeviceType.CUDA、torch.cuda.synchronize() 等换成基于当前 device_type 的调用，语义仍然成立，不会引入 CUDA 特有行为。

推荐做法
在测试文件顶部通过 torch.accelerator.current_accelerator() 获取当前 device_type，无加速器时回退到 "cpu"（若测试文件已有device_type获取逻辑，则直接使用device_type）。
将硬编码设备 API 替换为基于 device_type 的通用调用：
device="cuda" -> device=device_type
torch.device("cuda", rank) -> torch.device(device_type, rank)
torch.cuda.synchronize() -> torch.get_device_module(device_type).synchronize()
torch.cuda.set_device(rank) -> torch.get_device_module(device_type).set_device(rank)（若设备模块提供）
DeviceType.CUDA -> 由 device_type 字符串转换得到的 DeviceType 枚举值（如 getattr(DeviceType, device_type.upper(), None)）
保留 @skip_if_lt_x_devices 等通用设备数量装饰器（已在类型一中覆盖）。
小类 2：原逻辑确实只适用于特定类型设备
判断标准：例如代码依赖 CUDA 特有 API（如 torch.cuda.CUDAGraph、torch.cuda.Stream()、CUDA Graph capture/replay、CUDA 特定硬件能力等），无法通过 torch.get_device_module(device_type) 泛化。

推荐做法
完全不做改动，与社区 PR 保持一致。 社区对真正 CUDA 专属的测试采取 left unchanged 策略（如 PR #184593 中的 torch.cuda._sleep()、nvtx、pin_memory、DataParallel 等）。
保留已有的 CUDA-only 装饰器（如 @skip_if_lt_x_gpu、@unittest.skipIf(not TEST_CUDA_GRAPH, ...)、@onlyCUDA），不再额外抽取类或删除矛盾分支。
若未来 PyTorch 提供统一的 accelerator graph/stream 抽象，再考虑将此类测试泛化。
小类 1 示例：通用设备 API 硬编码替换为 device_type
原有代码：

# test/distributed/_composable/fsdp/test_fully_shard_comm.py
from torch._C._autograd import DeviceType
from torch.distributed._symmetric_memory import _SymmetricMemory

class TestFullyShardAllocFromPG(FSDPTest):
    @skip_if_lt_x_gpu(2)
    def test_fully_shard_alloc_from_pg(self):
        # Run this check inside test instead of using @requires_multicast_support().
        # The decorator would trigger an initialization of SymmMem allocator
        # when Python statically initializes classes in this file, causing
        # SymmMem to fix the allocate backend to "CUDA".
        if not _SymmetricMemory.has_multicast_support(DeviceType.CUDA, 0):
            self.skipTest("multicast support is not available")
        ...
        inp = torch.randint(0, model_args.vocab_size, (2, 16), device="cuda")
        ...
        torch.distributed.barrier()
        torch.cuda.synchronize()
修改后代码：

# test/distributed/_composable/fsdp/test_fully_shard_comm.py
import torch
from torch._C._autograd import DeviceType
from torch.distributed._symmetric_memory import _SymmetricMemory
from torch.testing._internal.common_distributed import skip_if_lt_x_devices

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"

class TestFullyShardAllocFromPG(FSDPTest):
    @skip_if_lt_x_devices(2)
    def test_fully_shard_alloc_from_pg(self):
        # Run this check inside test instead of using @requires_multicast_support().
        # The decorator would trigger an initialization of SymmMem allocator
        # when Python statically initializes classes in this file, causing
        # SymmMem to fix the allocate backend to "CUDA".
        device_type_enum = getattr(DeviceType, device_type.upper(), None)
        if device_type_enum is None or not _SymmetricMemory.has_multicast_support(
            device_type_enum, 0
        ):
            self.skipTest("multicast support is not available")
        ...
        inp = torch.randint(
            0, model_args.vocab_size, (2, 16), device=device_type
        )
        ...
        torch.distributed.barrier()
        torch.get_device_module(device_type).synchronize()
说明：

_SymmetricMemory.has_multicast_support 的 C++ 实现接受任意 c10::DeviceType，仅在该设备类型未注册 allocator 时返回 false。因此该检查本身不是 CUDA 特有逻辑，可以随 device_type 参数化。当前只有 CUDA 注册了 allocator，所以非 CUDA 设备会在运行时 skip，测试本身不再硬编码 CUDA。
getattr(DeviceType, device_type.upper(), None) 这种字符串转枚举的写法在 PyTorch 现有代码中已有先例，例如 torch/_inductor/utils.py:461 的 _do_bench_using_profiling：
device_type = get_gpu_type()              # "cuda" / "xpu" 等字符串
device_type_upper = device_type.upper()
...
getattr(DeviceType, device_type_upper)    # 转为 DeviceType.CUDA / DeviceType.XPU
示例中使用 getattr(..., None) 而不是直接 getattr(...)，是为了让未知设备类型 graceful skip；若严格照搬现有代码风格可去掉默认值。
小类 2 示例：GPU 特有逻辑 —— 以test_two_layer_fully_shard_cudagraph 与NCCL LOG为例
原有代码(`test_two_layer_fully_shard_cudagraph)
# test/distributed/_composable/fsdp/test_fully_shard_training.py
class TestFullyShardTraining(FSDPTest):
    @skip_if_lt_x_gpu(2, allow_cpu=True)
    @unittest.skipIf(
        not TEST_CUDA_GRAPH, "CUDA >= 11.0 or ROCM >= 5.3 required for graphs"
    )
    def test_two_layer_fully_shard_cudagraph(self):
        if device_type.type == "cuda":
            torch.cuda.set_device(self.rank)
        device = torch.device(device_type.type, self.rank)
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(8, 8, bias=False),
            nn.Linear(8, 8, bias=False),
        ).to(device)
        ...

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            ...
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            ...
修改建议
完全不做改动，与社区 PR 保持一致。

理由：

torch.cuda.Stream()、torch.cuda.CUDAGraph()、torch.cuda.graph() 是 CUDA 专属 API，社区 PR 对这类 API 的测试采取 left unchanged 策略。
当前 PyTorch 没有跨设备的图抽象，强行参数化需要引入 if device_type == "cuda": ... elif device_type == "xpu": ... 等设备名分支，反而增加硬件耦合。
原有代码虽然混有 if device_type.type == "cuda" 和 torch.device(device_type.type, self.rank) 等通用设备判断，但 @unittest.skipIf(not TEST_CUDA_GRAPH, ...) 已经保证该方法只在支持 CUDA Graph 的环境运行，本质上是 CUDA-only 测试。
若未来 PyTorch 提供统一的 accelerator graph API（如 torch.accelerator.graph()），再考虑将此类测试泛化。目前阶段保持原样。

原有代码(NCCL LOG)
# test/distributed/_composable/fsdp/test_fully_shard_comm.py
class TestFullyShardAllocFromPG(FSDPTest):
    MEMORY_REGISTER_RE = (
        "NCCL INFO register comm 0x[0-9a-f]+ buffer 0x[0-9a-f]+ size [0-9]+"
    )

    @classmethod
    def _run(cls, *args, **kwargs):
        cls.nccl_log_dir = tempfile.TemporaryDirectory()
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,ENV,REG"
        os.environ["NCCL_DEBUG_FILE"] = cls.nccl_log_dir.name + "/nccl_log"
        super()._run(*args, **kwargs)

    @skip_if_lt_x_gpu(2)
    def test_fully_shard_alloc_from_pg(self):
        ...
        with open(self.nccl_log_dir.name + "/nccl_log") as f:
            self.assertNotRegex(f.read(), self.MEMORY_REGISTER_RE)
        ...
        with open(self.nccl_log_dir.name + "/nccl_log") as f:
            self.assertRegex(f.read(), self.MEMORY_REGISTER_RE)
修改建议
完全不做改动，与社区 PR 保持一致。

理由：

NCCL 环境变量是 NCCL 专属机制

NCCL_DEBUG、NCCL_DEBUG_SUBSYS、NCCL_DEBUG_FILE 这些变量名本身就是 NCCL 的。
不存在 torch.distributed.set_collective_debug_env() 或 torch.accelerator.set_debug_env() 这类通用 API。
NCCL 日志格式是 NCCL 专属

断言里的 "NCCL INFO ..." 是 NCCL 打印的日志格式。
XCCL、Gloo、UCC 等后端没有等价格式，因此正则表达式无法泛化。
无法通过动态推导把 NCCL 专属调用变成通用调用

get_default_backend_for_device(device_type) 或 torch.accelerator.current_accelerator() 只能告诉你当前加速器/默认后端是什么。
它们不能把一个 NCCL_DEBUG 设置或 NCCL INFO 断言自动转换成 XCCL/Gloo 的等价操作。
因此，任何改造都会停留在“判断当前后端是不是 NCCL，然后执行 NCCL 专属逻辑”这一层，无法真正做到后端无关。

社区 PR 参考
PR #184593 (test_autograd.py)：将 torch.cuda.synchronize()、torch.cuda.memory_allocated()、torch.cuda.set_default_device("cuda")、device="cuda" 等大量替换为 torch.accelerator.* 或参数化 device；对 CUDA 特有 API（nvtx、_sleep、pin_memory、DataParallel 等）保留为 CUDA-only 测试，不强行泛化。

URL: https://github.com/pytorch/pytorch/pull/184593
参考方法：通用设备 API 用 torch.accelerator.* / torch.get_device_module(device_type)；CUDA/NCCL 特有逻辑保留原样。
PR #178336 ([Distributed] Make DDP tests and tensor parallel dependencies backend agnostic)：将 @requires_nccl() 替换为 @requires_accelerator_dist_backend()，并用 torch.accelerator.current_accelerator() 动态推导 DEVICE_TYPE 和 BACKEND，减少对 cuda/nccl 硬编码。

URL: https://github.com/pytorch/pytorch/pull/178336
参考方法：分布式测试应尽量通过动态推导获取后端，避免在测试代码中直接写死后端名；对真正依赖 NCCL 行为的断言仍保持 NCCL-only。
PR #160158 ([1/N] Port 6 fsdp distributed test cases to Intel GPU)：在将 FSDP 测试移植到 Intel GPU 时，把 backend="cpu:gloo,cuda:nccl" 改为 backend="cpu:gloo,xpu:xccl"， reviewers 建议使用 get_default_backend_for_device 进一步去设备名化。

URL: https://github.com/pytorch/pytorch/pull/160158
参考方法：后端字符串不应硬编码设备名；若无法完全消除后端名，则按后端隔离的测试保持原样。
PR #163063 (Restore environment after NcclUserBufferRegistrationTest)：NCCL 专属测试设置 NCCL_ALGO=NVLS 后恢复环境，说明 NCCL 环境变量操作只应出现在 NCCL 专属测试范围内。

URL: https://github.com/pytorch/pytorch/pull/163063
参考方法：NCCL 环境变量设置属于 NCCL 专属测试逻辑，不强行泛化。
类型三：模型/Tensor 显式设备名硬编码解耦
核心问题：
现有测试用例中存在MLP / Transformer / nn.Linear 等在 CPU 默认创建模型，再通过显式使用设备名字符串（如 "cuda"、"xpu"、"cpu"）将模型搬移至特定设备上的问题。
改造原则：
只修改调用点显式硬编码设备名的地方，用动态推导的 device_type 替换。
对于 to(device_type)、默认 CPU 创建后再 .to(device_type) 等已参数化路径，不作修改。
对于 genuinely CPU-only 的测试（模型始终停留在 CPU，"cpu" 是其预期行为），不作修改。
修改建议
将 .cuda() 替换为 .to(device_type)；将 .cpu() 替换为 .to("cpu")（若确实需要 CPU）或 .to(device_type)（若只是设备搬移）。

原有代码示例
# test/distributed/_composable/fsdp/test_fully_shard_state_dict.py
class TestFullyShardStateDict(FSDPTest):
    @skip_if_lt_x_gpu(2)
    def test_dp_state_dict_cpu_offload(self):
        ...
        if not mutate_after_state_dict:
            ...
        else:
            model = model.cpu()
            model = model.cuda()
            ...

        torch.manual_seed(42 + self.rank)
        inp = torch.rand(mlp_dim, mlp_dim, device="cuda")
        ...
修改后代码示例
# test/distributed/_composable/fsdp/test_fully_shard_state_dict.py
import torch
from torch.testing._internal.common_distributed import skip_if_lt_x_devices

device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestFullyShardStateDict(FSDPTest):
    @skip_if_lt_x_devices(2)
    def test_dp_state_dict_cpu_offload(self):
        ...
        if not mutate_after_state_dict:
            ...
        else:
            model = model.to("cpu")
            model = model.to(device_type)
            ...

        torch.manual_seed(42 + self.rank)
        device = torch.device(device_type, self.rank)
        inp = torch.rand(mlp_dim, mlp_dim, device=device)
        ...
说明：

.cuda() 必须改为 .to(device_type)，因为它硬编码 CUDA。
.cpu() 改为 .to("cpu") 只是为了统一写法；但此处测试语义是 CPU offload，属于测试逻辑需要，保留 "cpu" 作为目标设备是合理的。
同一代码块中的 device="cuda" 随 .cuda() 一起替换为 device_type。
社区 PR 参考
PR #184593 (test_autograd.py)：将大量 device="cuda"、.cuda()、model.cuda() 替换为参数化 device。

URL: https://github.com/pytorch/pytorch/pull/184593
参考方法：显式设备名统一替换为动态推导的 device_type；保留真正需要 CPU 路径的 "cpu" 字符串。
PR #184261 (test_serialization.py)：将 torch.device("cuda")、device="cuda" 替换为参数化 device。

URL: https://github.com/pytorch/pytorch/pull/184261
参考方法：tensor/module 创建时的 device 参数统一参数化。
PR #184315 (test_functional.py)：将 (x * y).cuda() 改为 (x * y).to(device)。

URL: https://github.com/pytorch/pytorch/pull/184315
参考方法：.cuda() 替换为 .to(device)。
PR #183728 (test_optim.py)：将 "cuda" in optim_info.supports_fused_on、params_cuda = [p.to(device="cuda")] 改为基于参数化 device 的判断。

URL: https://github.com/pytorch/pytorch/pull/183728
参考方法：避免在字符串层面硬编码 "cuda"，统一使用 _get_device_type(device) 或 device_type。
PR #184192 (test_lazy_modules.py)：将 if TEST_CUDA: device = "cuda" 手动分支删除，改为由框架注入 device 参数。

URL: https://github.com/pytorch/pytorch/pull/184192
参考方法：删除显式设备名分支，用参数化 device 替代。