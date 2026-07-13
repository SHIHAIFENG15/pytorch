方案1
instantiate_device_type_tests中用的GPU_TYPE新增PrivateUse1

方案2：GPU_TYPE/HAS_GPU/HAS_MULTIGPU/HAS_GPU_AND_TRITON/requires_gpu_with_enough_memory泛化
用例中存在大量这两个变量的使用，引用自：

from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
​
其内部实际分别调用了

# GPU_TYPE
torch._inductor.utils.get_gpu_type
​
# HAS_GPU
HAS_CUDA_AND_TRITON = torch.cuda.is_available() and HAS_TRITON

HAS_XPU_AND_TRITON = torch.xpu.is_available() and HAS_TRITON

HAS_MPS = torch.mps.is_available()

HAS_GPU = HAS_CUDA_AND_TRITON or HAS_XPU_AND_TRITON
​
当前get_gpu_type内部写死了GPU设备类型，可以考虑使用deviceinterface或者device_codegens去获取已注册的设备。torch.cuda.is_available()/requires_gpu_with_enough_memory硬编码可以使用torch.accelerator去解决，HAS_TRITON当前完全硬编码了设备范围，可以直接从已有的deviceinterface的方法中获取

@staticmethod
def is_triton_capable(device: torch.types.Device = None) -> bool:
    """
    Returns True if the device has Triton support, False otherwise, even if
    the appropriate Triton backend is not available.
    """
    return False
​
该方案统一在特性的RFC（Issue #189138 · pytorch/pytorch）中处理，用例解决需依赖那个RFC

方案3：requires_cuda_and_triton改写
用例中大量存在

@requires_cuda_and_triton
​
@unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
​
其中requires_cuda_and_triton定义为

HAS_CUDA_AND_TRITON = torch.cuda.is_available() and HAS_TRITON
​
需要去掉requires_cuda_and_triton或改成HAS_TRITON，推荐改写成HAS_TRITON，此修改依赖特性RFC对HAS_TRITON的整改

方案4：新增instantiate_device_type_tests
用例中仅用

run_tests()
​
需要加入

instantiate_device_type_tests
​
支持多设备

方案5：替换“cuda”硬编码字段
源码中存在大量的“cuda”字段硬编码

torch.randn(4, 4, 4, device="cuda")
​
修改方案：

在测试文件顶部通过 torch.accelerator.current_accelerator() 获取当前 device_type（若测试文件已有device_type获取逻辑，则直接使用device_type）。

# 若测试文件顶部已有 device_type 赋值，则无需添加本行
 device_type = getattr(torch.accelerator.current_accelerator(), "type", None)

 torch.randn(4, 4, 4, device=device_type)
​
方案6：AOTI动态获取接口
针对硬编码设备的用例：

if device == "cpu":
    return torch._C._aoti.AOTIModelContainerRunnerCpu(so_path, 1)
elif device == "xpu":
    return torch._C._aoti.AOTIModelContainerRunnerXpu(so_path, 1, device)
elif device == "mps":
    return torch._C._aoti.AOTIModelContainerRunnerMps(so_path, 1)
else:
    return torch._C._aoti.AOTIModelContainerRunnerCuda(so_path, 1, device)
​
做局部拼接实现动态加载：

@staticmethod
   def legacy_load_runner(device, so_path: str) -> "AOTIModelContainerRunner":
       if IS_FBCODE:
           # ... 省略 FBCODE 代码 ...
           pass

       # 按命名规则拼接 runner 类名：AOTIModelContainerRunner + 设备名首字母大写
       runner_cls_name = f"AOTIModelContainerRunner{device.capitalize()}"
       runner_cls = getattr(torch._C._aoti, runner_cls_name, None)

       if runner_cls is None:
           raise RuntimeError(
               f"Unsupported device '{device}': no AOTIModelContainerRunner class found "
               f"in torch._C._aoti. Expected class name: {runner_cls_name}"
           )

       # Cpu 和 Mps 的构造器签名不同（无 device 参数）
       if device in ("cpu", "mps"):
           return runner_cls(so_path, 1)
       else:
           return runner_cls(so_path, 1, device)

​
方案7：flex attention专属方案
问题根源：

​设备判别变量分散定义​：IS_FLEX_ATTENTION_CPU_PLATFORM_SUPPORTED、_CUDA_、_XPU_、_MPS_ 是在 common_device_type.py 中各写各的，每新增一个设备就要新增一个变量
​**test_device 的构造是 if/elif 链**​：第 267-281 行的链式判断每加一个设备就要改
​跳过装饰器也是每个设备一个​：skipCPUIf、skipCUDAIf、skipXPUIf、skipMPSIf
方案：
在 common_device_type.py 中，将这些平台支持变量改为一个注册字典：

# common_device_type.py
_flex_attention_device_support = {
    "cpu": not IS_MACOS and torch.cpu._is_avx2_supported() 
            and os.getenv("ATEN_CPU_CAPABILITY") != "default",
    "cuda": torch.cuda.is_available() and has_triton() 
            and torch.cuda.get_device_capability() >= (8, 0),
    "xpu": torch.xpu.is_available() and has_triton(),
    "mps": torch.mps.is_available(),
}

def is_flex_attention_supported(device: str) -> bool:
    """查询某设备是否支持 flex attention"""
    support = _flex_attention_device_support.get(device)
    if support is not None:
        return support
    # privateuse1 设备：通过 device_interface 判断
    try:
        from torch._dynamo.device_interface import get_interface_for_device
        interface = get_interface_for_device(device)
        return interface.is_available() and has_triton()
    except (RuntimeError, ImportError):
        return False

def flex_attention_supported_devices() -> tuple[str, ...]:
    """返回所有支持 flex attention 的设备列表"""
    return tuple(
        dev for dev, supported in _flex_attention_device_support.items()
        if supported
    )

​
当前的 skip_on_cuda、skip_on_cpu、skip_on_xpu 等也要泛化。可以引入一个统一装饰器工厂：

# common_device_type.py
def skip_device_if(skip_condition: bool, device: str, reason: str):
    """统一的设备跳过装饰器工厂"""
    skip_map = {
        "cpu": skipCPUIf,
        "cuda": skipCUDAIf,
        "xpu": skipXPUIf,
        "mps": skipMPSIf,
    }
    matcher = skip_map.get(device)
    if matcher:
        return matcher(skip_condition, reason)
    # privateuse1 兜底：用 unittest.skipIf
    return unittest.skipIf(skip_condition, reason)

​
这样之前的跳过代码

skip_on_cpu = skipCPUIf(True, "Not supported on CPU")
skip_on_cuda = skipCUDAIf(True, "Not supported on CUDA")
skip_on_xpu = skipXPUIf(True, "Not supported on Intel GPU")
skip_on_mps = skipMPSIf(True, "Not supported on MPS")

​
统一为一个装饰器

skip_device = lambda d: skip_device_if(not is_flex_attention_supported(d), d, f"Not supported on {d}")
@skip_device("cpu")
def test_something(self, device):
   ...
​
方案8：TEST_CUDA/TEST_MPS/TEST_XPU/dtypesIfCUDA
用例中大量使用这些变量在装饰器中，核心方案，将变量使用TEST_PRIVATEUSE1和dtypesIfPRIVATEUSE1补充，已经存在的变量。
核心思路：在dtypesIfCUDA的对等位置加上dtypesIfPRIVATEUSE1，dtype类型与CUDA一致，同时新增换机变量用于PRIVATEUSE1设备自己声明实际支持的dtype类型：

class dtypesIfPRIVATEUSE1(dtypes):
    def __init__(self, *args):
        dtypes = os.getenv('TORCH_TEST_OVERRIDE_DTYPE'):
        if dtypes:
            dtypes = split(dtypes)
         elsse:
            dtypes = *args   
        super().__init__(dtypes, device_type=torch._C._get_privateuse1_backend_name())
​
方案9：dist.init_process_group硬编码