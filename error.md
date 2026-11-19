1. 测试用例：test/inductor/test_aot_inductor_arrayref.py::AOTInductorTestABICompatibleCpuWithStackAllocation::test_kernel_profile_repeated_kernel_numel_cpu_with_stack_allocation：

报错：

```
Worker killed by signal (SIGABRT); case log carries the worker stdout tail captured by the parent
```

2. test/inductor/test_aot_inductor_arrayref.py::AOTInductorTestABICompatibleCpuWithStackAllocation::test_kernel_profile_scatter_fallback_arg_order_cpu_with_stack_allocation：

报错：

```
torch._inductor.exc.InductorError: CppCompileError: C++ compile error

Command:
g++ /tmp/tmpx1un98u8/czyrguxj6du2hxi2tn6kfta5o77qulwwksta4gi5eolwalqry5bv/cl42vfupymvyamdoginzsamlezjaafukqedtblanztyp7iss27qv.wrapper.cpp -D TORCH_INDUCTOR_CPP_WRAPPER -D STANDALONE_TORCH_HEADER -D TORCH_INDUCTOR_PRECOMPILE_HEADERS -D C10_USING_CUSTOM_GENERATED_MACROS -D CPU_CAPABILITY_AVX512 -O1 -DNDEBUG -fno-omit-frame-pointer -g1 -fno-trapping-math -funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros -fno-finite-math-only -fno-unsafe-math-optimizations -fmath-errno -ffp-contract=off -fexcess-precision=fast -fno-tree-loop-vectorize -fPIC -Wall -std=c++20 -Wno-unused-variable -Wno-unknown-pragmas -pedantic -fopenmp -include /tmp/torchinductor_root/precompiled_headers/civcqumquhryxccfaj65734maksp6foenlegtqgkpbhe3zfwpvaq.h -I/opt/conda/envs/py_3.11/include/python3.11 -I/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/include -I/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/include/torch/csrc/api/include -mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma -mavx512vnni -mavx512vl -c -o /tmp/tmpx1un98u8/czyrguxj6du2hxi2tn6kfta5o77qulwwksta4gi5eolwalqry5bv/cl42vfupymvyamdoginzsamlezjaafukqedtblanztyp7iss27qv.wrapper.o

Output:
In file included from /opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/include/torch/csrc/inductor/aoti_runtime/interface.h:8,
                 from /opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/include/torch/csrc/inductor/aoti_include/common.h:8,
                 from /opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/include/torch/csrc/inductor/aoti_include/cpu.h:4,
                 from /tmp/torchinductor_root/precompiled_headers/civcqumquhryxccfaj65734maksp6foenlegtqgkpbhe3zfwpvaq.h:1:
/tmp/tmpx1un98u8/czyrguxj6du2hxi2tn6kfta5o77qulwwksta4gi5eolwalqry5bv/cl42vfupymvyamdoginzsamlezjaafukqedtblanztyp7iss27qv.wrapper.cpp: In member function ‘void torch::aot_inductor::AOTInductorModel::run_impl(AtenTensorOpaque**, AtenTensorOpaque**, torch::aot_inductor::DeviceStreamType, AOTIProxyExecutorHandle)’:
/tmp/tmpx1un98u8/czyrguxj6du2hxi2tn6kfta5o77qulwwksta4gi5eolwalqry5bv/cl42vfupymvyamdoginzsamlezjaafukqedtblanztyp7iss27qv.wrapper.cpp:1076:33: error: ‘aoti_torch_cpu_scatter_reduce_two’ was not declared in this scope; did you mean ‘aoti_torch_cpu_scatter_reduce_two_out’?
 1076 |     AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cpu_scatter_reduce_two(arg0_1, 1L, arg1_1, arg2_1, "prod", 1, &buf0_handle));
      |                                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/include/torch/csrc/inductor/aoti_runtime/utils.h:27:8: note: in definition of macro ‘AOTI_TORCH_ERROR_CODE_CHECK’
   27 |   if ((call) != AOTI_TORCH_SUCCESS) {                                      \
      |        ^~~~


To execute this test, run the following from the base repo dir:
    python test/inductor/test_aot_inductor_arrayref.py AOTInductorTestABICompatibleCpuWithStackAllocation.test_kernel_profile_scatter_fallback_arg_order_cpu_with_stack_allocation

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0
inductor/test_torchinductor.py:21061: in new_test
    return value(self)
inductor/test_aot_inductor.py:7652: in test_kernel_profile_scatter_fallback_arg_order
    _, code = run_and_get_cpp_code(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/utils.py:4794: in run_and_get_cpp_code
    result = fn(*args, **kwargs)
inductor/test_aot_inductor_utils.py:222: in compile
    package_path = torch._inductor.aoti_compile_and_package(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/__init__.py:161: in aoti_compile_and_package
    return aot_inductor_minifier_wrapper(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/debug.py:1615: in aot_inductor_minifier_wrapper
    raise e
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/debug.py:1585: in aot_inductor_minifier_wrapper
    return func(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/__init__.py:206: in _aoti_compile_and_package_inner
    aoti_files = aot_compile(gm, args, kwargs, options=inductor_configs)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/__init__.py:342: in aot_compile
    return compile_fx_aot(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:2498: in compile_fx_aot
    compiled_artifacts = compile_fx(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:3179: in compile_fx
    return compile_fx(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:3239: in compile_fx
    return _maybe_wrap_and_compile_fx_main(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:3336: in _maybe_wrap_and_compile_fx_main
    return _compile_fx_main(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:3533: in _compile_fx_main
    return inference_compiler(unlifted_gm, example_inputs_)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_functorch/_aot_autograd/schemas.py:1530: in __call__
    output_code = self.compiler_fn(gm, example_inputs)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:3404: in fw_compiler_base
    return compile_fx_forward(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:2966: in compile_fx_forward
    result = inner_compile(
/opt/conda/envs/py_3.11/lib/python3.11/contextlib.py:81: in inner
    return func(*args, **kwds)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:1096: in compile_fx_inner
    compiled_graph = wrap_compiler_debug(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_dynamo/repro/after_aot.py:329: in debug_wrapper
    inner_compiled_fn = compiler_fn(gm, example_inputs)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:1331: in _compile_fx_inner
    raise InductorError(e, currentframe()).with_traceback(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:1323: in _compile_fx_inner
    mb_compiled_graph = fx_codegen_and_compile(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:2189: in fx_codegen_and_compile
    return scheme.codegen_and_compile(gm, example_inputs, inputs_to_check, graph_kwargs)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:1919: in codegen_and_compile
    compiled_fn = AotCodeCompiler.compile(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/codecache.py:3444: in compile
    raise e
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/codecache.py:3438: in compile
    wrapper_builder.build()
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/cpp_builder.py:2890: in build
    run_compile_cmd(build_cmd, cwd=_build_tmp_dir)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/cpp_builder.py:819: in run_compile_cmd
    _run_compile_cmd(cmd_line, cwd)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/cpp_builder.py:814: in _run_compile_cmd
    raise exc.CppCompileError(cmd, output) from e
E   torch._inductor.exc.InductorError: CppCompileError: C++ compile error
E   
E   Command:
E   g++ /tmp/tmpx1un98u8/czyrguxj6du2hxi2tn6kfta5o77qulwwksta4gi5eolwalqry5bv/cl42vfupymvyamdoginzsamlezjaafukqedtblanztyp7iss27qv.wrapper.cpp -D TORCH_INDUCTOR_CPP_WRAPPER -D STANDALONE_TORCH_HEADER -D TORCH_INDUCTOR_PRECOMPILE_HEADERS -D C10_USING_CUSTOM_GENERATED_MACROS -D CPU_CAPABILITY_AVX512 -O1 -DNDEBUG -fno-omit-frame-pointer -g1 -fno-trapping-math -funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros -fno-finite-math-only -fno-unsafe-math-optimizations -fmath-errno -ffp-contract=off -fexcess-precision=fast -fno-tree-loop-vectorize -fPIC -Wall -std=c++20 -Wno-unused-variable -Wno-unknown-pragmas -pedantic -fopenmp -include /tmp/torchinductor_root/precompiled_headers/civcqumquhryxccfaj65734maksp6foenlegtqgkpbhe3zfwpvaq.h -I/opt/conda/envs/py_3.11/include/python3.11 -I/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/include -I/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/include/torch/csrc/api/include -mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma -mavx512vnni -mavx512vl -c -o /tmp/tmpx1un98u8/czyrguxj6du2hxi2tn6kfta5o77qulwwksta4gi5eolwalqry5bv/cl42vfupymvyamdoginzsamlezjaafukqedtblanztyp7iss27qv.wrapper.o
E   
E   Output:
E   In file included from /opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/include/torch/csrc/inductor/aoti_runtime/interface.h:8,
E                    from /opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/include/torch/csrc/inductor/aoti_include/common.h:8,
E                    from /opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/include/torch/csrc/inductor/aoti_include/cpu.h:4,
E                    from /tmp/torchinductor_root/precompiled_headers/civcqumquhryxccfaj65734maksp6foenlegtqgkpbhe3zfwpvaq.h:1:
E   /tmp/tmpx1un98u8/czyrguxj6du2hxi2tn6kfta5o77qulwwksta4gi5eolwalqry5bv/cl42vfupymvyamdoginzsamlezjaafukqedtblanztyp7iss27qv.wrapper.cpp: In member function ‘void torch::aot_inductor::AOTInductorModel::run_impl(AtenTensorOpaque**, AtenTensorOpaque**, torch::aot_inductor::DeviceStreamType, AOTIProxyExecutorHandle)’:
E   /tmp/tmpx1un98u8/czyrguxj6du2hxi2tn6kfta5o77qulwwksta4gi5eolwalqry5bv/cl42vfupymvyamdoginzsamlezjaafukqedtblanztyp7iss27qv.wrapper.cpp:1076:33: error: ‘aoti_torch_cpu_scatter_reduce_two’ was not declared in this scope; did you mean ‘aoti_torch_cpu_scatter_reduce_two_out’?
E    1076 |     AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cpu_scatter_reduce_two(arg0_1, 1L, arg1_1, arg2_1, "prod", 1, &buf0_handle));
E         |                                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
E   /opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/include/torch/csrc/inductor/aoti_runtime/utils.h:27:8: note: in definition of macro ‘AOTI_TORCH_ERROR_CODE_CHECK’
E      27 |   if ((call) != AOTI_TORCH_SUCCESS) {                                      \
E         |        ^~~~
E   
E   
E   To execute this test, run the following from the base repo dir:
E       python test/inductor/test_aot_inductor_arrayref.py AOTInductorTestABICompatibleCpuWithStackAllocation.test_kernel_profile_scatter_fallback_arg_order_cpu_with_stack_allocation
E   
E   This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0
```

3. test/inductor/test_aot_inductor_arrayref.py::AOTInductorTestABICompatibleCpuWithStackAllocation::test_kernel_profile_scatter_fallback_cpu_with_stack_allocation：

报错：

```
torch._inductor.exc.InductorError: CppCompileError: C++ compile error

Command:
g++ /tmp/tmp6uu22fx4/czyrguxj6du2hxi2tn6kfta5o77qulwwksta4gi5eolwalqry5bv/clb2fcnb7kyk3w2mbep7clp7h3mmjl3yaa7av7i242z2x3heyrrw.wrapper.cpp -D TORCH_INDUCTOR_CPP_WRAPPER -D STANDALONE_TORCH_HEADER -D TORCH_INDUCTOR_PRECOMPILE_HEADERS -D C10_USING_CUSTOM_GENERATED_MACROS -D CPU_CAPABILITY_AVX512 -O1 -DNDEBUG -fno-omit-frame-pointer -g1 -fno-trapping-math -funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros -fno-finite-math-only -fno-unsafe-math-optimizations -fmath-errno -ffp-contract=off -fexcess-precision=fast -fno-tree-loop-vectorize -fPIC -Wall -std=c++20 -Wno-unused-variable -Wno-unknown-pragmas -pedantic -fopenmp -include /tmp/torchinductor_root/precompiled_headers/civcqumquhryxccfaj65734maksp6foenlegtqgkpbhe3zfwpvaq.h -I/opt/conda/envs/py_3.11/include/python3.11 -I/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/include -I/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/include/torch/csrc/api/include -mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma -mavx512vnni -mavx512vl -c -o /tmp/tmp6uu22fx4/czyrguxj6du2hxi2tn6kfta5o77qulwwksta4gi5eolwalqry5bv/clb2fcnb7kyk3w2mbep7clp7h3mmjl3yaa7av7i242z2x3heyrrw.wrapper.o

Output:
In file included from /opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/include/torch/csrc/inductor/aoti_runtime/interface.h:8,
                 from /opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/include/torch/csrc/inductor/aoti_include/common.h:8,
                 from /opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/include/torch/csrc/inductor/aoti_include/cpu.h:4,
                 from /tmp/torchinductor_root/precompiled_headers/civcqumquhryxccfaj65734maksp6foenlegtqgkpbhe3zfwpvaq.h:1:
/tmp/tmp6uu22fx4/czyrguxj6du2hxi2tn6kfta5o77qulwwksta4gi5eolwalqry5bv/clb2fcnb7kyk3w2mbep7clp7h3mmjl3yaa7av7i242z2x3heyrrw.wrapper.cpp: In member function ‘void torch::aot_inductor::AOTInductorModel::run_impl(AtenTensorOpaque**, AtenTensorOpaque**, torch::aot_inductor::DeviceStreamType, AOTIProxyExecutorHandle)’:
/tmp/tmp6uu22fx4/czyrguxj6du2hxi2tn6kfta5o77qulwwksta4gi5eolwalqry5bv/clb2fcnb7kyk3w2mbep7clp7h3mmjl3yaa7av7i242z2x3heyrrw.wrapper.cpp:1082:33: error: ‘aoti_torch_cpu_scatter_reduce_two’ was not declared in this scope; did you mean ‘aoti_torch_cpu_scatter_reduce_two_out’?
 1082 |     AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cpu_scatter_reduce_two(arg0_1, 1L, arg1_1, arg2_1, "prod", 1, &buf0_handle));
      |                                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/include/torch/csrc/inductor/aoti_runtime/utils.h:27:8: note: in definition of macro ‘AOTI_TORCH_ERROR_CODE_CHECK’
   27 |   if ((call) != AOTI_TORCH_SUCCESS) {                                      \
      |        ^~~~


To execute this test, run the following from the base repo dir:
    python test/inductor/test_aot_inductor_arrayref.py AOTInductorTestABICompatibleCpuWithStackAllocation.test_kernel_profile_scatter_fallback_cpu_with_stack_allocation

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0
inductor/test_torchinductor.py:21061: in new_test
    return value(self)
inductor/test_aot_inductor.py:7828: in test_kernel_profile_scatter_fallback
    _, code = run_and_get_cpp_code(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/utils.py:4794: in run_and_get_cpp_code
    result = fn(*args, **kwargs)
inductor/test_aot_inductor_utils.py:222: in compile
    package_path = torch._inductor.aoti_compile_and_package(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/__init__.py:161: in aoti_compile_and_package
    return aot_inductor_minifier_wrapper(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/debug.py:1615: in aot_inductor_minifier_wrapper
    raise e
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/debug.py:1585: in aot_inductor_minifier_wrapper
    return func(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/__init__.py:206: in _aoti_compile_and_package_inner
    aoti_files = aot_compile(gm, args, kwargs, options=inductor_configs)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/__init__.py:342: in aot_compile
    return compile_fx_aot(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:2498: in compile_fx_aot
    compiled_artifacts = compile_fx(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:3179: in compile_fx
    return compile_fx(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:3239: in compile_fx
    return _maybe_wrap_and_compile_fx_main(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:3336: in _maybe_wrap_and_compile_fx_main
    return _compile_fx_main(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:3533: in _compile_fx_main
    return inference_compiler(unlifted_gm, example_inputs_)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_functorch/_aot_autograd/schemas.py:1530: in __call__
    output_code = self.compiler_fn(gm, example_inputs)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:3404: in fw_compiler_base
    return compile_fx_forward(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:2966: in compile_fx_forward
    result = inner_compile(
/opt/conda/envs/py_3.11/lib/python3.11/contextlib.py:81: in inner
    return func(*args, **kwds)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:1096: in compile_fx_inner
    compiled_graph = wrap_compiler_debug(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_dynamo/repro/after_aot.py:329: in debug_wrapper
    inner_compiled_fn = compiler_fn(gm, example_inputs)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:1331: in _compile_fx_inner
    raise InductorError(e, currentframe()).with_traceback(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:1323: in _compile_fx_inner
    mb_compiled_graph = fx_codegen_and_compile(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:2189: in fx_codegen_and_compile
    return scheme.codegen_and_compile(gm, example_inputs, inputs_to_check, graph_kwargs)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:1919: in codegen_and_compile
    compiled_fn = AotCodeCompiler.compile(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/codecache.py:3444: in compile
    raise e
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/codecache.py:3438: in compile
    wrapper_builder.build()
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/cpp_builder.py:2890: in build
    run_compile_cmd(build_cmd, cwd=_build_tmp_dir)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/cpp_builder.py:819: in run_compile_cmd
    _run_compile_cmd(cmd_line, cwd)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/cpp_builder.py:814: in _run_compile_cmd
    raise exc.CppCompileError(cmd, output) from e
E   torch._inductor.exc.InductorError: CppCompileError: C++ compile error
E   
E   Command:
E   g++ /tmp/tmp6uu22fx4/czyrguxj6du2hxi2tn6kfta5o77qulwwksta4gi5eolwalqry5bv/clb2fcnb7kyk3w2mbep7clp7h3mmjl3yaa7av7i242z2x3heyrrw.wrapper.cpp -D TORCH_INDUCTOR_CPP_WRAPPER -D STANDALONE_TORCH_HEADER -D TORCH_INDUCTOR_PRECOMPILE_HEADERS -D C10_USING_CUSTOM_GENERATED_MACROS -D CPU_CAPABILITY_AVX512 -O1 -DNDEBUG -fno-omit-frame-pointer -g1 -fno-trapping-math -funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros -fno-finite-math-only -fno-unsafe-math-optimizations -fmath-errno -ffp-contract=off -fexcess-precision=fast -fno-tree-loop-vectorize -fPIC -Wall -std=c++20 -Wno-unused-variable -Wno-unknown-pragmas -pedantic -fopenmp -include /tmp/torchinductor_root/precompiled_headers/civcqumquhryxccfaj65734maksp6foenlegtqgkpbhe3zfwpvaq.h -I/opt/conda/envs/py_3.11/include/python3.11 -I/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/include -I/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/include/torch/csrc/api/include -mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma -mavx512vnni -mavx512vl -c -o /tmp/tmp6uu22fx4/czyrguxj6du2hxi2tn6kfta5o77qulwwksta4gi5eolwalqry5bv/clb2fcnb7kyk3w2mbep7clp7h3mmjl3yaa7av7i242z2x3heyrrw.wrapper.o
E   
E   Output:
E   In file included from /opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/include/torch/csrc/inductor/aoti_runtime/interface.h:8,
E                    from /opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/include/torch/csrc/inductor/aoti_include/common.h:8,
E                    from /opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/include/torch/csrc/inductor/aoti_include/cpu.h:4,
E                    from /tmp/torchinductor_root/precompiled_headers/civcqumquhryxccfaj65734maksp6foenlegtqgkpbhe3zfwpvaq.h:1:
E   /tmp/tmp6uu22fx4/czyrguxj6du2hxi2tn6kfta5o77qulwwksta4gi5eolwalqry5bv/clb2fcnb7kyk3w2mbep7clp7h3mmjl3yaa7av7i242z2x3heyrrw.wrapper.cpp: In member function ‘void torch::aot_inductor::AOTInductorModel::run_impl(AtenTensorOpaque**, AtenTensorOpaque**, torch::aot_inductor::DeviceStreamType, AOTIProxyExecutorHandle)’:
E   /tmp/tmp6uu22fx4/czyrguxj6du2hxi2tn6kfta5o77qulwwksta4gi5eolwalqry5bv/clb2fcnb7kyk3w2mbep7clp7h3mmjl3yaa7av7i242z2x3heyrrw.wrapper.cpp:1082:33: error: ‘aoti_torch_cpu_scatter_reduce_two’ was not declared in this scope; did you mean ‘aoti_torch_cpu_scatter_reduce_two_out’?
E    1082 |     AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cpu_scatter_reduce_two(arg0_1, 1L, arg1_1, arg2_1, "prod", 1, &buf0_handle));
E         |                                 ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
E   /opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/include/torch/csrc/inductor/aoti_runtime/utils.h:27:8: note: in definition of macro ‘AOTI_TORCH_ERROR_CODE_CHECK’
E      27 |   if ((call) != AOTI_TORCH_SUCCESS) {                                      \
E         |        ^~~~
E   
E   
E   To execute this test, run the following from the base repo dir:
E       python test/inductor/test_aot_inductor_arrayref.py AOTInductorTestABICompatibleCpuWithStackAllocation.test_kernel_profile_scatter_fallback_cpu_with_stack_allocation
E   
E   This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0
```





test/inductor/test_aoti_torchbind_constants.py::TestTorchbindAOTIDeviceCPU::test_custom_objs_exposed_through_loader_cpu：

```
torch._inductor.exc.InductorError: AssertionError: expected OpOverload or registered extension type, got WithEffects

To execute this test, run the following from the base repo dir:
    python test/inductor/test_aoti_torchbind_constants.py TestTorchbindAOTIDeviceCPU.test_custom_objs_exposed_through_loader_cpu

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0
inductor/test_aoti_torchbind_constants.py:48: in test_custom_objs_exposed_through_loader
    pt2_path = torch._inductor.aoti_compile_and_package(ep)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/__init__.py:161: in aoti_compile_and_package
    return aot_inductor_minifier_wrapper(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/debug.py:1615: in aot_inductor_minifier_wrapper
    raise e
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/debug.py:1585: in aot_inductor_minifier_wrapper
    return func(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/__init__.py:206: in _aoti_compile_and_package_inner
    aoti_files = aot_compile(gm, args, kwargs, options=inductor_configs)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/__init__.py:342: in aot_compile
    return compile_fx_aot(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:2498: in compile_fx_aot
    compiled_artifacts = compile_fx(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:3179: in compile_fx
    return compile_fx(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:3239: in compile_fx
    return _maybe_wrap_and_compile_fx_main(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:3336: in _maybe_wrap_and_compile_fx_main
    return _compile_fx_main(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:3533: in _compile_fx_main
    return inference_compiler(unlifted_gm, example_inputs_)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_functorch/_aot_autograd/schemas.py:1530: in __call__
    output_code = self.compiler_fn(gm, example_inputs)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:3404: in fw_compiler_base
    return compile_fx_forward(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:2966: in compile_fx_forward
    result = inner_compile(
/opt/conda/envs/py_3.11/lib/python3.11/contextlib.py:81: in inner
    return func(*args, **kwds)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:1096: in compile_fx_inner
    compiled_graph = wrap_compiler_debug(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_dynamo/repro/after_aot.py:329: in debug_wrapper
    inner_compiled_fn = compiler_fn(gm, example_inputs)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:1331: in _compile_fx_inner
    raise InductorError(e, currentframe()).with_traceback(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:1323: in _compile_fx_inner
    mb_compiled_graph = fx_codegen_and_compile(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:2189: in fx_codegen_and_compile
    return scheme.codegen_and_compile(gm, example_inputs, inputs_to_check, graph_kwargs)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:1896: in codegen_and_compile
    wrapper_code, kernel_code = graph.codegen_with_cpp_wrapper()
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/graph.py:2841: in codegen_with_cpp_wrapper
    return self.codegen()
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/graph.py:3091: in codegen
    self.scheduler.codegen()
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/scheduler.py:12652: in codegen
    else self._codegen(self.nodes)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/scheduler.py:12986: in _codegen
    self.codegen_extern_call(node)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/scheduler.py:11930: in codegen_extern_call
    scheduler_node.codegen(V.graph.wrapper_code)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/scheduler.py:3941: in codegen
    return self.node.codegen(wrapper)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/ir.py:10172: in codegen
    exported_args = self.export_extern_kernel_node()
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/ir.py:10011: in export_extern_kernel_node
    named_arguments = serializer.serialize_inputs(target, args, kwargs)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_export/serde/serialize.py:1207: in serialize_inputs
    raise AssertionError(
E   torch._inductor.exc.InductorError: AssertionError: expected OpOverload or registered extension type, got WithEffects
E   
E   To execute this test, run the following from the base repo dir:
E       python test/inductor/test_aoti_torchbind_constants.py TestTorchbindAOTIDeviceCPU.test_custom_objs_exposed_through_loader_cpu
E   
E   This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0
```

test/inductor/test_aoti_torchbind_constants.py::TestTorchbindAOTIDeviceCPU::test_custom_objs_empty_when_no_torchbind_cpu

```
Worker killed by signal (SIGABRT); case log carries the worker stdout tail captured by the parent
```

test/inductor/test_aoti_torchbind_constants.py::TestTorchbindAOTIDevicePRIVATEUSE1::test_custom_objs_empty_when_no_torchbind_npu：

```
torch._inductor.exc.InductorError: KeyError: 'GridNpu'

To execute this test, run the following from the base repo dir:
    python test/inductor/test_aoti_torchbind_constants.py TestTorchbindAOTIDevicePRIVATEUSE1.test_custom_objs_empty_when_no_torchbind_npu

This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0
inductor/test_aoti_torchbind_constants.py:116: in test_custom_objs_empty_when_no_torchbind
    pt2_path = torch._inductor.aoti_compile_and_package(ep)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/__init__.py:161: in aoti_compile_and_package
    return aot_inductor_minifier_wrapper(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/debug.py:1615: in aot_inductor_minifier_wrapper
    raise e
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/debug.py:1585: in aot_inductor_minifier_wrapper
    return func(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/__init__.py:206: in _aoti_compile_and_package_inner
    aoti_files = aot_compile(gm, args, kwargs, options=inductor_configs)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/__init__.py:342: in aot_compile
    return compile_fx_aot(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:2498: in compile_fx_aot
    compiled_artifacts = compile_fx(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:3179: in compile_fx
    return compile_fx(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:3239: in compile_fx
    return _maybe_wrap_and_compile_fx_main(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:3336: in _maybe_wrap_and_compile_fx_main
    return _compile_fx_main(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:3533: in _compile_fx_main
    return inference_compiler(unlifted_gm, example_inputs_)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_functorch/_aot_autograd/schemas.py:1530: in __call__
    output_code = self.compiler_fn(gm, example_inputs)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:3404: in fw_compiler_base
    return compile_fx_forward(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:2966: in compile_fx_forward
    result = inner_compile(
/opt/conda/envs/py_3.11/lib/python3.11/contextlib.py:81: in inner
    return func(*args, **kwds)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:1096: in compile_fx_inner
    compiled_graph = wrap_compiler_debug(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_dynamo/repro/after_aot.py:329: in debug_wrapper
    inner_compiled_fn = compiler_fn(gm, example_inputs)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:1331: in _compile_fx_inner
    raise InductorError(e, currentframe()).with_traceback(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:1323: in _compile_fx_inner
    mb_compiled_graph = fx_codegen_and_compile(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:2189: in fx_codegen_and_compile
    return scheme.codegen_and_compile(gm, example_inputs, inputs_to_check, graph_kwargs)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/compile_fx.py:1896: in codegen_and_compile
    wrapper_code, kernel_code = graph.codegen_with_cpp_wrapper()
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/graph.py:2788: in codegen_with_cpp_wrapper
    return self.codegen()
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/graph.py:3098: in codegen
    result = self.wrapper_code.generate(self.is_inference)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch_npu/_inductor/codegen/cpp_wrapper_npu.py:694: in generate
    return super().generate(is_inference)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/codegen/cpp_wrapper_gpu.py:1677: in generate
    return super().generate(is_inference)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/codegen/cpp_wrapper_cpu.py:1582: in generate
    return super().generate(is_inference)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/codegen/wrapper.py:3600: in generate
    return self._generate(is_inference)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/codegen/cpp_wrapper_gpu.py:1368: in _generate
    return super()._generate(is_inference)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/codegen/wrapper.py:3712: in _generate
    self.finalize_prefix()
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch_npu/_inductor/codegen/cpp_wrapper_npu.py:802: in finalize_prefix
    super().finalize_prefix()
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/codegen/cpp_wrapper_gpu.py:1721: in finalize_prefix
    kernel.generate(self)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch_npu/_inductor/codegen/cpp_wrapper_npu.py:110: in generate
    super().generate(wrapper)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/codegen/cpp_wrapper_gpu.py:536: in generate
    return self.generate_lazy(wrapper)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/codegen/cpp_wrapper_gpu.py:1000: in generate_lazy
    self._generate_lazy_grid(prefix)
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/codegen/cpp_wrapper_gpu.py:698: in _generate_lazy_grid
    grid = GridExpr.from_meta_lazy(
/opt/conda/envs/py_3.11/lib/python3.11/site-packages/torch/_inductor/runtime/triton_heuristics.py:5685: in from_meta_lazy
    grid_cls = globals()[grid_type]
E   torch._inductor.exc.InductorError: KeyError: 'GridNpu'
E   
E   To execute this test, run the following from the base repo dir:
E       python test/inductor/test_aoti_torchbind_constants.py TestTorchbindAOTIDevicePRIVATEUSE1.test_custom_objs_empty_when_no_torchbind_npu
E   
E   This message can be suppressed by setting PYTORCH_PRINT_REPRO_ON_FAILURE=0
```

