# Owner(s): ["module: functorch"]

import os
import tempfile
from unittest.mock import patch

import torch
from torch._functorch import benchmark_utils
from torch.testing._internal.common_utils import run_tests, TestCase


class _FakeProf:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def export_chrome_trace(self, filename: str) -> None:
        with open(filename, "w") as handle:
            handle.write('{"traceEvents":[]}')


class TestFunctorchBenchmarkUtils(TestCase):
    def _dump(self, devices=None, num_runs=1):
        calls: list[object] = []

        def fake_sync(device=None, /):
            calls.append(device)

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(benchmark_utils, "profile", return_value=_FakeProf()),
            patch.object(torch.accelerator, "is_available", return_value=True),
            patch.object(
                torch.accelerator,
                "current_accelerator",
                return_value=torch.device("cuda"),
            ),
            patch.object(torch.accelerator, "synchronize", side_effect=fake_sync),
        ):
            kwargs = {}
            if devices is not None:
                kwargs["devices"] = devices
            benchmark_utils.dump_chrome_trace(
                lambda x: x,
                (0,),
                os.path.join(tmp, "t.json"),
                torch.enable_grad(),
                [],
                num_runs=num_runs,
                **kwargs,
            )
        return calls

    def test_devices_cpu_does_not_sync(self):
        before = benchmark_utils.synchronize
        calls = self._dump(devices=["cpu"])
        self.assertEqual(calls, [])
        self.assertIs(benchmark_utils.synchronize, before)

    def test_default_devices_syncs_matching_cuda_device(self):
        calls = self._dump()
        self.assertEqual(len(calls), 8)
        self.assertTrue(all(c == torch.device("cuda") for c in calls))

    def test_cuda_sentinel_falls_back_when_accelerator_type_differs(self):
        calls: list[object] = []

        def fake_sync(device=None, /):
            calls.append(device)

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(benchmark_utils, "profile", return_value=_FakeProf()),
            patch.object(torch.accelerator, "is_available", return_value=True),
            patch.object(
                torch.accelerator,
                "current_accelerator",
                return_value=torch.device("xpu"),
            ),
            patch.object(torch.accelerator, "synchronize", side_effect=fake_sync),
        ):
            benchmark_utils.dump_chrome_trace(
                lambda x: x,
                (0,),
                os.path.join(tmp, "t.json"),
                torch.enable_grad(),
                [],
                num_runs=1,
            )
        self.assertEqual(len(calls), 8)
        self.assertTrue(all(c is None for c in calls))

    def test_does_not_rebind_module_synchronize(self):
        before = benchmark_utils.synchronize
        self._dump(devices=["cuda"])
        self.assertIs(benchmark_utils.synchronize, before)
        before()

    def test_explicit_device_index_is_passed_through(self):
        calls = self._dump(devices=["cuda:1"])
        self.assertEqual(len(calls), 8)
        self.assertTrue(all(c == torch.device("cuda:1") for c in calls))

    def test_cpu_then_accelerator_does_not_leak_global_sync(self):
        before = benchmark_utils.synchronize
        cpu_calls = self._dump(devices=["cpu"])
        gpu_calls = self._dump(devices=["cuda"])
        self.assertEqual(cpu_calls, [])
        self.assertEqual(len(gpu_calls), 8)
        self.assertIs(benchmark_utils.synchronize, before)


if __name__ == "__main__":
    run_tests()
