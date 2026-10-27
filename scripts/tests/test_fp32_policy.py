"""Exercise CUDA/FP32 guards with a stub runtime; no GPU session is created."""

from contextlib import ExitStack
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fp32_policy import configure


class FP32PolicyTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)

    def runtime(self, available=True, actual=True, tf32="0"):
        class Runtime:
            def __init__(self, path, runtime_config=None):
                self.inference_session = SimpleNamespace(
                    get_providers=lambda: ["CUDAExecutionProvider"] if actual else ["CPUExecutionProvider"],
                    get_provider_options=lambda: {"CUDAExecutionProvider": {"use_tf32": tf32}})
                self.closed = False

            def close(self):
                self.closed = True

        onnx = SimpleNamespace(_get_ort_provider=lambda device=None: ["CUDAExecutionProvider"] if available else
                               ["CPUExecutionProvider"], OnnxRuntimeSession=Runtime)
        session = ModuleType("dx_modelzoo.session")
        session.onnx_session = onnx
        self.stack.enter_context(patch.dict(sys.modules, {"dx_modelzoo.session": session}))
        self.stack.enter_context(patch.dict(os.environ))
        configure()
        return onnx

    def test_tf32_is_disabled_and_configuration_is_idempotent(self):
        onnx = self.runtime()
        self.assertEqual(os.environ["NVIDIA_TF32_OVERRIDE"], "0")
        self.assertEqual(onnx._get_ort_provider("gpu"), [("CUDAExecutionProvider", {"use_tf32": "0"})])
        initializer = onnx.OnnxRuntimeSession.__init__
        configure()
        self.assertIs(onnx.OnnxRuntimeSession.__init__, initializer)
        onnx.OnnxRuntimeSession("fixture", SimpleNamespace(device="gpu"))

    def test_missing_cuda_provider_is_rejected(self):
        onnx = self.runtime(available=False)
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            onnx._get_ort_provider("gpu")
        self.assertEqual(onnx._get_ort_provider("cpu"), ["CPUExecutionProvider"])

    def test_cuda_initialization_fallback_is_rejected(self):
        onnx = self.runtime(actual=False)
        with self.assertRaisesRegex(RuntimeError, "CUDA initialization failed"):
            onnx.OnnxRuntimeSession("fixture", SimpleNamespace(device="gpu"))

    def test_ignored_tf32_option_is_rejected(self):
        onnx = self.runtime(tf32="1")
        with self.assertRaisesRegex(RuntimeError, "FP32 policy was not applied"):
            onnx.OnnxRuntimeSession("fixture", SimpleNamespace(device="gpu"))


if __name__ == "__main__":
    unittest.main()
