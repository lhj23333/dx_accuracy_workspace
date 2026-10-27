"""FP32 launch policy for the local DX-ModelZoo 0.10.2 installation."""

import os


def configure():
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    from dx_modelzoo.session import onnx_session

    if getattr(onnx_session, "_local_fp32_policy", False):
        return

    original_providers = onnx_session._get_ort_provider
    original_init = onnx_session.OnnxRuntimeSession.__init__

    def gpu_requested(device):
        return device == "gpu" or isinstance(device, int) or (isinstance(device, list) and bool(device))

    def providers(device=None):
        result = original_providers(device)
        if not gpu_requested(device):
            return result
        configured = []
        for provider in result:
            name, options = provider if isinstance(provider, tuple) else (provider, {})
            if name == "CUDAExecutionProvider":
                options = {**options, "use_tf32": "0"}
                configured.append((name, options))
            else:
                configured.append(provider)
        if not any((p[0] if isinstance(p, tuple) else p) == "CUDAExecutionProvider" for p in configured):
            raise RuntimeError("GPU FP32 requested, but CUDAExecutionProvider is unavailable")
        return configured

    def initialize(self, path, runtime_config=None):
        original_init(self, path, runtime_config)
        if gpu_requested(getattr(runtime_config, "device", None)):
            actual = self.inference_session.get_providers()
            if "CUDAExecutionProvider" not in actual:
                self.close()
                raise RuntimeError(f"CUDA initialization failed; actual providers: {actual}")
            options = self.inference_session.get_provider_options()["CUDAExecutionProvider"]
            if options.get("use_tf32") != "0":
                self.close()
                raise RuntimeError(f"FP32 policy was not applied: {options}")

    onnx_session._get_ort_provider = providers
    onnx_session.OnnxRuntimeSession.__init__ = initialize
    onnx_session._local_fp32_policy = True
