"""
engine/session.py — Hardware-agnostic inference session.

Before: hardcoded ONNXCPUBackend, only worked on laptop.
After:  routes through the backend factory — same run.py works on Khadas (rknpu)
        and laptop (onnx) by changing one line in the YAML config.

Usage (unchanged from before):
    session = NuataraSession(config)
    output  = session.run(tensor)
"""

import os
from backends import create_backend


class NuataraSession:
    def __init__(self, config: dict):
        self.config = config
        self.backend = self._load_backend()

    def _load_backend(self):
        target     = self.config.get("target", "cpu")
        backend_name = self.config.get("backend", _target_to_backend(target))
        model_path = self._resolve_model_path()
        precision  = self.config.get("precision", "fp32")

        backend = create_backend(backend_name)
        backend.load(model_path, precision)
        return backend

    def _resolve_model_path(self) -> str:
        model = self.config.get("model", "yolov8n")
        # If it already looks like a path, use it as-is
        if os.sep in model or model.endswith(".onnx") or model.endswith(".rknn"):
            return model
        # Otherwise treat as a model name and look in models/
        return f"models/{model}.onnx"

    def run(self, input_tensor):
        return self.backend.infer(input_tensor)

    @property
    def backend_name(self) -> str:
        return type(self.backend).__name__


def _target_to_backend(target: str) -> str:
    """Map legacy 'target' values to backend names for backwards compatibility."""
    t = target.lower()
    if t in ("cpu", "laptop", "laptop-x86", "x86"):
        return "onnx"
    if t in ("khadas", "khadas-edge2", "rk3588", "npu"):
        return "rknpu"
    return "onnx"   # safe default
