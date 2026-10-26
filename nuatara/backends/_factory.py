"""
backends/_factory.py — THE only place that maps backend name → class.
Adding a new chip = one entry here + one new file. Nothing else changes.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from backends.base import InferenceBackend

def create_backend(name: str, **kwargs) -> "InferenceBackend":
    key = name.lower().strip()
    if key in ("rknpu","rknn","rockchip"):
        from backends.rknpu import RKNNBackend
        return RKNNBackend(**kwargs)
    if key in ("onnx","onnxruntime","ort","cpu"):
        from backends.onnx_rt import OnnxRuntimeBackend
        return OnnxRuntimeBackend(**kwargs)
    raise ValueError(
        f"Unknown backend '{name}'. Available: 'rknpu', 'onnx'. "
        f"To add a new backend, see backends/_factory.py.")

AVAILABLE_BACKENDS: frozenset = frozenset({"rknpu", "onnx"})
