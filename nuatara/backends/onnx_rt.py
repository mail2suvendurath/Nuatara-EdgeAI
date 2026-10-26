"""
backends/onnx_rt.py — OnnxRuntimeBackend for CPU/GPU (laptop, Pi, x86).
ONLY imports: onnxruntime, numpy, stdlib. Never imports rknn or tensorrt.

Output (YOLOv8n ONNX): single tensor (1, 84, 8400) float32 merged format.
Input: (1, 3, 640, 640) float32 NCHW, normalised [0,1], RGB.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
from engine.types import BackendInfo

_DEFAULT_PROVIDERS = [
    "CUDAExecutionProvider",
    "CoreMLExecutionProvider",
    "CPUExecutionProvider",
]

class OnnxRuntimeBackend:
    def __init__(self, providers: Optional[list] = None) -> None:
        self._providers = providers or _DEFAULT_PROVIDERS
        self._session = None
        self._input_name: Optional[str] = None
        self._input_shape: Optional[tuple] = None
        self._loaded = False
        self._info: Optional[BackendInfo] = None

    def load(self, model_path: str, precision: str) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError(
                "onnxruntime not installed. Run: pip install onnxruntime\n"
                f"Original: {e}") from e

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX model not found: {path.resolve()}")

        precision = precision.lower().strip()
        if precision not in ("fp32","fp16","int8"):
            raise ValueError(f"Unknown precision '{precision}'")

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3

        available = ort.get_available_providers()
        providers  = [p for p in self._providers if p in available] or ["CPUExecutionProvider"]

        session = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
        inputs = session.get_inputs()
        self._input_name  = inputs[0].name
        self._input_shape = tuple(d if isinstance(d,int) else 1 for d in inputs[0].shape)

        active   = session.get_providers()[0] if session.get_providers() else "CPUExecutionProvider"
        device   = {"CUDAExecutionProvider":"CUDA GPU",
                    "CoreMLExecutionProvider":"Apple Neural Engine",
                    "CPUExecutionProvider":"CPU"}.get(active, active)

        self._session = session
        self._loaded  = True
        self._info    = BackendInfo(
            backend_name="ONNX Runtime", device=device,
            precision=precision, sdk_version=ort.__version__,
            model_path=str(path.resolve()))

    def infer(self, tensor: np.ndarray) -> list[np.ndarray]:
        if not self._loaded or self._session is None:
            raise RuntimeError("infer() called before load()")
        return self._session.run(None, {self._input_name: tensor})

    def warmup(self, iterations: int = 10) -> None:
        if not self._loaded or self._session is None:
            raise RuntimeError("warmup() called before load()")
        shape = self._input_shape or (1, 3, 640, 640)
        dummy = np.zeros(shape, dtype=np.float32)
        for _ in range(iterations):
            self._session.run(None, {self._input_name: dummy})

    @property
    def info(self) -> BackendInfo:
        if self._info is None:
            raise RuntimeError("info accessed before load()")
        return self._info

    def __enter__(self): return self
    def __exit__(self, *_): self._session = None; self._loaded = False
    def __repr__(self): return f"OnnxRuntimeBackend({self._info})" if self._loaded else "OnnxRuntimeBackend(not loaded)"
