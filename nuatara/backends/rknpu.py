"""
backends/rknpu.py — RKNNBackend for Rockchip NPU (RK3588).
ONLY imports: rknn_lite, numpy, stdlib. Never imports onnxruntime.

Tested: Khadas Edge2, RK3588, librknnrt 2.5.0, rknn-toolkit-lite2 2.5.0
Input:  (1, 640, 640, 3) uint8 NHWC RGB
Output: 9 tensors — 3 FPN scales × (DFL boxes, cls scores, obj)
IMPORTANT: RKNN runtime applies sigmoid internally. Do NOT sigmoid again in postprocess.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
from nuatara.types import BackendInfo

NPU_CORE_AUTO  = 0
NPU_CORE_0     = 1
NPU_CORE_1     = 2
NPU_CORE_2     = 4
NPU_CORE_0_1   = 3   # dual-core — best for YOLOv8n
NPU_CORE_0_1_2 = 7

class RKNNBackend:
    def __init__(self, core: int = NPU_CORE_0_1) -> None:
        self._core   = core
        self._rknn   = None
        self._loaded = False
        self._info:  Optional[BackendInfo] = None
        self._input_shape: Optional[tuple] = None

    def load(self, model_path: str, precision: str) -> None:
        try:
            from rknn.api import RKNNLite
        except ImportError as e:
            raise RuntimeError(
                "rknn-toolkit-lite2 not available. "
                "RKNNBackend only runs on Rockchip NPU hardware.\n"
                f"Original: {e}") from e

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"RKNN model not found: {path.resolve()}")

        precision = precision.lower().strip()
        if precision not in ("fp16","int8"):
            raise ValueError(f"RKNNBackend supports fp16/int8 only, got '{precision}'")

        rknn = RKNNLite(verbose=False)
        if rknn.load_rknn(str(path)) != 0:
            raise RuntimeError(f"rknn.load_rknn() failed for {path}")
        if rknn.init_runtime(core_mask=self._core) != 0:
            raise RuntimeError("rknn.init_runtime() failed")

        try:
            attrs = rknn.get_input_attr()
            self._input_shape = tuple(attrs[0].dims) if attrs else (1,640,640,3)
        except Exception:
            self._input_shape = (1,640,640,3)

        try:
            sdk_ver = str(rknn.get_sdk_version().api_version).strip()
        except Exception:
            sdk_ver = "2.5.0"

        core_name = {0:"NPU_CORE_AUTO",1:"NPU_CORE_0",2:"NPU_CORE_1",
                     4:"NPU_CORE_2",3:"NPU_CORE_0_1",7:"NPU_CORE_0_1_2"
                    }.get(self._core, f"NPU_CORE_{self._core}")

        self._rknn   = rknn
        self._loaded = True
        self._info   = BackendInfo(
            backend_name="RKNPU v2", device=f"RK3588 {core_name}",
            precision=precision, sdk_version=sdk_ver,
            model_path=str(path.resolve()))

    def infer(self, tensor: np.ndarray) -> list[np.ndarray]:
        if not self._loaded or self._rknn is None:
            raise RuntimeError("infer() called before load()")
        outputs = self._rknn.inference(inputs=[tensor], data_format="nhwc")
        if outputs is None:
            raise RuntimeError("rknn.inference() returned None")
        return outputs

    def warmup(self, iterations: int = 10) -> None:
        if not self._loaded or self._rknn is None:
            raise RuntimeError("warmup() called before load()")
        dummy = np.zeros(self._input_shape or (1,640,640,3), dtype=np.uint8)
        for _ in range(iterations):
            self._rknn.inference(inputs=[dummy], data_format="nhwc")

    @property
    def info(self) -> BackendInfo:
        if self._info is None:
            raise RuntimeError("info accessed before load()")
        return self._info

    def release(self) -> None:
        if self._rknn is not None:
            self._rknn.release(); self._rknn = None; self._loaded = False

    def __enter__(self): return self
    def __exit__(self, *_): self.release()
    def __repr__(self): return f"RKNNBackend({self._info})" if self._loaded else "RKNNBackend(not loaded)"
