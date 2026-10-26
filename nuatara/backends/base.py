"""
backends/base.py — InferenceBackend Protocol.
ZERO SDK IMPORTS. Runs clean on any machine.
"""
from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np
from engine.types import BackendInfo

@runtime_checkable
class InferenceBackend(Protocol):
    """
    The L2 boundary. Every backend implements exactly this interface.
    Rules:
      1. load() before infer() or warmup(). infer() before load() → RuntimeError.
      2. infer() is stateless — same input, same output, no side effects.
      3. warmup() runs real inferences on dummy data. Not optional on NPU.
      4. info is frozen after load(). Does not change.
      5. Backends must NOT import each other.
      6. infer() need not be thread-safe — engine serialises calls.
    """
    def load(self, model_path: str, precision: str) -> None: ...
    def infer(self, tensor: np.ndarray) -> list[np.ndarray]: ...
    def warmup(self, iterations: int = 10) -> None: ...
    @property
    def info(self) -> BackendInfo: ...

SUPPORTED_PRECISIONS = frozenset({"fp32", "fp16", "int8"})

def validate_precision(precision: str) -> str:
    p = precision.lower().strip()
    if p not in SUPPORTED_PRECISIONS:
        raise ValueError(f"Unsupported precision '{precision}'. Must be one of: {sorted(SUPPORTED_PRECISIONS)}")
    return p
