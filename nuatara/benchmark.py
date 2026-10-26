"""
Run inference benchmark for a model on a given device config.
"""

import time
import numpy as np
import yaml
from pathlib import Path
from nuatara.backends._factory import create_backend


MODEL_REGISTRY = {
    "yolov8n": {
        "onnx": "models/yolov8n.onnx",
        "rknpu": "models/yolov8n.rknn",
        "input_shape": (1, 3, 640, 640),
    },
    "tinyllama": {
        "onnx": None,
        "rknpu": "models/tinyllama.rknn",
        "input_shape": None,
    },
}

WARMUP_ITERS = 5
BENCH_ITERS = 100


def run_benchmark(model_name, device):
    config_path = device["config"]
    backend_name = device["backend"]

    if not Path(config_path).exists():
        return None

    meta = MODEL_REGISTRY.get(model_name)
    if not meta:
        print(f"    Unknown model: {model_name}")
        return None

    model_path = meta.get(backend_name)
    if not model_path or not Path(model_path).exists():
        print(f"    Model not found for {backend_name}: {model_path}")
        return None

    try:
        backend = create_backend(backend_name)
        backend.load(model_path, precision="int8" if backend_name == "rknpu" else "fp32")

        input_shape = meta["input_shape"]
        dummy = np.random.rand(*input_shape).astype(np.float32)

        # Warmup
        for _ in range(WARMUP_ITERS):
            backend.infer(dummy)

        # Benchmark
        start = time.perf_counter()
        for _ in range(BENCH_ITERS):
            backend.infer(dummy)
        elapsed = time.perf_counter() - start

        latency_ms = (elapsed / BENCH_ITERS) * 1000
        fps = BENCH_ITERS / elapsed

        return {
            "device": device["name"],
            "backend": backend_name,
            "latency_ms": round(latency_ms, 1),
            "fps": round(fps, 1),
        }

    except Exception as e:
        print(f"    Benchmark failed: {e}")
        return None


def run_benchmark_from_config(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    print(f"Loaded config: {config_path}")
    print(cfg)
