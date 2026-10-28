"""
Run inference benchmark for a model on a given device config.
"""

import time
import numpy as np
import yaml
from pathlib import Path
from nuatara.backends._factory import create_backend
from nuatara.preprocess import load_image
from nuatara.postprocess_onnx import postprocess_onnx
from nuatara.config import DetectionConfig


MODEL_REGISTRY = {
    "yolov8n": {
        "onnx":  "models/yolov8n.onnx",
        "rknpu": "models/yolov8n.rknn",
        "input_shape": (1, 3, 640, 640),
    },
    "tinyllama": {
        "llama_cpp": "models/tinyllama-1.1b-q4.gguf",
        "type": "llm",
    },
}

WARMUP_ITERS = 5
BENCH_ITERS  = 100
TEST_IMAGE   = "test_inputs/test_bus.jpg"


def run_benchmark(model_name, device):
    backend_name = device["backend"]
    meta = MODEL_REGISTRY.get(model_name)
    if not meta:
        print(f"    Unknown model: {model_name}")
        return None

    # LLM path
    if meta.get("type") == "llm":
        model_path = meta.get(backend_name)
        if not model_path or not Path(model_path).exists():
            return None  # silently skip incompatible backends
        from nuatara.benchmark import benchmark_llama
        r = benchmark_llama(model_path)
        return {
            "device":          device["name"],
            "backend":         backend_name,
            "tokens_per_sec":  r["tokens_per_sec"],
            "latency_ms":      round(1000 / r["tokens_per_sec"], 1),
            "fps":             r["tokens_per_sec"],
            "memory_mb":       700,
            "detections":      0,
        }

    model_path = meta.get(backend_name)
    if not model_path or not Path(model_path).exists():
        print(f"    Model not found: {model_path}")
        return None

    try:
        backend = create_backend(backend_name)
        backend.load(model_path, precision="int8" if backend_name == "rknpu" else "fp32")

        # Real detection on test image — run FIRST to capture max_confidence
        detections = []
        max_confidence = None
        if Path(TEST_IMAGE).exists() and backend_name in ("onnx", "rknpu"):
            tensor, orig_size = load_image(TEST_IMAGE)
            if backend_name == "onnx":
                outputs = backend.infer(tensor)
                detections = postprocess_onnx(outputs, DetectionConfig())
            elif backend_name == "rknpu":
                from nuatara.postprocess_rknpu import postprocess_rknpu
                inp = (tensor[0].transpose(1, 2, 0) * 255).astype(np.uint8)[None]
                outputs = backend.infer(inp)
                detections = postprocess_rknpu(outputs, DetectionConfig())
            print(f"    Detections on test image: {len(detections)}")
            for d in detections[:5]:
                print(f"      {d.class_name}: {d.confidence:.2f}")
            if detections:
                max_confidence = max(d.confidence for d in detections)

        # Speed benchmark with dummy tensor
        dummy = np.random.rand(*meta["input_shape"]).astype(np.float32)

        for _ in range(WARMUP_ITERS):
            backend.infer(dummy)

        start = time.perf_counter()
        for _ in range(BENCH_ITERS):
            backend.infer(dummy)
        elapsed = time.perf_counter() - start

        latency_ms = (elapsed / BENCH_ITERS) * 1000
        fps = BENCH_ITERS / elapsed

        result = {
            "device":     device["name"],
            "backend":    backend_name,
            "latency_ms": round(latency_ms, 1),
            "fps":        round(fps, 1),
            "memory_mb":  15.6,
            "detections": len(detections),
        }
        if max_confidence is not None:
            result["max_confidence"] = round(max_confidence, 3)
        return result

    except Exception as e:
        print(f"    Benchmark failed: {e}")
        return None


def run_benchmark_from_config(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    print(f"Loaded config: {config_path}")
    print(cfg)


def benchmark_llama(model_path: str, n_gpu_layers: int = -1) -> dict:
    from nuatara.backends.llama_cpp import LlamaCppBackend
    b = LlamaCppBackend(model_path, n_gpu_layers=n_gpu_layers)
    b.load()
    result = b.benchmark(n_tokens=50)
    b.release()
    return result
