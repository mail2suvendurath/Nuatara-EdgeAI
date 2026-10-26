# Nuatara-EdgeAI

Hardware-aware AI deployment advisor for edge devices.

## What It Does

Given a model and your available hardware, Nuatara:
- Discovers available AI runtimes on each device
- Benchmarks every viable deployment alternative
- Scores and recommends the optimal execution path

nuatara recommend yolov8n

MacBook Air M2: ONNX Runtime (ANE) 5.3ms 188 FPS Score: 91

Khadas Edge 2: RKNN INT8 23.7ms 42 FPS Score: 78

Recommendation:
Develop on MacBook → fast iteration, no quantization needed

Deploy on Khadas → 6× lower power, production edge hardware


## Status

| Milestone | Target | Status |
|-----------|--------|--------|
| YOLOv8n on MacBook + Khadas | Nov 2026 | 🔧 In progress |
| TinyLlama on MacBook + Khadas | Nov 2026 | 📋 Planned |

## Devices Supported

- MacBook (Apple Silicon — ANE via CoreML)
- Khadas Edge 2 (RK3588 NPU via RKNN)

## License

MIT
