# Nuatara-EdgeAI

Hardware-agnostic AI deployment toolkit for Edge devices.

## What It Does

Given a model and available hardware, Nuatara:
- Discovers AI accelerators (ANE, NPU, CPU)
- Benchmarks every viable deployment option
- Recommends the optimal runtime with a score

## Quick Start

```bash
git clone https://github.com/mail2suvendurath/Nuatara-EdgeAI
cd Nuatara-EdgeAI
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
nuatara recommend yolov8n
nuatara recommend tinyllama
```

## Commands

```bash
nuatara recommend yolov8n                               # vision model
nuatara recommend tinyllama                             # language model
nuatara demo yolov8n --video input.mp4                 # detection video (Mac)
nuatara demo yolov8n --video input.mp4 --backend rknpu # detection video (Khadas)
```

## Benchmark Results (October 2026)

### Vision — YOLOv8n
| Device | Backend | FPS | Score |
|--------|---------|-----|-------|
| MacBook Air M2 | ONNX / ANE | 184 | 93 |
| Khadas Edge 2 | RKNN INT8 NPU | 48 | 78 |
| Khadas Edge 2 | ONNX CPU | 7.7 | 41 |

### Language — TinyLlama 1.1B
| Device | Backend | Tokens/sec | Score |
|--------|---------|-----------|-------|
| MacBook Air M2 | llama.cpp Metal | 63 | 55 |
| Khadas Edge 2 | llama.cpp CPU | 26 | 55 |

## Milestones

- ✅ Milestone 1: YOLOv8n vision model — Mac + Khadas Edge 2
- ✅ Milestone 2: TinyLlama 1.1B language model — llama.cpp backend
- 🔲 Milestone 3: Quantization-aware recommendation (INT8 vs FP16 trade-off)

## Supported Hardware

| Device | Runtimes |
|--------|---------|
| MacBook Air M2 | ONNX Runtime (ANE), llama.cpp (Metal) |
| Khadas Edge 2 (RK3588) | RKNN NPU, llama.cpp (CPU) |

## Docs
- [Vision & Milestones](docs/architecture/nuatara-vision.md)
- [Benchmark Results & RKNN Lessons](docs/results/INT8-backbone-result-oct2026.md)
