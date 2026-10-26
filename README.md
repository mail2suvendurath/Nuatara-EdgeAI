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
```

## Commands

```bash
nuatara recommend yolov8n                          # get recommendation + score
nuatara demo yolov8n --video input.mp4             # run detection on video (Mac)
nuatara demo yolov8n --video input.mp4 --backend rknpu  # run on Khadas NPU
```

## Benchmark Results (October 2026)

| Device | Backend | FPS | Score |
|--------|---------|-----|-------|
| MacBook Air M2 | ONNX / ANE | 184 | 93 |
| Khadas Edge 2 | RKNN INT8 NPU | 48 | 78 |
| Khadas Edge 2 | ONNX CPU | 7.7 | 41 |

## Milestones

- ✅ Milestone 1: YOLOv8n vision model — Mac + Khadas Edge 2
- 🔲 Milestone 2: TinyLlama 1.1B language model — llama.cpp backend
- 🔲 Milestone 3: Quantization-aware recommendation (INT8 vs FP16 trade-off)

## Supported Hardware

| Device | Runtime |
|--------|---------|
| MacBook Air M2 | ONNX Runtime (Apple Neural Engine) |
| Khadas Edge 2 (RK3588) | RKNN Toolkit 2.5.0 (NPU) |

## Docs

- [Vision & Milestones](docs/architecture/nuatara-vision.md)
- [RKNN Lessons Learnt](docs/results/INT8-backbone-result-oct2026.md)
