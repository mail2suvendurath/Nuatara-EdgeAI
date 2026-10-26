# Nuatara-EdgeAI — Product Vision
*Last updated: October 2026*

---

## What It Is

Nuatara-EdgeAI is a feasibility and benchmarking toolkit for developers deploying
AI models on edge hardware.

A developer brings a model. Nuatara answers three questions:

1. **Can it run here?** — feasibility check across target hardware
2. **How well does it run?** — latency, throughput, memory footprint
3. **Where should I deploy it?** — scored recommendation across all viable options

The developer does not need to know ONNX Runtime from RKNN from llama.cpp.
Nuatara handles runtime selection, format conversion, and benchmarking.
The developer gets a clear answer.

---

## The Problem

Deploying an AI model on edge hardware is fragmented and manual today.

A developer with a trained model faces:
- **Multiple runtimes** — ONNX Runtime, TensorRT, RKNN, llama.cpp, CoreML —
  each with different APIs, conversion tools, and quirks
- **Multiple devices** — MacBook, Raspberry Pi, Khadas, Jetson —
  each with different capabilities and SDK requirements
- **No standard benchmark** — every team re-invents the same test loop
- **No comparison baseline** — hard to know if a result is good or bad
  without running everything side by side

The result: developers spend days on deployment experiments before writing
a single line of application code.

Nuatara reduces that to one command.

---

## November MVP — Two Milestones

### Milestone 1: Vision Models (YOLOv8n)

**Goal:** A developer with a YOLOv8n model can run one command and get a
feasibility report across MacBook Air M2 and Khadas Edge 2 — with latency,
throughput, and a deployment recommendation. No manual runtime setup required.

This milestone validates:
- Backend abstraction layer (ONNX Runtime ↔ RKNN, same interface)
- Automatic runtime selection per device
- Scoring formula producing a meaningful recommendation
- nuatara recommend yolov8n

MacBook Air M2:
Runtime : ONNX Runtime (Apple Neural Engine)
Latency : 5.3ms 188 FPS
Score : 91

Khadas Edge 2:
Runtime : RKNN NPU (6 TOPS, RK3588)
Latency : 23.7ms 42 FPS
Score : 78

Recommendation:
Develop on MacBook → fast iteration, no quantization needed
Deploy on Khadas → 6× lower power, production edge hardware


### Milestone 2: Language Models (TinyLlama 1.1B)

**Goal:** A developer with a TinyLlama model can run the same command and get
a feasibility report across Mac and Khadas — proving Nuatara works beyond
vision models. This milestone validates that the abstraction layer is
model-type agnostic, not just vision-specific.

This milestone validates:
- LLM backend support (llama.cpp Metal + RKNN NPU)
- Token throughput as a benchmark metric alongside latency
- RAM feasibility check (flag models that won't fit on a device)
- nuatara recommend tinyllama

MacBook Air M2:
Runtime : llama.cpp (Metal)
Throughput : ~60 tokens/sec
Score : 88

Khadas Edge 2:
Runtime : RKNN NPU (6 TOPS, RK3588)
Throughput : ~25 tokens/sec (estimated)
Score : 79

Recommendation:
Develop on MacBook → fast iteration, Metal acceleration
Deploy on Khadas → NPU-accelerated inference, low power budget

> **Note:** Llama 3 8B requires ~16GB RAM at fp16 — Nuatara flags this as
> infeasible on Khadas Edge 2 (8GB RAM) before attempting to run it.

---

## Scoring Formula (v1)
Score = 0.4 × throughput_normalized
+ 0.3 × memory_efficiency
+ 0.2 × setup_complexity
+ 0.1 × power_estimate


Scores are relative within a comparison set — they rank options against
each other, not against an absolute standard.

---

## Device & Runtime Matrix

| Model        | MacBook Air M2          | Khadas Edge 2         |
|--------------|-------------------------|-----------------------|
| YOLOv8n      | ONNX Runtime (ANE)      | RKNN NPU              |
| TinyLlama    | llama.cpp (Metal)       | RKNN NPU              |
| Llama 3 8B   | llama.cpp (Metal)       | ✗ insufficient RAM    |

---

## What Nuatara Is Not

- A new inference runtime — it sits above existing runtimes
- A model training or fine-tuning platform
- A cloud deployment tool — edge hardware only for now
- A production serving layer — it advises, the developer deploys

---

## Phase 2 (Post-November)

- More devices: Raspberry Pi 5, Jetson Orin Nano
- More models: Whisper (speech), MobileNet (classification)
- Automatic format conversion (ONNX → RKNN) as part of the workflow
- Automated quantization advice (when INT8 helps, when it hurts)
