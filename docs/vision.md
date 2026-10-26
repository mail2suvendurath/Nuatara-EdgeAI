# Nuatara-EdgeAI — Product Vision
*Last updated: October 2026*

---

## What It Is

Nuatara-EdgeAI Control Plane is a hardware-aware AI deployment advisor.

Given a model and available hardware, the platform:
- Discovers available AI hardware and runtimes
- Benchmarks every viable deployment alternative
- Evaluates performance, memory, power, and setup cost
- Recommends the optimal execution path

**The goal:** developers deploy AI workloads without becoming experts in every hardware vendor ecosystem.

---

## The Problem

Edge AI hardware is fragmented. Deploying a model today means:
- Manually testing ONNX Runtime vs TensorRT vs RKNN vs llama.cpp
- Re-doing this for every new device or model
- No standard way to compare results across hardware

Nuatara solves this with

### Milestone 2: Language Models (TinyLlama 1.1B)

**Goal:** Extend the platform beyond vision to language models.
Prove that the same hardware-agnostic abstraction works for LLM inference
via llama.cpp and RKNN, giving developers a single tool to evaluate edge LLM
deployment across Mac (Metal GPU) and Khadas (NPU), and understand the
trade-offs before committing to a runtime.

> **Note:** Llama 3 8B requires ~16GB RAM at fp16 — beyond Khadas Edge 2.
> TinyLlama 1.1B (~700MB at q4) is the realistic Khadas target for November.

---

## Scoring Formula (v1)

Scores are relative within a comparison set.
A score of 91 vs 79 means MacBook wins on this workload —
not that Khadas is a bad deployment target.

---

## Device & Runtime Matrix

| Model        | MacBook Air M2          | Khadas Edge 2         |
|--------------|-------------------------|-----------------------|
| YOLOv8n      | ONNX Runtime (ANE)      | RKNN NPU              |
| TinyLlama    | llama.cpp (Metal)       | RKNN NPU              |
| Llama 3 8B   | llama.cpp (Metal)       | ✗ insufficient RAM    |

---

## What We Are Not Building

- A new inference runtime (we sit above llama.cpp, ONNX Runtime, RKNN)
- A cloud deployment tool (edge-only for now)
- A model training or fine-tuning platform

---

## Phase 2 (Post-November)

- More devices: Raspberry Pi 5, Jetson Orin Nano
- More models: Whisper (speech), MobileNet (classification)
- RKNN runtime for LLMs (when Rockchip SDK matures)
- Automated quantization advice (when to go INT8, when not to)
