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

The goal: developers deploy AI workloads without becoming experts in every hardware vendor ecosystem.

---

## November MVP — Three Milestones

### Milestone 1: Vision Models (YOLOv8n) ✅ Complete

**Command:**
```
nuatara recommend yolov8n
```

**Actual output:**
```
MacBook Air M2:
  Runtime : ONNX Runtime (Apple Neural Engine)
  Latency : 5.4ms   184 FPS
  Score   : 93

Khadas Edge 2:
  Runtime : RKNN INT8 NPU
  Latency : 20.8ms   48 FPS
  Score   : 78

Khadas Edge 2 (baseline):
  Runtime : ONNX CPU
  Latency : 130ms   7.7 FPS
  Score   : 41

Recommendation: MacBook Air M2 → Score 93
```

**Key finding:** INT8 quantization on RKNN compresses confidence scores (max ~0.50 vs ~0.90 on FP32). Developers must lower detection thresholds or switch to FP16 if accuracy matters more than speed. NPU gives 6× speedup over CPU on Khadas.

---

### Milestone 2: Language Models (TinyLlama 1.1B) ✅ Complete

**Command:**
```
nuatara recommend tinyllama
```

**Actual output:**
```
MacBook Air M2:
  Runtime    : llama.cpp (Metal GPU)
  Throughput : 63 tokens/sec
  Score      : 55

Khadas Edge 2:
  Runtime    : llama.cpp (CPU, 4-thread)
  Throughput : 26 tokens/sec
  Score      : 55

Recommendation: MacBook Air M2 → 2.4× faster
```

**Key finding:** RKNN NPU cannot run LLMs — it requires static input shapes, but transformers have dynamic attention (KV cache grows per token). Khadas runs LLMs on CPU only. 26 tokens/sec is viable for edge chatbot and on-device summarization. Live chat tested interactively on both devices.

---

### Milestone 3: Quantization-Aware Recommendation

**The problem discovered in Milestone 1:**
Running the same YOLOv8n on Mac (FP32) vs Khadas (INT8) produces visibly different detection quality — Mac draws more accurate boxes, Khadas misses some detections. A developer deploying to Khadas needs to know this *before* they ship.

**Command:**
```
nuatara recommend yolov8n --target khadas --quality-report
```

**Expected output:**
```
Khadas Edge 2 — Quantization Trade-off Report:

  INT8  (current):  48 FPS   conf scores capped ~0.50   some missed detections
  FP16  (option):  ~25 FPS   conf scores up to ~0.80    near-FP32 accuracy
  FP32  (option):  ~12 FPS   full accuracy              defeats edge purpose

Recommendation:
  Use INT8 if throughput > accuracy  (surveillance, counting)
  Use FP16 if accuracy > throughput  (safety-critical, medical)
  Lower conf_threshold to 0.10–0.15 to recover missed detections in INT8
```

**What Nuatara will do automatically:**
1. Detect score compression by comparing max scores across backends
2. Suggest adjusted `conf_threshold` per backend
3. Show mAP alongside FPS so developer sees the accuracy cost
4. Compare `yolov8n INT8` vs `yolov8s FP16` — which hits the sweet spot

---

## Scoring (v1)

```
Score = 0.4 × throughput_normalized
      + 0.3 × memory_efficiency
      + 0.2 × setup_complexity
      + 0.1 × power_estimate
```

Scores are relative within a comparison set, not absolute.
Scores across different model types (YOLO vs TinyLlama) are not comparable.

---

## Device & Runtime Matrix

| Model | MacBook Air M2 | Khadas Edge 2 |
|-------|---------------|---------------|
| YOLOv8n | ONNX Runtime (ANE) — 184 FPS | RKNN INT8 NPU — 48 FPS |
| TinyLlama 1.1B | llama.cpp (Metal) — 63 tok/sec | llama.cpp (CPU) — 26 tok/sec |
| Llama 3 8B | llama.cpp (Metal) | ✗ insufficient RAM |

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
- Full quantization pipeline: FP32 → FP16 → INT8 with automated accuracy benchmarks
