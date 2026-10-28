# Nuatara-EdgeAI — Product Vision
*Last updated: October 2026*

---

## What It Is

Nuatara-EdgeAI is a hardware-agnostic toolkit for Edge AI deployment.

Given a model and available hardware, the platform:
- Discovers available AI hardware and runtimes automatically
- Benchmarks every viable deployment alternative on real inference
- Evaluates performance, memory, power, and setup cost
- Recommends the optimal execution path
- Detects quantization quality loss and advises threshold adjustments

The goal: developers deploy AI workloads without becoming experts in every hardware vendor ecosystem.

---

## POC — Three Milestones ✅ Complete (October 2026)

### Milestone 1: Vision Models (YOLOv8n) ✅

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
  Latency : 20.7ms   48 FPS
  Score   : 87

Khadas Edge 2 (baseline):
  Runtime : ONNX CPU
  Latency : 130ms   7.7 FPS
  Score   : 41

Recommendation: MacBook Air M2 → Score 93
```

**Key finding:** Same model, same CLI, two completely different hardware stacks. RKNN NPU gives 6× speedup over CPU on Khadas. INT8 quantization compresses confidence scores (max ~0.50 vs ~0.89 on FP32) — Milestone 3 detects and reports this automatically.

---

### Milestone 2: Language Models (TinyLlama 1.1B) ✅

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

**Key finding:** RKNN NPU cannot run LLMs — it requires static input shapes, but transformers have dynamic attention (KV cache grows per token). Khadas runs LLMs on CPU only. 26 tokens/sec is viable for edge chatbot and on-device summarization. Tested interactively on both devices with zero code changes between them.

---

### Milestone 3: Quantization Quality Report ✅

**Command:**
```
nuatara recommend yolov8n --quality-report
```

**Actual output:**
```
Backend      Max Conf     Status               Suggested threshold
-------      --------     ------               -------------------
onnx         0.89         ✓  Full range        0.25
rknpu        0.50         ⚠  INT8 compression  0.12

FP32 vs INT8 — measured on real test image:

  Backend               FPS     Max conf   Precision
  -------               ---     ---------  ---------
  Mac  ONNX (FP32)      184      0.89      full range — reference
  Khadas RKNN (INT8)     48      0.50      ⚠ compressed by INT8

Recommendation:
  Lower conf_threshold to 0.12 on Khadas to recover missed detections
  Use INT8 when throughput matters  (surveillance, counting)
  Re-export to FP16 when accuracy matters  (safety-critical, medical)
```

**Key finding:** All numbers come from live inference on a real test image — nothing hardcoded. Mac ONNX (FP32) is used as the reference baseline. The toolkit detects INT8 score compression automatically and tells the developer exactly what to adjust.

---

## What the POC Proved

1. **Hardware abstraction works** — one CLI, one model name, runs on Apple Silicon and RK3588 NPU without developer changes
2. **Runtime detection is automatic** — `nuatara recommend` discovers ONNX Runtime on Mac and RKNNLite on Khadas with no configuration
3. **Quantization effects are measurable and reportable** — INT8 confidence compression is caught automatically from real inference, not guesswork
4. **LLM + Vision in one toolkit** — same framework handles both model families across both devices

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
| YOLOv8n | ONNX Runtime (ANE) — 184 FPS, conf 0.89 | RKNN INT8 NPU — 48 FPS, conf 0.50 |
| TinyLlama 1.1B | llama.cpp (Metal) — 63 tok/sec | llama.cpp (CPU) — 26 tok/sec |
| Llama 3 8B | llama.cpp (Metal) | ✗ insufficient RAM |

---

## What We Are Not Building

- A new inference runtime (we sit above llama.cpp, ONNX Runtime, RKNN)
- A cloud deployment tool (edge-only for now)
- A model training or fine-tuning platform

---

## Phase 2 (Post-POC)

- More devices: Raspberry Pi 5, Jetson Orin Nano
- More models: Whisper (speech), MobileNet (classification)
- FP16 RKNN export and benchmark (vs INT8 accuracy trade-off)
- RKNN runtime for LLMs (when Rockchip SDK matures)
- Full quantization pipeline: FP32 → FP16 → INT8 with automated accuracy benchmarks
