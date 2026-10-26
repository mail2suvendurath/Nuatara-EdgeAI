# Nuatara-EdgeAI — Product Vision
*Last updated: October 2026*

## What It Is

Nuatara-EdgeAI is a hardware-aware AI deployment advisor.

Given a model and available hardware, the platform:
- Discovers available AI hardware and runtimes
- Benchmarks every viable deployment alternative
- Evaluates performance, memory, power, and setup cost
- Recommends the optimal execution path

The goal: developers deploy AI workloads without becoming experts in every hardware vendor ecosystem.

## November MVP — Three Milestones

### Milestone 1: Vision Models (YOLOv8n) ✅ Complete
- Mac M2: 184 FPS via ONNX/ANE, Score 93
- Khadas Edge 2: 48 FPS via RKNN INT8, Score 78
- Key finding: INT8 compresses scores to ~0.50 max — threshold tuning needed

### Milestone 2: Language Models (TinyLlama 1.1B)
- Mac M2: ~60 tokens/sec via llama.cpp Metal
- Khadas Edge 2: ~8 tokens/sec via llama.cpp CPU

### Milestone 3: Quantization-Aware Recommendation
- Auto-detect score compression across backends
- Show INT8 vs FP16 trade-off (speed vs accuracy)
- Suggest adjusted conf_threshold per backend

## Scoring (v1)
Score = 0.4 x throughput + 0.3 x memory_efficiency + 0.2 x setup_complexity + 0.1 x power

## What We Are Not Building
- A new inference runtime
- A cloud deployment tool
- A model training platform
