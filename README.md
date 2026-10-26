# Nuatara-EdgeAI
Hardware-aware AI deployment advisor that discovers hardware, benchmarks runtimes, and recommends the optimal execution path for AI models.
**Nuatara-EdgeAI Control Plane**
**Vision**
Nuatara-EdgeAI Control Plane is a hardware-aware AI deployment advisor.
Given a model and available hardware, the platform:
- Discovers available AI hardware
- Discovers available runtimes
- Benchmarks deployment alternatives
- Evaluates performance
- Recommends the optimal execution path

The goal is to allow developers to deploy AI workloads without becoming experts in every hardware vendor ecosystem.

**Initial Scope**
Devices:
  - MacBook
  - Khadas Edge 2
Models:
  - TinyLlama
  - Llama 3 8B
Runtimes:
  - llama.cpp
  - RKNN (future)
**November MVP**
**Command:**
  Nuatara-EdgeAI recommend llama
**Output:**
  MacBook:
  - Runtime: llama.cpp
  - Score: 91
  Khadas Edge 2:
  - Runtime: RKNN
  - Score: 78
Recommendation:
  Develop on MacBook
  Deploy on Khadas
