
## Why Khadas RKNN NPU Cannot Run TinyLlama

The RK3588 NPU is a **fixed-function accelerator** built for CNN workloads. Here is why it fails for LLMs:

| Requirement | RKNN NPU | TinyLlama (Transformer) |
|-------------|----------|------------------------|
| Input shape | Fixed at export time | Dynamic — grows each token |
| Operation type | Conv2D, DepthwiseConv | Multi-head attention, FFN |
| Memory access | Sequential, predictable | KV cache — random access, grows |
| Supported by RKNN toolkit | ✅ | ❌ |

**In simple terms:**
- RKNN NPU needs to know the exact tensor shapes before it runs — you bake them in at model export
- TinyLlama generates one token at a time, and the attention window grows with each token — the shape changes every step
- RKNN toolkit cannot export a model with dynamic shapes → LLM cannot run on NPU

**What the RK3588 NPU IS good at:**
- YOLOv8, MobileNet, ResNet — fixed 640×640 or 224×224 input, static shapes throughout
- Image classification, object detection, segmentation

**What needs the CPU instead:**
- Any transformer model (BERT, GPT, LLaMA, Whisper)
- Any model with dynamic sequence length

This is a Nuatara insight: same chip, two very different capabilities depending on model type.
