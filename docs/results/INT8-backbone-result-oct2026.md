# INT8 Backbone Benchmark Results — October 2026

## YOLOv8n Results

| Device | Backend | FPS | Notes |
|--------|---------|-----|-------|
| Mac M2 | ONNX ANE | 184 | FP32, full accuracy |
| Khadas Edge 2 | RKNN INT8 | 48 | scores capped ~0.50 |
| Khadas Edge 2 | ONNX CPU | 7.7 | no NPU, baseline only |

NPU gives 6x speedup over CPU on Khadas.

## Lessons Learnt: RKNN Postprocessing

### ONNX vs RKNN export difference
- ONNX: single fused tensor (1,84,8400) — boxes already decoded inside model
- RKNN: 9 raw tensors — 3 scales x (DFL bins, class scores, objectness) — decode in postprocessor

### Surprise 1: Sigmoid already applied
- RKNN toolkit bakes sigmoid into export
- Class scores already in [0,1] range
- Applying sigmoid again squashes scores to ~0.5 — no detections
- Fix: use scores directly, do not apply sigmoid

### Surprise 2: Objectness tensors are useless
- YOLOv8 has no real objectness head
- RKNN exports dummy (1,1,H,W) tensors — constant near 0.003
- Multiplying by near-zero objectness kills all detections
- Fix: ignore objectness, use class scores as confidence directly

### Surprise 3: DFL ltrb values are in pixel units
- After softmax over 16 bins, ltrb is already in pixels
- RKNN bakes stride into bin scale at export
- Multiplying by stride again gives coordinates 8-32x too large
- Fix: do not multiply ltrb by stride

### Working decode formula
stride = 640 // H
ltrb = softmax(box_bins).dot(bins) # pixel units — NO stride multiply
cx = (grid_x + 0.5) * stride
cy = (grid_y + 0.5) * stride
x1, y1 = cx - ltrb[0], cy - ltrb[1]
x2, y2 = cx + ltrb[2], cy + ltrb[3]

### General Rule
Always probe raw tensor stats before writing a postprocessor for a new backend:
- Check value ranges to see if activations are pre-applied
- Check for constant tensors indicating dummy heads
- Decode one cell manually and verify against a known image
