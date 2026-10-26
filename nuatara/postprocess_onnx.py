"""
engine/postprocess_onnx.py — Postprocessor for ONNX backend output.

Input:  list of 1 tensor: (1, 84, 8400) float32
        ├── rows 0–3   : box coords in DFL-decoded cx,cy,w,h (already decoded by export)
        └── rows 4–83  : 80 COCO class scores (raw logits, NOT sigmoid'd)

        NOTE: YOLOv8 ONNX exports from Ultralytics already decode the DFL
        regression internally — output is cx,cy,w,h directly, not DFL bins.
        The 80 class scores ARE raw logits; apply sigmoid before thresholding.

Output: list[Detection]

NOTE (future Rust migration):
    fn postprocess_onnx(output: &ArrayView2<f32>, cfg: &DetectionConfig)
        -> Vec<Detection>
    Box decoding stays identical; NMS becomes rayon parallel.
"""

from __future__ import annotations

import numpy as np
from engine.types import Detection
from engine.config import DetectionConfig

# COCO class names (80 classes, index matches row 4+i in the output tensor)
COCO_NAMES: list[str] = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush",
]


def postprocess_onnx(
    outputs: list[np.ndarray],
    cfg: DetectionConfig,
    img_w: int = 640,
    img_h: int = 640,
) -> list[Detection]:
    """
    Decode YOLOv8 ONNX output into Detection objects.

    Args:
        outputs  : list of 1 tensor from OnnxRuntimeBackend.infer()
        cfg      : DetectionConfig (conf_threshold, nms_threshold, max_detections)
        img_w/h  : original image size for scaling boxes (default 640×640 = no scale)

    Returns:
        list[Detection] sorted by confidence descending, after NMS
    """
    if not outputs or outputs[0].shape != (1, 84, 8400):
        raise ValueError(
            f"Expected output shape (1, 84, 8400), got "
            f"{[o.shape for o in outputs]}"
        )

    pred = outputs[0][0]          # (84, 8400)  — drop batch dim
    boxes_raw = pred[:4, :]       # (4, 8400)   — cx, cy, w, h  (model input space)
    scores_raw = pred[4:, :]      # (80, 8400)  — raw logits

    # NOTE: Ultralytics YOLOv8 ONNX export (default) bakes sigmoid into the model.
    # Do NOT apply sigmoid again. If you exported with --no-sigmoid, flip this.
    # sigmoid → probabilities
    scores = scores_raw   # sigmoid already applied by Ultralytics export

    class_ids = np.argmax(scores, axis=0)          # (8400,)
    confidences = scores[class_ids, np.arange(8400)]  # (8400,)

    # pre-filter by confidence before NMS (fast rejection)
    mask = confidences >= cfg.conf_threshold
    if not np.any(mask):
        return []

    boxes_raw = boxes_raw[:, mask]      # (4, N)
    confidences = confidences[mask]     # (N,)
    class_ids = class_ids[mask]         # (N,)

    # cx,cy,w,h → x1,y1,x2,y2  (still in 640×640 model space)
    cx, cy, w, h = boxes_raw
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # scale to original image dimensions
    scale_x = img_w / 640.0
    scale_y = img_h / 640.0
    x1 *= scale_x; x2 *= scale_x
    y1 *= scale_y; y2 *= scale_y

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)  # (N, 4)

    # NMS per class
    detections: list[Detection] = []
    for cls in np.unique(class_ids):
        cls_mask = class_ids == cls
        cls_boxes = boxes_xyxy[cls_mask]
        cls_scores = confidences[cls_mask]
        kept = _nms(cls_boxes, cls_scores, cfg.nms_threshold)
        for idx in kept:
            detections.append(Detection(
                x1=float(cls_boxes[idx, 0]),
                y1=float(cls_boxes[idx, 1]),
                x2=float(cls_boxes[idx, 2]),
                y2=float(cls_boxes[idx, 3]),
                confidence=float(cls_scores[idx]),
                class_id=int(cls),
                class_name=COCO_NAMES[int(cls)] if int(cls) < len(COCO_NAMES) else str(cls),
            ))

    detections.sort(key=lambda d: d.confidence, reverse=True)
    if cfg.max_detections > 0:
        detections = detections[:cfg.max_detections]
    return detections


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Pure-numpy Non-Maximum Suppression. Returns kept indices."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    kept = []
    while order.size > 0:
        i = order[0]
        kept.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        order = order[1:][iou <= iou_threshold]
    return kept
