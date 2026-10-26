"""
engine/postprocess_rknpu.py — Postprocessor for RKNPU backend output.

Input:  list of 9 tensors from RKNNBackend.infer()
        3 FPN scales × 3 tensors each:
          [0]  (1, 4*reg_max, 80, 80)   box DFL bins, large objects   (stride 8)
          [1]  (1, 80,        80, 80)   class scores, sigmoid already applied by runtime
          [2]  (1, 1,         80, 80)   objectness, sigmoid already applied by runtime
          [3]  (1, 4*reg_max, 40, 40)   box DFL bins, medium          (stride 16)
          [4]  (1, 80,        40, 40)   class scores
          [5]  (1, 1,         40, 40)   objectness
          [6]  (1, 4*reg_max, 20, 20)   box DFL bins, small objects   (stride 32)
          [7]  (1, 80,        20, 20)   class scores
          [8]  (1, 1,         20, 20)   objectness

        IMPORTANT: The RKNN runtime applies sigmoid to class scores and
        objectness internally. Do NOT apply sigmoid again here.

Output: list[Detection]

NOTE (future Rust migration):
    fn postprocess_rknpu(outputs: &[ArrayD<f32>], cfg: &DetectionConfig, reg_max: usize)
        -> Vec<Detection>
    DFL softmax stays the same; NMS → rayon parallel by class.
"""

from __future__ import annotations

import numpy as np
from nuatara.types import Detection
from nuatara.config import DetectionConfig
from nuatara.postprocess_onnx import COCO_NAMES, _nms   # reuse NMS + class names


# Strides for each of the 3 FPN levels (must match export settings)
_STRIDES = [8, 16, 32]


def postprocess_rknpu(
    outputs: list[np.ndarray],
    cfg: DetectionConfig,
    img_w: int = 640,
    img_h: int = 640,
) -> list[Detection]:
    """
    Decode 9-tensor RKNPU output into Detection objects.

    Args:
        outputs  : list[9 tensors] from RKNNBackend.infer()
        cfg      : DetectionConfig (conf_threshold, nms_threshold, reg_max, ...)
        img_w/h  : original image dimensions for box scaling

    Returns:
        list[Detection] sorted by confidence descending, after NMS
    """
    if len(outputs) != 9:
        raise ValueError(f"Expected 9 output tensors, got {len(outputs)}")

    reg_max = cfg.reg_max       # 16 for YOLOv8
    all_boxes:   list[np.ndarray] = []
    all_scores:  list[np.ndarray] = []
    all_cls_ids: list[np.ndarray] = []

    for level in range(3):
        dfl_raw  = outputs[level * 3 + 0]   # (1, 4*reg_max, H, W)
        cls_raw  = outputs[level * 3 + 1]   # (1, 80, H, W) — already sigmoid'd
        obj_raw  = outputs[level * 3 + 2]   # (1, 1,  H, W) — already sigmoid'd

        stride = _STRIDES[level]
        _, _, H, W = dfl_raw.shape

        # ── DFL decode ───────────────────────────────────────────────────
        # reshape to (4, reg_max, H*W), softmax over reg_max dim, then expectation
        dfl = dfl_raw[0].reshape(4, reg_max, H * W)         # (4, reg_max, N)
        dfl = dfl - dfl.max(axis=1, keepdims=True)           # numerical stability
        dfl = np.exp(dfl)
        dfl = dfl / dfl.sum(axis=1, keepdims=True)           # softmax → (4, reg_max, N)
        bins = np.arange(reg_max, dtype=np.float32)
        ltrb = (dfl * bins[None, :, None]).sum(axis=1)       # (4, N) = l,t,r,b

        # ── grid anchors ─────────────────────────────────────────────────
        gy, gx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        cx = (gx.ravel() + 0.5) * stride    # (N,)
        cy = (gy.ravel() + 0.5) * stride    # (N,)

        # ltrb → xyxy in model input space (640×640)
        x1 = cx - ltrb[0]
        y1 = cy - ltrb[1]
        x2 = cx + ltrb[2]
        y2 = cy + ltrb[3]

        # ── scores (sigmoid already applied by RKNN runtime) ─────────────
        cls_scores = cls_raw[0].reshape(80, H * W)   # (80, N)
        obj_scores = obj_raw[0].reshape(1,  H * W)   # (1, N)

        # YOLOv8 uses cls * obj as final confidence
        combined = cls_scores * obj_scores            # (80, N)

        cls_ids = np.argmax(combined, axis=0)         # (N,)
        confs   = combined[cls_ids, np.arange(H * W)] # (N,)

        # pre-filter
        mask = confs >= cfg.conf_threshold
        if not np.any(mask):
            continue

        boxes = np.stack([x1, y1, x2, y2], axis=1)[mask]   # (K, 4)
        all_boxes.append(boxes)
        all_scores.append(confs[mask])
        all_cls_ids.append(cls_ids[mask])

    if not all_boxes:
        return []

    boxes_all   = np.concatenate(all_boxes,   axis=0)
    scores_all  = np.concatenate(all_scores,  axis=0)
    cls_ids_all = np.concatenate(all_cls_ids, axis=0)

    # scale to original image
    scale_x = img_w / 640.0
    scale_y = img_h / 640.0
    boxes_all[:, [0, 2]] *= scale_x
    boxes_all[:, [1, 3]] *= scale_y

    # NMS per class
    detections: list[Detection] = []
    for cls in np.unique(cls_ids_all):
        m = cls_ids_all == cls
        kept = _nms(boxes_all[m], scores_all[m], cfg.nms_threshold)
        cb = boxes_all[m]
        cs = scores_all[m]
        for idx in kept:
            detections.append(Detection(
                x1=float(cb[idx, 0]),
                y1=float(cb[idx, 1]),
                x2=float(cb[idx, 2]),
                y2=float(cb[idx, 3]),
                confidence=float(cs[idx]),
                class_id=int(cls),
                class_name=COCO_NAMES[int(cls)] if int(cls) < len(COCO_NAMES) else str(cls),
            ))

    detections.sort(key=lambda d: d.confidence, reverse=True)
    if cfg.max_detections > 0:
        detections = detections[:cfg.max_detections]
    return detections
