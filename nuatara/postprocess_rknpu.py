import numpy as np
from nuatara.types import Detection
from nuatara.config import DetectionConfig

COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
]

def softmax(x):
    e = np.exp(x - x.max(axis=0, keepdims=True))
    return e / e.sum(axis=0, keepdims=True)

def decode_boxes(box_raw, H, W, reg_max=16):
    stride = 640 // H
    box = box_raw[0].reshape(4, reg_max, H * W)
    box = softmax(box)
    bins = np.arange(reg_max, dtype=np.float32)
    ltrb = (box * bins[:, np.newaxis]).sum(axis=1)  # pixel units, no stride multiply
    gy, gx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    cx = (gx.flatten() + 0.5) * stride
    cy = (gy.flatten() + 0.5) * stride
    x1 = cx - ltrb[0]
    y1 = cy - ltrb[1]
    x2 = cx + ltrb[2]
    y2 = cy + ltrb[3]
    return np.stack([x1, y1, x2, y2], axis=1)

def postprocess_rknpu(outputs, cfg):
    scales = [(0, 1, 80, 80), (3, 4, 40, 40), (6, 7, 20, 20)]
    all_boxes, all_scores, all_class_ids = [], [], []
    for bi, ci, H, W in scales:
        cls_scores = outputs[ci][0].reshape(80, H * W)
        max_scores = cls_scores.max(axis=0)
        class_ids  = cls_scores.argmax(axis=0)
        mask = max_scores > cfg.conf_threshold
        if mask.sum() == 0:
            continue
        boxes = decode_boxes(outputs[bi], H, W, cfg.reg_max)
        all_boxes.append(boxes[mask])
        all_scores.append(max_scores[mask])
        all_class_ids.append(class_ids[mask])
    if not all_boxes:
        return []
    boxes  = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    cids   = np.concatenate(all_class_ids)
    keep = nms(boxes, scores, cfg.iou_threshold)
    return [Detection(
        class_id=int(cids[i]),
        class_name=COCO_CLASSES[int(cids[i])] if int(cids[i]) < len(COCO_CLASSES) else str(cids[i]),
        confidence=float(scores[i]),
        x1=float(boxes[i,0]), y1=float(boxes[i,1]),
        x2=float(boxes[i,2]), y2=float(boxes[i,3]),
    ) for i in keep]

def nms(boxes, scores, iou_thresh):
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas = (x2-x1)*(y2-y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]; keep.append(i)
        xx1=np.maximum(x1[i],x1[order[1:]])
        yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]])
        yy2=np.minimum(y2[i],y2[order[1:]])
        inter=np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
        iou=inter/(areas[i]+areas[order[1:]]-inter)
        order=order[1:][iou<=iou_thresh]
    return keep
