from dataclasses import dataclass

@dataclass
class DetectionConfig:
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    nms_threshold: float = 0.45
    max_detections: int = 100
    reg_max: int = 16
    input_size: int = 640
