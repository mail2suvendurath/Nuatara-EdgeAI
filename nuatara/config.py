from dataclasses import dataclass, field

@dataclass
class DetectionConfig:
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    reg_max: int = 16
    input_size: int = 640
