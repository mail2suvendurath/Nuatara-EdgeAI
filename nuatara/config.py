from dataclasses import dataclass

@dataclass
class DetectionConfig:
    conf_threshold: float = 0.25
    nms_threshold: float = 0.45
    max_detections: int = 100
