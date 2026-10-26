"""
Image preprocessing for YOLO inference.
Resize to 640x640, normalize to [0,1], NCHW format.
"""

import numpy as np
from PIL import Image


def load_image(path: str, size: int = 640) -> tuple[np.ndarray, tuple]:
    """
    Load and preprocess image for YOLOv8 inference.
    Returns (tensor NCHW float32, original_size (w, h))
    """
    img = Image.open(path).convert("RGB")
    orig_size = img.size  # (w, h)
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)          # HWC -> CHW
    arr = np.expand_dims(arr, axis=0)     # CHW -> NCHW
    return arr, orig_size
