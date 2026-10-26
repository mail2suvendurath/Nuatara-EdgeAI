import platform
from pathlib import Path


def discover_devices():
    devices = []

    if platform.system() == "Darwin":
        devices.append({
            "name": "MacBook (Apple Neural Engine)",
            "backend": "onnx",
            "config": "configs/laptop.yaml",
        })

    try:
        import rknnlite
        devices.append({
            "name": "Khadas Edge 2 (RKNN NPU)",
            "backend": "rknpu",
            "config": "configs/khadas.yaml",
        })
    except ImportError:
        pass

    return devices
