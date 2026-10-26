"""
nuatara demo — run detection on a video with live preview or headless.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from nuatara.backends._factory import create_backend
from nuatara.postprocess_onnx import postprocess_onnx
from nuatara.config import DetectionConfig

MODEL_REGISTRY = {
    "yolov8n": {
        "onnx":  "models/yolov8n.onnx",
        "rknpu": "models/yolov8n.rknn",
        "input_size": 640,
    },
}

COLORS = {}

def get_color(class_name):
    if class_name not in COLORS:
        np.random.seed(hash(class_name) % 2**32)
        COLORS[class_name] = tuple(int(x) for x in np.random.randint(50, 255, 3))
    return COLORS[class_name]


def preprocess(frame, size, backend_name):
    resized = cv2.resize(frame, (size, size))
    if backend_name == "rknpu":
        # RKNN expects NHWC uint8
        tensor = resized[:, :, ::-1].astype(np.uint8)
        tensor = np.expand_dims(tensor, axis=0)
    else:
        # ONNX expects NCHW float32
        tensor = resized[:, :, ::-1].astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)[np.newaxis]
    return tensor


def run_demo(model_name, video_path, backend_name="onnx", output_path=None):
    meta = MODEL_REGISTRY.get(model_name)
    if not meta:
        print(f"Unknown model: {model_name}")
        return

    model_path = meta.get(backend_name)
    if not model_path or not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        return

    print(f"\nLoading {model_name} on {backend_name}...")
    backend = create_backend(backend_name)
    precision = "int8" if backend_name == "rknpu" else "fp32"
    backend.load(model_path, precision=precision)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_path is None:
        stem = Path(video_path).stem
        output_path = f"{stem}_detected_{backend_name}.mp4"

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
    cfg = DetectionConfig(conf_threshold=0.3)
    size = meta["input_size"]

    frame_count = 0
    fps_list = []

    headless = False
    try:
        cv2.namedWindow("Nuatara Demo", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Nuatara Demo", 1280, 720)
    except Exception:
        headless = True
        print("No display found — running headless, saving output only.")

    print(f"Processing {total_frames} frames — press Q to quit early")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tensor = preprocess(frame, size, backend_name)

        t0 = time.perf_counter()
        outputs = backend.infer(tensor)
        fps = 1.0 / (time.perf_counter() - t0)
        fps_list.append(fps)

        if backend_name == "onnx":
            detections = postprocess_onnx(outputs, cfg)
        else:
            from nuatara.postprocess_rknpu import postprocess_rknpu
            detections = postprocess_rknpu(outputs, cfg)

        sx, sy = w / size, h / size
        for d in detections:
            x1, y1 = int(d.x1 * sx), int(d.y1 * sy)
            x2, y2 = int(d.x2 * sx), int(d.y2 * sy)
            color = get_color(d.class_name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{d.class_name} {d.confidence:.2f}"
            cv2.putText(frame, label, (x1, max(y1-6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        avg_fps = sum(fps_list[-30:]) / len(fps_list[-30:])
        cv2.putText(frame, f"Nuatara | {backend_name.upper()} | {avg_fps:.1f} FPS",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        if not headless:
            cv2.imshow("Nuatara Demo", frame)
        out.write(frame)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"  {frame_count}/{total_frames} frames — {avg_fps:.1f} FPS")

        if not headless and cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopped by user.")
            break

    cap.release()
    out.release()
    if not headless:
        cv2.destroyAllWindows()

    avg = sum(fps_list) / len(fps_list) if fps_list else 0
    print(f"\nDone. {frame_count} frames, avg {avg:.1f} FPS")
    print(f"Saved: {output_path}")
