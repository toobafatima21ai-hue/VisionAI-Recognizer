"""
================================================================================
 VisionAI Recognizer | utils/object_detector.py
 Module: Image Recognition / Object Detection Engine
--------------------------------------------------------------------------------
 Wraps a pre-trained YOLOv8 model (via the `ultralytics` package) for
 general-purpose object detection. Provides a simple, decoupled API:
 load the model once, run detection on any frame, get back plain
 Python data structures, and draw results on demand.

 Why YOLOv8:
 - Single-stage detector → very fast (real-time capable on CPU for
   small images).
 - Pre-trained on COCO (80 common object classes: person, car, dog,
   laptop, bottle, etc.) — no training required for a general demo.
 - `ultralytics` package handles model download, inference, and
   post-processing (NMS) internally — minimal boilerplate.
================================================================================
"""

from __future__ import annotations

import time
import numpy as np
import cv2

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:  # pragma: no cover
    YOLO_AVAILABLE = False


# A small, curated color palette so boxes are visually distinct
# and reproducible across runs (cycled by class index).
_PALETTE = [
    (255, 99, 71), (60, 179, 113), (65, 105, 225), (255, 215, 0),
    (218, 112, 214), (0, 191, 255), (255, 140, 0), (50, 205, 50),
    (199, 21, 133), (0, 206, 209),
]


# ──────────────────────────────────────────────────────────────────────────
# MODEL FACTORY
# ──────────────────────────────────────────────────────────────────────────
def load_detector(model_name: str = "yolov8n.pt"):
    """
    Load a pre-trained YOLOv8 model.

    The first call downloads the weights file (a few MB for the 'n'/nano
    variant) to the local `models/` cache; subsequent calls reuse it.
    The caller (app.py) should wrap this with st.cache_resource so the
    model is loaded only once per session.

    Args:
        model_name: Ultralytics model identifier. 'yolov8n.pt' (nano) is
                     the fastest / smallest — ideal for CPU demos.
                     Other options: 'yolov8s.pt', 'yolov8m.pt' (larger,
                     more accurate, slower).

    Returns:
        A YOLO model instance ready for inference.

    Raises:
        RuntimeError if the `ultralytics` package is not installed.
    """
    if not YOLO_AVAILABLE:
        raise RuntimeError(
            "ultralytics (YOLOv8) is not installed. Run: pip install ultralytics"
        )

    return YOLO(model_name)


# ──────────────────────────────────────────────────────────────────────────
# CORE DETECTION
# ──────────────────────────────────────────────────────────────────────────
def detect_objects(
    model,
    image: np.ndarray,
    confidence_threshold: float = 0.25,
) -> dict:
    """
    Run object detection on an image and return structured results.

    Args:
        model: A loaded YOLO model instance (from load_detector).
        image: Input image as a numpy array (BGR, as from OpenCV).
        confidence_threshold: Minimum confidence (0-1) for a detection
                               to be kept.

    Returns:
        dict with keys:
            'detections'     : list of dicts, each with
                                {label, confidence, bbox} where bbox is
                                [x1, y1, x2, y2] in pixel coordinates.
            'count'          : total number of objects detected.
            'class_counts'   : dict mapping class label -> count.
            'processing_time': seconds taken for inference.
    """
    if image is None:
        raise ValueError("detect_objects: input image is None")

    # YOLOv8 requires a 3-channel (color) image. If the preprocessing
    # pipeline produced a single-channel grayscale/binary image (e.g.
    # 'grayscale' or 'threshold' steps were applied), convert it back
    # to 3-channel BGR so the model's first conv layer (expects 3
    # input channels) doesn't raise a shape-mismatch error.
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    start = time.time()
    results = model.predict(
        source=image,
        conf=confidence_threshold,
        verbose=False,
    )
    elapsed = time.time() - start

    detections = []
    class_counts: dict[str, int] = {}

    result = results[0]
    names = result.names  # class-index -> class-name mapping

    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = names.get(cls_id, str(cls_id))
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

        detections.append({
            "label": label,
            "confidence": round(confidence, 4),
            "bbox": [x1, y1, x2, y2],
            "class_id": cls_id,
        })
        class_counts[label] = class_counts.get(label, 0) + 1

    return {
        "detections": detections,
        "count": len(detections),
        "class_counts": class_counts,
        "processing_time": round(elapsed, 3),
    }


# ──────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────
def draw_object_boxes(
    image: np.ndarray,
    detections: list[dict],
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw bounding boxes, class labels, and confidence scores for each
    detected object, using a consistent color per class.

    Args:
        image: Original image (BGR) to draw on. A copy is made internally.
        detections: List of detection dicts as returned by detect_objects().
        thickness: Box line thickness in pixels.

    Returns:
        A new image (copy) with boxes and labels drawn.
    """
    if image is None:
        raise ValueError("draw_object_boxes: input image is None")

    # Ensure 3-channel output so colored boxes/labels are visible even
    # if a grayscale/binary preprocessed image was passed in.
    if len(image.shape) == 2:
        output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        output = image.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = _PALETTE[det.get("class_id", 0) % len(_PALETTE)]

        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

        label = f"{det['label']} {det['confidence'] * 100:.0f}%"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

        # Label background for readability
        cv2.rectangle(
            output,
            (x1, max(0, y1 - text_h - 10)),
            (x1 + text_w + 6, y1),
            color,
            -1,
        )
        cv2.putText(
            output, label, (x1 + 3, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return output
