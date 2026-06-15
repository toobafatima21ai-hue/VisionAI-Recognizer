"""
================================================================================
 VisionAI Recognizer | utils/ocr_engine.py
 Module: Text Recognition (OCR) Engine
--------------------------------------------------------------------------------
 Wraps EasyOCR (primary) with an optional Tesseract fallback. Provides
 text extraction with bounding boxes and per-detection confidence scores,
 plus helper functions to draw boxes on images and summarize results.

 Design notes:
 - The EasyOCR Reader is expensive to initialize (loads neural network
   weights), so it is created ONCE and cached via @st.cache_resource
   at the app layer. This module itself just exposes a factory function.
 - All public functions return plain Python data structures (lists/dicts)
   so the Streamlit UI layer stays decoupled from OCR internals.
================================================================================
"""

from __future__ import annotations

import time
import numpy as np
import cv2

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:  # pragma: no cover
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:  # pragma: no cover
    TESSERACT_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────
# READER FACTORY
# ──────────────────────────────────────────────────────────────────────────
def load_ocr_reader(languages: list[str] | None = None, gpu: bool = False):
    """
    Create and return an EasyOCR Reader instance.

    This is the heaviest object in the app (loads detection + recognition
    neural networks, ~100MB+ of weights on first run). The caller
    (app.py) should cache this with st.cache_resource so it loads only
    once per session.

    Args:
        languages: List of language codes (e.g. ['en']). Defaults to English.
        gpu: Whether to use GPU acceleration if available.

    Returns:
        An easyocr.Reader instance.

    Raises:
        RuntimeError if EasyOCR is not installed.
    """
    if not EASYOCR_AVAILABLE:
        raise RuntimeError(
            "EasyOCR is not installed. Run: pip install easyocr"
        )

    if languages is None:
        languages = ["en"]

    return easyocr.Reader(languages, gpu=gpu)


# ──────────────────────────────────────────────────────────────────────────
# CORE EXTRACTION
# ──────────────────────────────────────────────────────────────────────────
def extract_text(
    reader,
    image: np.ndarray,
    min_confidence: float = 0.0,
) -> dict:
    """
    Run OCR on an image and return structured results.

    Args:
        reader: An initialized easyocr.Reader (from load_ocr_reader).
        image: Input image as a numpy array (BGR, as from OpenCV).
        min_confidence: Filter out detections below this confidence
                        (0.0 - 1.0). Use 0.0 to keep everything.

    Returns:
        dict with keys:
            'detections'   : list of dicts, each with
                              {text, confidence, bbox} where bbox is a
                              list of 4 (x, y) corner points.
            'full_text'    : all detected text joined with newlines.
            'avg_confidence': mean confidence across kept detections.
            'processing_time': seconds taken for the OCR call.
            'count'        : number of text regions detected.
    """
    if image is None:
        raise ValueError("extract_text: input image is None")

    start = time.time()

    # EasyOCR expects RGB; OpenCV images are BGR by default.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image

    raw_results = reader.readtext(rgb_image)
    elapsed = time.time() - start

    detections = []
    for bbox, text, confidence in raw_results:
        if confidence < min_confidence:
            continue
        detections.append({
            "text": text,
            "confidence": round(float(confidence), 4),
            "bbox": [[int(p[0]), int(p[1])] for p in bbox],
        })

    full_text = "\n".join(d["text"] for d in detections)
    avg_conf = (
        round(sum(d["confidence"] for d in detections) / len(detections), 4)
        if detections else 0.0
    )

    return {
        "detections": detections,
        "full_text": full_text,
        "avg_confidence": avg_conf,
        "processing_time": round(elapsed, 3),
        "count": len(detections),
    }


# ──────────────────────────────────────────────────────────────────────────
# TESSERACT FALLBACK (optional secondary engine)
# ──────────────────────────────────────────────────────────────────────────
def extract_text_tesseract(image: np.ndarray) -> dict:
    """
    Alternative OCR using pytesseract (requires system Tesseract binary).

    Useful as a fallback comparison engine or for environments where
    EasyOCR's deep-learning models are too heavy to install.

    Args:
        image: Input image as a numpy array (BGR).

    Returns:
        Same structured dict shape as extract_text().
    """
    if not TESSERACT_AVAILABLE:
        raise RuntimeError(
            "pytesseract is not installed. Run: pip install pytesseract "
            "and install the Tesseract OCR binary on your system."
        )

    start = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    elapsed = time.time() - start

    detections = []
    n_boxes = len(data["text"])
    for i in range(n_boxes):
        text = data["text"][i].strip()
        conf = float(data["conf"][i])
        if not text or conf < 0:
            continue

        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

        detections.append({
            "text": text,
            "confidence": round(conf / 100.0, 4),  # tesseract conf is 0-100
            "bbox": bbox,
        })

    full_text = "\n".join(d["text"] for d in detections)
    avg_conf = (
        round(sum(d["confidence"] for d in detections) / len(detections), 4)
        if detections else 0.0
    )

    return {
        "detections": detections,
        "full_text": full_text,
        "avg_confidence": avg_conf,
        "processing_time": round(elapsed, 3),
        "count": len(detections),
    }


# ──────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────
def draw_text_boxes(
    image: np.ndarray,
    detections: list[dict],
    box_color: tuple[int, int, int] = (0, 255, 120),
    thickness: int = 2,
    show_labels: bool = True,
) -> np.ndarray:
    """
    Draw bounding boxes and confidence labels for each OCR detection.

    Args:
        image: Original image (BGR) to draw on. A copy is made internally.
        detections: List of detection dicts as returned by extract_text().
        box_color: BGR color tuple for the boxes.
        thickness: Line thickness in pixels.
        show_labels: Whether to draw confidence-score labels above boxes.

    Returns:
        A new image (copy) with boxes and labels drawn.
    """
    if image is None:
        raise ValueError("draw_text_boxes: input image is None")

    output = image.copy()

    for det in detections:
        pts = np.array(det["bbox"], dtype=np.int32)
        cv2.polylines(output, [pts], isClosed=True, color=box_color, thickness=thickness)

        if show_labels:
            x, y = pts[0]
            label = f"{det['confidence'] * 100:.0f}%"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # Draw a filled background rectangle so the label is legible
            cv2.rectangle(
                output,
                (x, max(0, y - text_h - 8)),
                (x + text_w + 4, y),
                box_color,
                -1,
            )
            cv2.putText(
                output, label, (x + 2, max(12, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
            )

    return output
