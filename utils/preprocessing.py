"""
================================================================================
 VisionAI Recognizer | utils/preprocessing.py
 Module: Image Preprocessing Engine
--------------------------------------------------------------------------------
 Provides classical computer-vision preprocessing operations that improve
 the quality of input images BEFORE they are passed to OCR or object
 detection models. Each function is independent, type-hinted, and wrapped
 in error handling so a single bad operation never crashes the pipeline.
================================================================================
"""

from __future__ import annotations

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────
# 1. GRAYSCALE CONVERSION
# ──────────────────────────────────────────────────────────────────────────
def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR/RGB image to single-channel grayscale.

    Why: OCR engines and many thresholding algorithms work on intensity
    values only — color information is unnecessary and adds noise.

    Args:
        image: Input image (H, W, 3) or already grayscale (H, W).

    Returns:
        Grayscale image (H, W).
    """
    if image is None:
        raise ValueError("to_grayscale: input image is None")

    if len(image.shape) == 2:
        # Already grayscale
        return image

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ──────────────────────────────────────────────────────────────────────────
# 2. NOISE REDUCTION
# ──────────────────────────────────────────────────────────────────────────
def reduce_noise(image: np.ndarray, strength: int = 10) -> np.ndarray:
    """
    Apply Non-Local Means Denoising to remove sensor / compression noise.

    Why: Noisy images (e.g. low-light phone photos) confuse OCR and
    object detectors. Denoising smooths random pixel variation while
    preserving edges — critical for keeping text legible.

    Args:
        image: Input image (grayscale or color).
        strength: Filter strength (h parameter). Higher = more smoothing.

    Returns:
        Denoised image, same shape and dtype as input.
    """
    if image is None:
        raise ValueError("reduce_noise: input image is None")

    try:
        if len(image.shape) == 2:
            return cv2.fastNlMeansDenoising(image, None, h=strength)
        return cv2.fastNlMeansDenoisingColored(image, None, h=strength, hColor=strength)
    except cv2.error:
        # Fallback to a cheaper Gaussian blur if NLM fails on huge images
        return cv2.GaussianBlur(image, (3, 3), 0)


# ──────────────────────────────────────────────────────────────────────────
# 3. THRESHOLDING (BINARIZATION)
# ──────────────────────────────────────────────────────────────────────────
def apply_threshold(image: np.ndarray, method: str = "adaptive") -> np.ndarray:
    """
    Convert a grayscale image to pure black/white (binary).

    Why: OCR accuracy improves dramatically on high-contrast binary images
    where text is solid black on solid white (or vice-versa).

    Args:
        image: Grayscale image (H, W). Will be converted if color is passed.
        method: 'adaptive' (handles uneven lighting) | 'otsu' (global threshold).

    Returns:
        Binary image (H, W) with values {0, 255}.
    """
    if image is None:
        raise ValueError("apply_threshold: input image is None")

    gray = to_grayscale(image)

    if method == "adaptive":
        # Adaptive: threshold is computed locally per neighborhood —
        # robust to shadows / uneven lighting across a document/photo.
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )
    elif method == "otsu":
        # Otsu: automatically picks a single global threshold that best
        # separates foreground/background — fast, works on uniform lighting.
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    else:
        raise ValueError(f"apply_threshold: unknown method '{method}'")


# ──────────────────────────────────────────────────────────────────────────
# 4. CONTRAST ENHANCEMENT
# ──────────────────────────────────────────────────────────────────────────
def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Improve local contrast using CLAHE (Contrast Limited Adaptive
    Histogram Equalization).

    Why: Photos taken in poor lighting often have washed-out or overly
    dark regions. CLAHE redistributes intensity values locally, making
    faint text and object edges much more visible without blowing out
    bright areas.

    Args:
        image: Input image, grayscale or color.
        clip_limit: Contrast amplification limit (higher = stronger effect).

    Returns:
        Contrast-enhanced image, same shape as input.
    """
    if image is None:
        raise ValueError("enhance_contrast: input image is None")

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))

    if len(image.shape) == 2:
        return clahe.apply(image)

    # For color images: apply CLAHE on the L-channel of LAB color space
    # to boost contrast without distorting hue/saturation.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


# ──────────────────────────────────────────────────────────────────────────
# 5. IMAGE RESIZING
# ──────────────────────────────────────────────────────────────────────────
def resize_image(
    image: np.ndarray,
    max_dimension: int = 1280,
) -> np.ndarray:
    """
    Resize an image so its largest dimension does not exceed max_dimension,
    preserving aspect ratio.

    Why: Very large images slow down OCR / detection models and consume
    excessive memory. Very small images lose detail needed for accurate
    recognition. This keeps inputs in a sensible, consistent range.

    Args:
        image: Input image.
        max_dimension: Target maximum width/height in pixels.

    Returns:
        Resized image. If already within bounds, returns original unchanged.
    """
    if image is None:
        raise ValueError("resize_image: input image is None")

    h, w = image.shape[:2]
    longest_side = max(h, w)

    if longest_side <= max_dimension:
        return image

    scale = max_dimension / float(longest_side)
    new_w, new_h = int(w * scale), int(h * scale)

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


# ──────────────────────────────────────────────────────────────────────────
# 6. ROTATION CORRECTION (DESKEW)
# ──────────────────────────────────────────────────────────────────────────
def correct_rotation(image: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Detect and correct skew/rotation in an image — particularly useful
    for scanned documents or photos taken at an angle.

    Method: Uses minAreaRect on thresholded foreground pixels to estimate
    the dominant text/content angle, then rotates the image to level it.

    Args:
        image: Input image (grayscale or color).

    Returns:
        Tuple of (deskewed_image, detected_angle_in_degrees).
    """
    if image is None:
        raise ValueError("correct_rotation: input image is None")

    gray = to_grayscale(image)

    # Invert so foreground (text/objects) is white on black background
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(binary > 0))

    if coords.shape[0] < 10:
        # Not enough foreground pixels to estimate angle reliably
        return image, 0.0

    angle = cv2.minAreaRect(coords)[-1]

    # minAreaRect returns angles in range [-90, 0); normalize to [-45, 45]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Skip correction for negligible angles to avoid unnecessary blur
    if abs(angle) < 0.5:
        return image, 0.0

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        image, rotation_matrix, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return rotated, float(angle)


# ──────────────────────────────────────────────────────────────────────────
# PIPELINE — apply a configurable sequence of operations
# ──────────────────────────────────────────────────────────────────────────
def preprocess_pipeline(
    image: np.ndarray,
    steps: list[str] | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Run a configurable sequence of preprocessing steps and return both
    the processed image and a metadata log of what was applied.

    Args:
        image: Original input image (BGR, as read by OpenCV/Streamlit).
        steps: Ordered list of step names to apply. Supported values:
               'resize', 'denoise', 'contrast', 'grayscale',
               'threshold', 'deskew'.
               If None, a sensible default pipeline is used.

    Returns:
        Tuple of (processed_image, metadata_dict).
        metadata_dict contains info such as detected rotation angle
        and the final image shape.
    """
    if image is None:
        raise ValueError("preprocess_pipeline: input image is None")

    if steps is None:
        steps = ["resize", "denoise", "contrast"]

    result = image.copy()
    meta: dict = {"steps_applied": [], "rotation_angle": 0.0}

    for step in steps:
        try:
            if step == "resize":
                result = resize_image(result)
            elif step == "denoise":
                result = reduce_noise(result)
            elif step == "contrast":
                result = enhance_contrast(result)
            elif step == "grayscale":
                result = to_grayscale(result)
            elif step == "threshold":
                result = apply_threshold(result, method="adaptive")
            elif step == "deskew":
                result, angle = correct_rotation(result)
                meta["rotation_angle"] = angle
            else:
                continue
            meta["steps_applied"].append(step)
        except Exception as exc:  # noqa: BLE001 - log and continue pipeline
            meta.setdefault("errors", []).append(f"{step}: {exc}")

    meta["final_shape"] = result.shape
    return result, meta
