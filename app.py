"""
================================================================================
 VisionAI Recognizer | app.py
 Production-quality Image & Text Recognition Dashboard
--------------------------------------------------------------------------------
 A Streamlit application that lets a user upload an image and:
   1. Apply classical CV preprocessing (grayscale, denoise, threshold,
      contrast enhancement, resize, rotation correction).
   2. Run OCR (EasyOCR) to extract text with confidence scores and
      bounding boxes.
   3. Run object detection (YOLOv8) to identify and localize objects
      with confidence scores.
   4. View statistics, confidence charts, and export results as
      CSV / TXT.

 Run with:  streamlit run app.py
================================================================================
"""

from __future__ import annotations

import io
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

from utils import preprocessing as prep
from utils import ocr_engine as ocr
from utils import object_detector as det


# ════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION & GLOBAL STYLE (Dark Mode Dashboard)
# ════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="VisionAI Recognizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    /* ---- Overall dark theme ---- */
    .stApp {
        background-color: #0e1117;
        color: #e6e6e6;
    }

    /* ---- Headings ---- */
    h1, h2, h3 {
        color: #58a6ff;
        font-family: 'Segoe UI', sans-serif;
    }

    /* ---- Metric cards ---- */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #161b22, #1f242c);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 14px 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }

    /* ---- Buttons ---- */
    .stButton > button {
        background: linear-gradient(135deg, #1f6feb, #58a6ff);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.55rem 1.2rem;
        transition: transform 0.08s ease-in-out;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(88,166,255,0.35);
    }

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        border-radius: 8px 8px 0 0;
        padding: 8px 18px;
        color: #c9d1d9;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f6feb !important;
        color: white !important;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* ---- Dataframe ---- */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* ---- Caption / helper text ---- */
    .vision-caption {
        color: #8b949e;
        font-size: 0.85rem;
    }

    /* ---- Badge ---- */
    .vision-badge {
        display: inline-block;
        background: #1f6feb22;
        color: #58a6ff;
        border: 1px solid #1f6feb55;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.8rem;
        margin-right: 6px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# CACHED MODEL LOADERS
# Models are large/expensive to initialize — load them ONCE per session.
# ════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading OCR engine (EasyOCR)...")
def get_ocr_reader():
    """Cache and return the EasyOCR reader (English)."""
    return ocr.load_ocr_reader(languages=["en"], gpu=False)


@st.cache_resource(show_spinner="Loading object detection model (YOLOv8)...")
def get_object_model():
    """Cache and return the YOLOv8n object detection model."""
    return det.load_detector("yolov8n.pt")


# ════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════
def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image to an OpenCV BGR numpy array."""
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv2_to_display(image: np.ndarray) -> np.ndarray:
    """Convert a BGR/grayscale OpenCV image to RGB for st.image display."""
    if image is None:
        return image
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def image_to_bytes(image: np.ndarray, fmt: str = ".png") -> bytes:
    """Encode an OpenCV image to bytes for download buttons."""
    rgb = cv2_to_display(image)
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG" if fmt == ".png" else "JPEG")
    return buf.getvalue()


def make_confidence_chart(values: list[float], labels: list[str], title: str):
    """Build a horizontal bar chart of confidence scores using matplotlib,
    styled to match the dark dashboard theme."""
    fig, ax = plt.subplots(figsize=(6, max(2, 0.35 * len(values))))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    colors = ["#58a6ff" if v >= 0.5 else "#f78166" for v in values]
    bars = ax.barh(labels, [v * 100 for v in values], color=colors)

    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence (%)", color="#c9d1d9")
    ax.set_title(title, color="#58a6ff", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#c9d1d9")
    for spine in ax.spines.values():
        spine.set_color("#30363d")

    for bar, v in zip(bars, values):
        ax.text(
            min(v * 100 + 2, 96), bar.get_y() + bar.get_height() / 2,
            f"{v*100:.1f}%", va="center", color="#e6e6e6", fontsize=8,
        )

    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════
# SIDEBAR — CONTROLS
# ════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.markdown("---")

    st.markdown("#### 📤 Upload")
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        help="Upload a photo, scanned document, or screenshot.",
    )

    st.markdown("---")
    st.markdown("#### 🛠️ Preprocessing")

    preprocess_steps = st.multiselect(
        "Pipeline steps (applied in order)",
        options=["resize", "deskew", "denoise", "contrast", "grayscale", "threshold"],
        default=["resize", "denoise", "contrast"],
        help="Choose which classical CV operations to apply before recognition.",
    )

    st.markdown("---")
    st.markdown("#### 🔎 OCR Settings")
    ocr_min_conf = st.slider(
        "Minimum OCR confidence", 0.0, 1.0, 0.10, 0.05,
        help="Filter out low-confidence text detections.",
    )

    st.markdown("---")
    st.markdown("#### 🎯 Object Detection Settings")
    det_min_conf = st.slider(
        "Minimum detection confidence", 0.10, 0.90, 0.25, 0.05,
        help="Filter out low-confidence object detections.",
    )

    st.markdown("---")
    st.markdown(
        '<span class="vision-badge">OpenCV</span>'
        '<span class="vision-badge">EasyOCR</span>'
        '<span class="vision-badge">YOLOv8</span>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <h1>🧠 VisionAI Recognizer</h1>
    <p class="vision-caption">
        An end-to-end Image &amp; Text Recognition dashboard — preprocess,
        extract text with OCR, and detect objects with a pre-trained
        YOLOv8 model. Built with OpenCV, EasyOCR, and Ultralytics.
    </p>
    """,
    unsafe_allow_html=True,
)
st.markdown("---")


# ════════════════════════════════════════════════════════════════════════
# MAIN WORKFLOW
# ════════════════════════════════════════════════════════════════════════
if uploaded_file is None:
    st.info(
        "👈 Upload an image from the sidebar to get started. "
        "Try a photo with visible text, a scanned document, or a "
        "scene containing everyday objects (people, cars, animals, etc.)."
    )
    st.markdown(
        """
        ### What this app does
        1. **Preprocessing** — cleans up the image (denoise, contrast,
           deskew, threshold) so downstream models perform better.
        2. **Text Recognition (OCR)** — extracts any readable text using
           EasyOCR, with per-word confidence scores and bounding boxes.
        3. **Object Detection** — identifies and localizes common objects
           (80 COCO classes) using YOLOv8, with confidence scores.
        4. **Export** — download extracted text as `.txt` and detection
           results as `.csv`.
        """
    )
else:
    # ── Load image ──────────────────────────────────────────────────────
    pil_image = Image.open(uploaded_file)
    original_bgr = pil_to_cv2(pil_image)

    # ── Run preprocessing pipeline ─────────────────────────────────────
    processed_bgr, prep_meta = prep.preprocess_pipeline(
        original_bgr, steps=preprocess_steps
    )

    # ── Tabs layout ──────────────────────────────────────────────────────
    tab_overview, tab_ocr, tab_objects, tab_export = st.tabs(
        ["🖼️ Overview", "📝 Text Recognition (OCR)", "📦 Object Detection", "📥 Export"]
    )

    # ────────────────────────────────────────────────────────────────────
    # TAB 1: OVERVIEW
    # ────────────────────────────────────────────────────────────────────
    with tab_overview:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(cv2_to_display(original_bgr), use_container_width=True)
            st.caption(
                f"Shape: {original_bgr.shape[1]} × {original_bgr.shape[0]} px"
            )

        with col2:
            st.subheader("Processed Image")
            st.image(cv2_to_display(processed_bgr), use_container_width=True)
            applied = ", ".join(prep_meta["steps_applied"]) or "none"
            st.caption(f"Steps applied: {applied}")
            if prep_meta.get("rotation_angle"):
                st.caption(f"Detected rotation correction: {prep_meta['rotation_angle']:.2f}°")
            if prep_meta.get("errors"):
                for err in prep_meta["errors"]:
                    st.warning(f"Preprocessing warning — {err}")

        st.markdown("---")
        st.markdown(
            """
            #### Pipeline Reference
            | Step | Purpose |
            |---|---|
            | **resize** | Caps image size for speed/memory consistency |
            | **deskew** | Detects & corrects rotation/skew |
            | **denoise** | Removes sensor/compression noise (Non-Local Means) |
            | **contrast** | CLAHE local contrast enhancement |
            | **grayscale** | Single-channel conversion |
            | **threshold** | Adaptive binarization (black/white) |
            """
        )

    # ────────────────────────────────────────────────────────────────────
    # TAB 2: OCR
    # ────────────────────────────────────────────────────────────────────
    with tab_ocr:
        st.subheader("Text Recognition")

        run_ocr = st.button("▶️ Run OCR", key="run_ocr_btn")

        if run_ocr:
            try:
                reader = get_ocr_reader()
                with st.spinner("Extracting text..."):
                    ocr_result = ocr.extract_text(
                        reader, processed_bgr, min_confidence=ocr_min_conf
                    )
                st.session_state["ocr_result"] = ocr_result
            except Exception as exc:  # noqa: BLE001
                st.error(f"OCR failed: {exc}")

        ocr_result = st.session_state.get("ocr_result")

        if ocr_result:
            # ── Metrics row ──────────────────────────────────────────────
            m1, m2, m3 = st.columns(3)
            m1.metric("Text Regions Detected", ocr_result["count"])
            m2.metric("Average Confidence", f"{ocr_result['avg_confidence']*100:.1f}%")
            m3.metric("Processing Time", f"{ocr_result['processing_time']}s")

            st.markdown("---")

            col_img, col_text = st.columns([1.1, 1])

            with col_img:
                st.markdown("##### Detected Text Regions")
                if ocr_result["detections"]:
                    boxed = ocr.draw_text_boxes(processed_bgr, ocr_result["detections"])
                    st.image(cv2_to_display(boxed), use_container_width=True)
                else:
                    st.info("No text detected above the confidence threshold.")

            with col_text:
                st.markdown("##### Extracted Text")
                st.text_area(
                    "Full text", ocr_result["full_text"] or "(no text detected)",
                    height=220,
                )

                if ocr_result["detections"]:
                    st.markdown("##### Confidence by Region")
                    df_ocr = pd.DataFrame(
                        [
                            {"Text": d["text"][:30], "Confidence": d["confidence"]}
                            for d in ocr_result["detections"]
                        ]
                    )
                    st.dataframe(df_ocr, use_container_width=True, height=180)

            if ocr_result["detections"]:
                st.markdown("---")
                st.markdown("##### Confidence Visualization")
                labels = [d["text"][:20] or "(blank)" for d in ocr_result["detections"]]
                values = [d["confidence"] for d in ocr_result["detections"]]
                fig = make_confidence_chart(values, labels, "OCR Detection Confidence")
                st.pyplot(fig)
        else:
            st.markdown(
                '<p class="vision-caption">Click "Run OCR" to extract text '
                "from the processed image.</p>",
                unsafe_allow_html=True,
            )

    # ────────────────────────────────────────────────────────────────────
    # TAB 3: OBJECT DETECTION
    # ────────────────────────────────────────────────────────────────────
    with tab_objects:
        st.subheader("Object Detection")

        run_detection = st.button("▶️ Run Object Detection", key="run_det_btn")

        if run_detection:
            try:
                model = get_object_model()
                with st.spinner("Detecting objects..."):
                    det_result = det.detect_objects(
                        model, processed_bgr, confidence_threshold=det_min_conf
                    )
                st.session_state["det_result"] = det_result
            except Exception as exc:  # noqa: BLE001
                st.error(f"Object detection failed: {exc}")

        det_result = st.session_state.get("det_result")

        if det_result:
            m1, m2, m3 = st.columns(3)
            m1.metric("Objects Detected", det_result["count"])
            m2.metric("Unique Classes", len(det_result["class_counts"]))
            m3.metric("Processing Time", f"{det_result['processing_time']}s")

            st.markdown("---")

            col_img, col_data = st.columns([1.1, 1])

            with col_img:
                st.markdown("##### Detected Objects")
                if det_result["detections"]:
                    boxed = det.draw_object_boxes(processed_bgr, det_result["detections"])
                    st.image(cv2_to_display(boxed), use_container_width=True)
                else:
                    st.info("No objects detected above the confidence threshold.")

            with col_data:
                if det_result["detections"]:
                    st.markdown("##### Detection Table")
                    df_det = pd.DataFrame(
                        [
                            {
                                "Label": d["label"],
                                "Confidence": d["confidence"],
                                "BBox (x1,y1,x2,y2)": str(d["bbox"]),
                            }
                            for d in det_result["detections"]
                        ]
                    )
                    st.dataframe(df_det, use_container_width=True, height=220)

                    st.markdown("##### Object Class Distribution")
                    df_counts = pd.DataFrame(
                        list(det_result["class_counts"].items()),
                        columns=["Class", "Count"],
                    ).sort_values("Count", ascending=False)
                    st.bar_chart(df_counts.set_index("Class"))

            if det_result["detections"]:
                st.markdown("---")
                st.markdown("##### Confidence Visualization")
                labels = [
                    f"{d['label']} #{i+1}"
                    for i, d in enumerate(det_result["detections"])
                ]
                values = [d["confidence"] for d in det_result["detections"]]
                fig = make_confidence_chart(values, labels, "Object Detection Confidence")
                st.pyplot(fig)
        else:
            st.markdown(
                '<p class="vision-caption">Click "Run Object Detection" to '
                "identify objects in the processed image. The first run "
                "downloads the YOLOv8n weights (~6MB).</p>",
                unsafe_allow_html=True,
            )

    # ────────────────────────────────────────────────────────────────────
    # TAB 4: EXPORT
    # ────────────────────────────────────────────────────────────────────
    with tab_export:
        st.subheader("Export Results")

        ocr_result = st.session_state.get("ocr_result")
        det_result = st.session_state.get("det_result")

        col1, col2, col3 = st.columns(3)

        # ── Export OCR text as TXT ─────────────────────────────────────
        with col1:
            st.markdown("##### 📝 OCR Text (.txt)")
            if ocr_result and ocr_result["full_text"]:
                st.download_button(
                    "Download extracted text",
                    data=ocr_result["full_text"],
                    file_name="visionai_extracted_text.txt",
                    mime="text/plain",
                )
            else:
                st.caption("Run OCR first to enable this export.")

        # ── Export OCR detections as CSV ───────────────────────────────
        with col2:
            st.markdown("##### 📊 OCR Results (.csv)")
            if ocr_result and ocr_result["detections"]:
                df_ocr_export = pd.DataFrame(ocr_result["detections"])
                st.download_button(
                    "Download OCR CSV",
                    data=df_ocr_export.to_csv(index=False),
                    file_name="visionai_ocr_results.csv",
                    mime="text/csv",
                )
            else:
                st.caption("Run OCR first to enable this export.")

        # ── Export Object Detection results as CSV ─────────────────────
        with col3:
            st.markdown("##### 📦 Detection Results (.csv)")
            if det_result and det_result["detections"]:
                df_det_export = pd.DataFrame(det_result["detections"])
                st.download_button(
                    "Download Detection CSV",
                    data=df_det_export.to_csv(index=False),
                    file_name="visionai_object_detections.csv",
                    mime="text/csv",
                )
            else:
                st.caption("Run object detection first to enable this export.")

        st.markdown("---")
        st.markdown("##### 🖼️ Processed Image")
        st.download_button(
            "Download processed image (PNG)",
            data=image_to_bytes(processed_bgr),
            file_name="visionai_processed_image.png",
            mime="image/png",
        )

        # ── Combined summary report ─────────────────────────────────────
        if ocr_result or det_result:
            st.markdown("---")
            st.markdown("##### 📋 Summary Report")
            summary_lines = ["VisionAI Recognizer — Summary Report", "=" * 40]

            if ocr_result:
                summary_lines += [
                    "",
                    "OCR RESULTS",
                    f"  Text regions detected : {ocr_result['count']}",
                    f"  Average confidence    : {ocr_result['avg_confidence']*100:.1f}%",
                    f"  Processing time       : {ocr_result['processing_time']}s",
                    "",
                    "  Extracted text:",
                    "  " + (ocr_result["full_text"].replace("\n", "\n  ") or "(none)"),
                ]

            if det_result:
                summary_lines += [
                    "",
                    "OBJECT DETECTION RESULTS",
                    f"  Objects detected      : {det_result['count']}",
                    f"  Unique classes        : {len(det_result['class_counts'])}",
                    f"  Processing time       : {det_result['processing_time']}s",
                    "",
                    "  Class breakdown:",
                ]
                for label, cnt in det_result["class_counts"].items():
                    summary_lines.append(f"    - {label}: {cnt}")

            summary_text = "\n".join(summary_lines)
            st.text_area("Report preview", summary_text, height=240)
            st.download_button(
                "Download full report (.txt)",
                data=summary_text,
                file_name="visionai_summary_report.txt",
                mime="text/plain",
            )


# ════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    '<p class="vision-caption">VisionAI Recognizer · Built with Streamlit, '
    "OpenCV, EasyOCR &amp; YOLOv8 · ",
    unsafe_allow_html=True,
)
