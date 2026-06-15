# 🧠 VisionAI Recognizer

**An end-to-end Image & Text Recognition Dashboard powered by OpenCV, EasyOCR, and YOLOv8.**

VisionAI Recognizer is a computer vision web application that lets users upload any image and instantly extract readable text with confidence scores, detect and label 80+ object types, and apply classical image-processing techniques like denoising, contrast enhancement, and deskewing — all through an interactive, dark-themed Streamlit dashboard with exportable results.

---

## 🌐 Live Demo

  https://visionai-recognizerafvh9zlxgxg66py274ximv.streamlit.app/

>  On first load, the OCR and object detection models download automatically (one-time setup) — please allow a moment for the app to initialize.

---

##  Features

-  **Text Recognition (OCR)** — Extract text from images using EasyOCR, with per-region confidence scores and bounding-box overlays
-  **Object Detection** — Identify and localize 80+ object classes using a pre-trained YOLOv8 model
-  **Configurable Preprocessing Pipeline** — Denoising, contrast enhancement (CLAHE), adaptive/Otsu thresholding, resizing, and automatic rotation correction (deskew)
-  **Interactive Dashboard** — Side-by-side image comparison, confidence charts, detection tables, and class distribution graphs
-  **Export Results** — Download extracted text (`.txt`), detection data (`.csv`), processed images (`.png`), and a combined summary report
-  **Dark Mode UI** — Modern, custom-styled Streamlit interface
-  **Cached Model Loading** — Heavy ML models load once per session for a responsive experience

---

##  How It Works

1. **Upload** an image (photo, scanned document, or screenshot)
2. **Preprocess** — apply a configurable pipeline (resize, deskew, denoise, contrast enhancement, grayscale, thresholding)
3. **Extract Text** — run OCR to detect and read text with confidence scores
4. **Detect Objects** — run YOLOv8 to identify and localize objects with labeled bounding boxes
5. **Export** — download all results as TXT, CSV, PNG, or a combined summary report

---


---

## 🚀 Run Locally

### Prerequisites

- Python **3.10 or 3.11** recommended (some dependencies lack pre-built wheels for newer Python versions on Windows)
- pip
- Internet connection for first-run model downloads

### 1. Clone the repository

```bash
git clone https://github.com/toobafatima21ai-hue/visionai-recognizer.git
cd visionai-recognizer
```

### 2. Create and activate a virtual environment

**macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

> If PowerShell blocks the activation script, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **PyTorch note:** EasyOCR and YOLOv8 depend on PyTorch. If installation is slow or fails, install the CPU-only build first:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

### 4. Run the application

```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**.

---

## Module Overview

### `utils/preprocessing.py` — Image Preprocessing Engine

| Function | Purpose |
|---|---|
| `to_grayscale()` | Converts to single-channel grayscale |
| `reduce_noise()` | Non-Local Means denoising |
| `enhance_contrast()` | CLAHE local contrast enhancement |
| `apply_threshold()` | Adaptive or Otsu binarization |
| `resize_image()` | Caps max dimension while preserving aspect ratio |
| `correct_rotation()` | Detects and corrects image skew |
| `preprocess_pipeline()` | Runs a configurable, ordered sequence of the above |

### `utils/ocr_engine.py` — Text Recognition

- `load_ocr_reader()` — initializes EasyOCR (English by default)
- `extract_text()` — returns extracted text, confidence scores, and bounding boxes
- `draw_text_boxes()` — overlays detection boxes and confidence labels on the image
- `extract_text_tesseract()` — optional fallback OCR engine using pytesseract

### `utils/object_detector.py` — Object Detection

- `load_detector()` — loads a pre-trained YOLOv8n model (COCO, 80 classes)
- `detect_objects()` — returns labels, confidence scores, bounding boxes, and class counts
- `draw_object_boxes()` — overlays color-coded bounding boxes and labels

### `app.py` — Dashboard

- **Sidebar** — upload images, configure the preprocessing pipeline and confidence thresholds
- **Overview tab** — original vs. processed image comparison
- **OCR tab** — extracted text, confidence table & chart, annotated image
- **Object Detection tab** — annotated image, detection table, class distribution, confidence chart
- **Export tab** — download results as TXT / CSV / PNG and a combined summary report

---

## Configuration

All settings are adjustable live from the sidebar:

- **Preprocessing steps** — toggle and reorder: `resize`, `deskew`, `denoise`, `contrast`, `grayscale`, `threshold`
- **OCR confidence threshold** — filter out low-confidence text detections
- **Object detection confidence threshold** — filter out low-confidence object detections

---

##  Tech Stack

| Category | Tools |
|---|---|
| UI Framework | Streamlit |
| Computer Vision | OpenCV |
| Text Recognition | EasyOCR, pytesseract (optional) |
| Object Detection | Ultralytics YOLOv8 |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib |
| Deployment | Streamlit Community Cloud |

---

 
---

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to open a pull request or issue.

 

---

## Author

Built by **Tooba Fatima** as part of the **DecodeLabs AI Industrial Training Program — Batch 2026**.

- GitHub: [@toobafatima21ai-hue](https://github.com/toobafatima21ai-hue)
