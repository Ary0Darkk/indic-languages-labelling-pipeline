# Indic Languages OCR & Handwriting (HTR) Pipeline

A high-performance, modular Python pipeline designed specifically for transcribing and benchmarking **mixed printed and handwritten Indic script documents** (Kannada, Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, etc.).

---

## Key Features

- **Hybrid OCR / VLM Architecture**: Connectors for Google Gemini Multimodal Vision, Sarvam AI Vision OCR, Google Cloud Vision, and open-source models (Surya OCR, Tesseract).
- **Indic Script Normalization**: Canonical Unicode NFC normalization, zero-width character handling (`ZWJ` \u200D, `ZWNJ` \u200C), nukta canonicalization, and Indic-to-ASCII digit conversion to prevent false CER inflation.
- **Accurate CER & WER Metrics**: Fast Levenshtein distance calculation with detailed edit breakdowns (Substitutions, Deletions, Insertions).
- **Comprehensive Benchmarking & Reporting**:
  - Console summary tables with color-coded CER/WER indicators.
  - CSV & JSON detailed per-page metrics export.
  - Interactive standalone HTML report with visual side-by-side ground truth vs. model predictions.
- **Image Preprocessing Subsystem**: Illumination flattening, CLAHE contrast enhancement for faded handwriting, automatic deskewing, and resolution upscaling for crisp diacritics.
- **Disk Caching**: Avoid duplicate billing on repeat API runs.

---

## Installation & Setup

1. **Clone and Install Dependencies with uv / pip**:
   ```bash
   # Using uv
   uv sync

   # Or standard pip
   pip install -e .
   ```

2. **Configure API Keys (Optional)**:
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   SARVAM_API_KEY=your_sarvam_api_key_here
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
   ```

---

## Quickstart & CLI Usage

### 1. Run Multi-Engine Benchmark
Benchmark your candidate OCR/VLM engines against pilot pages (paired with `.json` or `.txt` ground truth):
```bash
# Run benchmark on Kannada sample dataset using mock simulation
indic-ocr benchmark --data-dir ./data/samples --lang kn --engines mock

# Run benchmark using live Gemini Vision
indic-ocr benchmark --data-dir ./data/samples --lang kn --engines gemini
```

Outputs generated:
- Console summary table
- `data/outputs/benchmark_per_page.csv`
- `data/outputs/benchmark_summary.json`
- `data/outputs/benchmark_report.html` (Open in browser to inspect side-by-side visual diffs!)

---

### 2. Transcribe a Single Page
```bash
indic-ocr transcribe --input ./data/samples/kn_sample_01.png --lang kn --engine gemini --output ./transcription.txt
```

---

### 3. Evaluate Ground Truth vs Prediction
```bash
indic-ocr evaluate --gt ./data/samples/kn_sample_01.json --pred ./transcription.txt --lang kn
```

---

### 4. Preprocess Raw Scans / Photos
```bash
indic-ocr preprocess --input ./raw_scans/ --output ./cleaned_scans/
```

---

### 5. Clear Cached Responses
```bash
indic-ocr clear-cache
```

---

## Python API Usage

```python
from indic_pipeline.engines import GeminiEngine, SarvamEngine
from indic_pipeline.evaluation import IndicNormalizer, evaluate_prediction
from indic_pipeline.preprocessing import PreprocessingPipeline

# 1. Preprocess Image
pipeline = PreprocessingPipeline()
enhanced_img_path = pipeline.process_to_temp_file("document.png")

# 2. Transcribe with Gemini Multimodal Vision
engine = GeminiEngine(model_name="gemini-2.5-flash")
result = engine.transcribe(enhanced_img_path, language="kn")
print(result.raw_text)

# 3. Evaluate against ground truth with Indic normalization
metrics = evaluate_prediction(
    reference="ಕರ್ನಾಟಕ ರಾಜ್ಯ ಸರ್ಕಾರ",
    hypothesis=result.raw_text,
    normalizer=IndicNormalizer(ignore_zwnj=True, ignore_zwj=True)
)
print(f"CER: {metrics.cer * 100:.2f}%, WER: {metrics.wer * 100:.2f}%")
```

---

## Running Tests
```bash
uv run pytest
```
