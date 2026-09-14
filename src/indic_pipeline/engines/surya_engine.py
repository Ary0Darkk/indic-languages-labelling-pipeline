"""
Surya OCR open-source model engine connector.
"""

from pathlib import Path
from typing import Optional
from PIL import Image

from indic_pipeline.engines.base import BaseOCREngine, LineItem, OCRResult
from indic_pipeline.utils.logger import logger


class SuryaEngine(BaseOCREngine):
    def __init__(self, enable_cache: bool = True):
        super().__init__(name="surya_ocr", enable_cache=enable_cache)
        self._initialized = False
        self._model = None
        self._processor = None

    def _lazy_init(self):
        if not self._initialized:
            try:
                import importlib
                rec_module = importlib.import_module("surya.recognition")
                det_module = importlib.import_module("surya.detection")

                logger.info("Initializing Surya OCR models...")
                self._rec_model = getattr(rec_module, "RecognitionPredictor")()
                self._det_model = getattr(det_module, "DetectionPredictor")()
                self._initialized = True
            except (ImportError, ModuleNotFoundError) as e:
                raise ImportError(
                    "Surya OCR is not installed. To use surya_engine, install it with `pip install surya-ocr`."
                ) from e

    def _execute_transcribe(self, image_path: Path, language: str, prompt_hint: Optional[str] = None) -> OCRResult:
        self._lazy_init()
        pil_img = Image.open(image_path).convert("RGB")

        # Surya predictions
        predictions = self._rec_model([pil_img], langs=[[language]])

        lines = []
        raw_text_parts = []
        if predictions and len(predictions) > 0:
            pred = predictions[0]
            for text_line in pred.text_lines:
                lines.append(
                    LineItem(
                        text=text_line.text,
                        confidence=text_line.confidence,
                        bounding_box=text_line.bbox,
                    )
                )
                raw_text_parts.append(text_line.text)

        raw_text = "\n".join(raw_text_parts)

        return OCRResult(
            engine_name=self.name,
            image_path=str(image_path),
            language=language,
            raw_text=raw_text,
            lines=lines,
        )
