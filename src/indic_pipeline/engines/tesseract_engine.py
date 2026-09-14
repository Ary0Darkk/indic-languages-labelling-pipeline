"""
Tesseract OCR engine connector with Indic script language support.
"""

from pathlib import Path
from typing import Optional
from PIL import Image

from indic_pipeline.config import settings
from indic_pipeline.engines.base import BaseOCREngine, LineItem, OCRResult
from indic_pipeline.utils.logger import logger


class TesseractEngine(BaseOCREngine):
    LANG_CODE_MAP = {
        "kn": "kan",
        "hi": "hin",
        "bn": "ben",
        "ta": "tam",
        "te": "tel",
        "mr": "mar",
        "gu": "guj",
        "ml": "mal",
        "or": "ori",
        "pa": "pan",
        "as": "asm",
    }

    def __init__(self, tesseract_cmd: Optional[str] = None, enable_cache: bool = True):
        super().__init__(name="tesseract", enable_cache=enable_cache)
        self.tesseract_cmd = tesseract_cmd or settings.engine.tesseract_cmd

    def _execute_transcribe(self, image_path: Path, language: str, prompt_hint: Optional[str] = None) -> OCRResult:
        try:
            import importlib
            pytesseract = importlib.import_module("pytesseract")
        except (ImportError, ModuleNotFoundError) as e:
            raise ImportError("pytesseract is not installed. Run `pip install pytesseract` to use Tesseract.") from e

        if self.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

        tess_lang = self.LANG_CODE_MAP.get(language.lower(), language)
        pil_img = Image.open(image_path)

        # Include eng as fallback for numbers/mixed headers
        combo_lang = f"{tess_lang}+eng"

        try:
            raw_text = pytesseract.image_to_string(pil_img, lang=combo_lang)
        except Exception as e:
            logger.warning(f"Failed with {combo_lang}, trying {tess_lang}: {e}")
            raw_text = pytesseract.image_to_string(pil_img, lang=tess_lang)

        lines = [LineItem(text=line) for line in raw_text.splitlines() if line.strip()]

        return OCRResult(
            engine_name=self.name,
            image_path=str(image_path),
            language=language,
            raw_text=raw_text.strip(),
            lines=lines,
        )
