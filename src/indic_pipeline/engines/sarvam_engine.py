"""
Sarvam AI Vision OCR engine connector for Indian languages.
"""

import os
from pathlib import Path
from typing import Optional
import httpx

from indic_pipeline.config import settings
from indic_pipeline.engines.base import BaseOCREngine, LineItem, OCRResult
from indic_pipeline.utils.logger import logger


class SarvamEngine(BaseOCREngine):
    BASE_URL = "https://api.sarvam.ai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        enable_cache: bool = True,
    ):
        super().__init__(name="sarvam_ocr", enable_cache=enable_cache)
        self.api_key = api_key or settings.sarvam_api_key or os.getenv("SARVAM_API_KEY")
        if not self.api_key:
            logger.warning("Sarvam API key not found. Set SARVAM_API_KEY to run live Sarvam OCR.")

    def _execute_transcribe(self, image_path: Path, language: str, prompt_hint: Optional[str] = None) -> OCRResult:
        if not self.api_key:
            raise ValueError("Sarvam API key is not configured. Set SARVAM_API_KEY in environment or settings.")

        headers = {
            "api-subscription-key": self.api_key,
        }

        # Map language code to Sarvam format if needed (e.g. kn-IN, hi-IN)
        lang_code = f"{language.lower()}-IN" if len(language) == 2 else language

        url = f"{self.BASE_URL}/ocr"

        with open(image_path, "rb") as f:
            files = {"file": (image_path.name, f, "image/png")}
            data = {"language_code": lang_code}

            with httpx.Client(timeout=settings.engine.timeout_seconds) as client:
                response = client.post(url, headers=headers, files=files, data=data)

                if response.status_code != 200:
                    raise RuntimeError(f"Sarvam API error (status {response.status_code}): {response.text}")

                res_json = response.json()

        # Parse response text and bounding boxes if present
        raw_text = res_json.get("transcript", "") or res_json.get("text", "")
        lines = []
        if "lines" in res_json:
            for item in res_json["lines"]:
                lines.append(
                    LineItem(
                        text=item.get("text", ""),
                        confidence=item.get("confidence"),
                        bounding_box=item.get("bbox"),
                    )
                )
        else:
            lines = [LineItem(text=line) for line in raw_text.splitlines() if line.strip()]

        return OCRResult(
            engine_name=self.name,
            image_path=str(image_path),
            language=language,
            raw_text=raw_text,
            lines=lines,
            metadata=res_json,
        )
