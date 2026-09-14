"""
Google Cloud Vision OCR engine connector.
"""

import base64
import os
from pathlib import Path
from typing import Optional
import httpx

from indic_pipeline.config import settings
from indic_pipeline.engines.base import BaseOCREngine, LineItem, OCRResult
from indic_pipeline.utils.logger import logger


class GoogleCloudVisionEngine(BaseOCREngine):
    REST_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"

    def __init__(
        self,
        api_key: Optional[str] = None,
        enable_cache: bool = True,
    ):
        super().__init__(name="google_cloud_vision", enable_cache=enable_cache)
        self.api_key = api_key or os.getenv("GOOGLE_CLOUD_API_KEY") or settings.gemini_api_key

    def _execute_transcribe(self, image_path: Path, language: str, prompt_hint: Optional[str] = None) -> OCRResult:
        if not self.api_key:
            raise ValueError("Google Cloud Vision API key is not configured.")

        # Read and base64 encode the image
        with open(image_path, "rb") as f:
            content_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "requests": [
                {
                    "image": {"content": content_b64},
                    "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                    "imageContext": {"languageHints": [language]},
                }
            ]
        }

        url = f"{self.REST_ENDPOINT}?key={self.api_key}"
        with httpx.Client(timeout=settings.engine.timeout_seconds) as client:
            response = client.post(url, json=payload)
            if response.status_code != 200:
                raise RuntimeError(f"Google Cloud Vision error ({response.status_code}): {response.text}")
            data = response.json()

        responses = data.get("responses", [{}])
        if not responses or "fullTextAnnotation" not in responses[0]:
            return OCRResult(
                engine_name=self.name,
                image_path=str(image_path),
                language=language,
                raw_text="",
                lines=[],
            )

        full_text = responses[0]["fullTextAnnotation"].get("text", "")
        lines = [LineItem(text=line) for line in full_text.splitlines() if line.strip()]

        return OCRResult(
            engine_name=self.name,
            image_path=str(image_path),
            language=language,
            raw_text=full_text,
            lines=lines,
            metadata={"pages_count": len(responses[0]["fullTextAnnotation"].get("pages", []))},
        )
