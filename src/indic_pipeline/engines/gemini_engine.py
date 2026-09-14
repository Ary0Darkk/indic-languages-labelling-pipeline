"""
Google Gemini Vision Multimodal OCR engine for Indic printed and handwritten text.
"""

import os
from pathlib import Path
from typing import Optional
from PIL import Image
from google import genai
from google.genai import types

from indic_pipeline.config import settings
from indic_pipeline.engines.base import BaseOCREngine, LineItem, OCRResult
from indic_pipeline.utils.logger import logger


class GeminiEngine(BaseOCREngine):
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        enable_cache: bool = True,
    ):
        super().__init__(name=f"gemini_{model_name or settings.engine.gemini_model}", enable_cache=enable_cache)
        self.model_name = model_name or settings.engine.gemini_model
        self.api_key = api_key or settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.temperature = temperature

        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None
            logger.warning("Gemini API key not found. GeminiEngine will require GEMINI_API_KEY to run live queries.")

    def _build_system_instruction(self, lang_name: str, script_name: str) -> str:
        return (
            f"You are an expert OCR and Document Transcription system specialized in Indian languages and scripts, "
            f"specifically {lang_name} ({script_name} script).\n"
            f"The provided image may contain both PRINTED and HANDWRITTEN text, form fields, tables, or freeform handwriting.\n\n"
            f"Transcription Guidelines:\n"
            f"1. Transcribe the document verbatim, preserving all character diacritics, conjuncts (vattakshara/sanyuktakshar), "
            f"vowel modifiers (matras), halants/viramas, and punctuation accurately in Unicode {script_name} script.\n"
            f"2. Faithfully transcribe both printed text and handwritten text in top-to-bottom, left-to-right reading order.\n"
            f"3. Do not omit, summarize, translate, or correct spelling errors from the original image.\n"
            f"4. If there is English or Latin text/numbers present, transcribe them exactly as written.\n"
            f"5. Output ONLY the extracted text. Do not add explanations, conversational greetings, markdown commentary, or backtick wraps unless they are part of the document."
        )

    def _execute_transcribe(self, image_path: Path, language: str, prompt_hint: Optional[str] = None) -> OCRResult:
        if not self.client:
            raise ValueError("Gemini API key is not configured. Set GEMINI_API_KEY environment variable.")

        lang_info = self.get_language_info(language)
        lang_name = lang_info["name"]
        script_name = lang_info["script"]

        system_instruction = self._build_system_instruction(lang_name, script_name)

        user_prompt = (
            f"Please transcribe all printed and handwritten {lang_name} text from this document image accurately."
        )
        if prompt_hint:
            user_prompt += f"\nAdditional Context/Hint: {prompt_hint}"

        # Open image with PIL
        pil_img = Image.open(image_path)

        # Call Gemini API
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=self.temperature,
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[pil_img, user_prompt],
            config=config,
        )

        raw_text = response.text.strip() if response.text else ""

        # Break into lines
        lines = [LineItem(text=line) for line in raw_text.splitlines() if line.strip()]

        return OCRResult(
            engine_name=self.name,
            image_path=str(image_path),
            language=language,
            raw_text=raw_text,
            lines=lines,
            metadata={
                "model": self.model_name,
                "temperature": self.temperature,
            },
        )
