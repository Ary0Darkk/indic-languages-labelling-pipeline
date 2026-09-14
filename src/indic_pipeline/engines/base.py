"""
Abstract Base Class and data models for OCR/HTR/VLM engines.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from indic_pipeline.config import INDIC_LANGUAGES, settings
from indic_pipeline.utils.cache import disk_cache
from indic_pipeline.utils.logger import logger


class LineItem(BaseModel):
    text: str
    confidence: Optional[float] = None
    bounding_box: Optional[List[float]] = None  # [ymin, xmin, ymax, xmax] or [x, y, w, h]
    text_type: Optional[str] = None  # "printed", "handwritten", "unknown"


class OCRResult(BaseModel):
    engine_name: str
    image_path: str
    language: str
    raw_text: str
    lines: List[LineItem] = Field(default_factory=list)
    latency_seconds: float = 0.0
    confidence: Optional[float] = None
    estimated_cost_usd: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    cached: bool = False


class BaseOCREngine(ABC):
    def __init__(self, name: str, enable_cache: bool = True):
        self.name = name
        self.enable_cache = enable_cache

    def get_language_info(self, lang_code: str) -> Dict[str, str]:
        """Returns language name and script for the given ISO code."""
        return INDIC_LANGUAGES.get(lang_code.lower(), {"name": lang_code, "script": "Indic", "code": lang_code})

    @abstractmethod
    def _execute_transcribe(self, image_path: Path, language: str, prompt_hint: Optional[str] = None) -> OCRResult:
        """Internal engine execution method to be overridden by subclasses."""
        pass

    def transcribe(
        self,
        image_path: Path,
        language: str = "kn",
        prompt_hint: Optional[str] = None,
        bypass_cache: bool = False,
    ) -> OCRResult:
        """
        Transcribes an image with disk caching, latency measurement, and error handling.
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Check Cache
        if self.enable_cache and not bypass_cache:
            cache_params = {"language": language, "prompt_hint": prompt_hint}
            cached_data = disk_cache.get(image_path, self.name, cache_params)
            if cached_data:
                result = OCRResult(**cached_data)
                result.cached = True
                return result

        # Execute OCR with timing
        start_time = time.perf_counter()
        try:
            result = self._execute_transcribe(image_path, language, prompt_hint)
            result.latency_seconds = round(time.perf_counter() - start_time, 3)
        except Exception as e:
            logger.error(f"Engine {self.name} failed on {image_path.name}: {e}")
            raise e

        # Store in cache
        if self.enable_cache:
            cache_params = {"language": language, "prompt_hint": prompt_hint}
            disk_cache.set(image_path, self.name, result.model_dump(), cache_params)

        return result

    def transcribe_batch(
        self,
        image_paths: List[Path],
        language: str = "kn",
        prompt_hint: Optional[str] = None,
        bypass_cache: bool = False,
    ) -> List[OCRResult]:
        """Transcribes a batch of images sequentially or using a pool."""
        results = []
        for img_path in image_paths:
            try:
                res = self.transcribe(img_path, language=language, prompt_hint=prompt_hint, bypass_cache=bypass_cache)
                results.append(res)
            except Exception as e:
                logger.error(f"Batch item failed for {img_path}: {e}")
                results.append(
                    OCRResult(
                        engine_name=self.name,
                        image_path=str(img_path),
                        language=language,
                        raw_text="",
                        metadata={"error": str(e)},
                    )
                )
        return results
