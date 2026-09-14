"""
Engines module providing unified interface and factory loader.
"""

from typing import Dict, Type
from .base import BaseOCREngine, LineItem, OCRResult
from .gemini_engine import GeminiEngine
from .sarvam_engine import SarvamEngine
from .gcv_engine import GoogleCloudVisionEngine
from .surya_engine import SuryaEngine
from .tesseract_engine import TesseractEngine
from .mock_engine import MockOCREngine

ENGINE_REGISTRY: Dict[str, Type[BaseOCREngine]] = {
    "gemini": GeminiEngine,
    "sarvam": SarvamEngine,
    "gcv": GoogleCloudVisionEngine,
    "google_cloud_vision": GoogleCloudVisionEngine,
    "surya": SuryaEngine,
    "tesseract": TesseractEngine,
    "mock": MockOCREngine,
}


def get_engine(engine_name: str, **kwargs) -> BaseOCREngine:
    """Factory method to instantiate an engine by name."""
    name_clean = engine_name.lower().strip()
    if name_clean in ENGINE_REGISTRY:
        return ENGINE_REGISTRY[name_clean](**kwargs)
    elif name_clean.startswith("gemini"):
        return GeminiEngine(model_name=engine_name, **kwargs)
    else:
        raise ValueError(f"Unknown engine '{engine_name}'. Available: {list(ENGINE_REGISTRY.keys())}")


__all__ = [
    "BaseOCREngine",
    "LineItem",
    "OCRResult",
    "GeminiEngine",
    "SarvamEngine",
    "GoogleCloudVisionEngine",
    "SuryaEngine",
    "TesseractEngine",
    "MockOCREngine",
    "get_engine",
    "ENGINE_REGISTRY",
]
