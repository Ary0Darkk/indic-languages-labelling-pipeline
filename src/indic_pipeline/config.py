"""
Configuration management for Indic OCR & HTR pipeline.
"""

from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


INDIC_LANGUAGES: Dict[str, Dict[str, str]] = {
    "kn": {"name": "Kannada", "script": "Kannada", "code": "kn"},
    "hi": {"name": "Hindi", "script": "Devanagari", "code": "hi"},
    "bn": {"name": "Bengali", "script": "Bengali", "code": "bn"},
    "ta": {"name": "Tamil", "script": "Tamil", "code": "ta"},
    "te": {"name": "Telugu", "script": "Telugu", "code": "te"},
    "mr": {"name": "Marathi", "script": "Devanagari", "code": "mr"},
    "gu": {"name": "Gujarati", "script": "Gujarati", "code": "gu"},
    "ml": {"name": "Malayalam", "script": "Malayalam", "code": "ml"},
    "or": {"name": "Odia", "script": "Odia", "code": "or"},
    "pa": {"name": "Punjabi", "script": "Gurmukhi", "code": "pa"},
    "as": {"name": "Assamese", "script": "Bengali/Assamese", "code": "as"},
    "ur": {"name": "Urdu", "script": "Perso-Arabic", "code": "ur"},
    "sa": {"name": "Sanskrit", "script": "Devanagari", "code": "sa"},
}


class PreprocessingConfig(BaseModel):
    enabled: bool = True
    deskew: bool = True
    clahe_contrast: bool = True
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8
    denoise: bool = True
    adaptive_binarize: bool = False
    min_dpi: int = 300
    target_dpi: int = 300


class EngineConfig(BaseModel):
    gemini_model: str = "gemini-2.5-flash"
    gemini_temperature: float = 0.0
    sarvam_model: str = "sarvam-ocr"
    gcv_features: List[str] = Field(default_factory=lambda: ["DOCUMENT_TEXT_DETECTION"])
    tesseract_cmd: Optional[str] = None
    timeout_seconds: int = 60
    max_retries: int = 3
    retry_delay: float = 2.0


class EvaluationConfig(BaseModel):
    normalize_unicode: bool = True
    unicode_form: str = "NFC"  # NFC or NFD
    ignore_zwnj: bool = True   # Ignore \u200C
    ignore_zwj: bool = True    # Ignore \u200D
    ignore_case: bool = False  # Not relevant for Indic scripts, but for mixed English
    strip_punctuation: bool = False
    normalize_whitespace: bool = True
    normalize_nukta: bool = True


class PipelineSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # API Keys
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    sarvam_api_key: Optional[str] = Field(default=None, alias="SARVAM_API_KEY")
    google_application_credentials: Optional[str] = Field(default=None, alias="GOOGLE_APPLICATION_CREDENTIALS")

    # Directory Paths
    data_dir: Path = Path("./data")
    cache_dir: Path = Path("./data/cache")
    output_dir: Path = Path("./data/outputs")

    # Sub-configs
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


settings = PipelineSettings()
