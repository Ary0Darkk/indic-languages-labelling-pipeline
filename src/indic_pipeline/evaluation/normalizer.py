"""
Indic Unicode normalization and text cleaning.
"""

import re
import unicodedata
from typing import Optional
from indic_pipeline.config import EvaluationConfig, settings


class IndicNormalizer:
    # Indic Danda and double Danda
    DANDA = "\u0964"
    DOUBLE_DANDA = "\u0965"

    # Zero Width Joiners / Non-Joiners
    ZWJ = "\u200D"
    ZWNJ = "\u200C"

    # Indic digit mappings to standard ASCII digits
    INDIC_DIGITS_MAP = {
        # Devanagari (Hindi, Marathi, Sanskrit)
        0x0966: "0", 0x0967: "1", 0x0968: "2", 0x0969: "3", 0x096A: "4",
        0x096B: "5", 0x096C: "6", 0x096D: "7", 0x096E: "8", 0x096F: "9",
        # Bengali / Assamese
        0x09E6: "0", 0x09E7: "1", 0x09E8: "2", 0x09E9: "3", 0x09EA: "4",
        0x09EB: "5", 0x09EC: "6", 0x09ED: "7", 0x09EE: "8", 0x09EF: "9",
        # Gurmukhi (Punjabi)
        0x0A66: "0", 0x0A67: "1", 0x0A68: "2", 0x0A69: "3", 0x0A6A: "4",
        0x0A6B: "5", 0x0A6C: "6", 0x0A6D: "7", 0x0A6E: "8", 0x0A6F: "9",
        # Gujarati
        0x0AE6: "0", 0x0AE7: "1", 0x0AE8: "2", 0x0AE9: "3", 0x0AEA: "4",
        0x0AEB: "5", 0x0AEC: "6", 0x0AED: "7", 0x0AEE: "8", 0x0AEF: "9",
        # Odia
        0x0B66: "0", 0x0B67: "1", 0x0B68: "2", 0x0B69: "3", 0x0B6A: "4",
        0x0B6B: "5", 0x0B6C: "6", 0x0B6D: "7", 0x0B6E: "8", 0x0B6F: "9",
        # Tamil
        0x0BE6: "0", 0x0BE7: "1", 0x0BE8: "2", 0x0BE9: "3", 0x0BEA: "4",
        0x0BEB: "5", 0x0BEC: "6", 0x0BED: "7", 0x0BEE: "8", 0x0BEF: "9",
        # Telugu
        0x0C66: "0", 0x0C67: "1", 0x0C68: "2", 0x0C69: "3", 0x0C6A: "4",
        0x0C6B: "5", 0x0C6C: "6", 0x0C6D: "7", 0x0C6E: "8", 0x0C6F: "9",
        # Kannada
        0x0CE6: "0", 0x0CE7: "1", 0x0CE8: "2", 0x0CE9: "3", 0x0CEA: "4",
        0x0CEB: "5", 0x0CEC: "6", 0x0CED: "7", 0x0CEE: "8", 0x0CEF: "9",
        # Malayalam
        0x0D66: "0", 0x0D67: "1", 0x0D68: "2", 0x0D69: "3", 0x0D6A: "4",
        0x0D6B: "5", 0x0D6C: "6", 0x0D6D: "7", 0x0D6E: "8", 0x0D6F: "9",
    }

    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or settings.evaluation

    def normalize_digits(self, text: str) -> str:
        """Converts Indic script digits to standard ASCII 0-9 digits."""
        return text.translate(self.INDIC_DIGITS_MAP)

    def normalize(self, text: Optional[str], convert_digits: bool = False) -> str:
        """
        Full Indic normalization pipeline.
        Ensures fair comparison for OCR/HTR character metrics.
        """
        if not text:
            return ""

        # 1. Unicode Canonical Decomposition/Composition (NFC by default)
        if self.config.normalize_unicode:
            text = unicodedata.normalize(self.config.unicode_form, text)

        # 2. Handle Zero-Width Non-Joiner (ZWNJ) and Joiner (ZWJ)
        if self.config.ignore_zwnj:
            text = text.replace(self.ZWNJ, "")
        if self.config.ignore_zwj:
            text = text.replace(self.ZWJ, "")

        # 3. Optional: normalize Indic digits to ASCII digits
        if convert_digits:
            text = self.normalize_digits(text)

        # 4. Standardize quotes and dashes
        text = text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
        text = text.replace("—", "-").replace("–", "-")

        # 5. Punctuation stripping if configured
        if self.config.strip_punctuation:
            text = re.sub(r"[^\w\s]", "", text)

        # 6. Normalize whitespace (tabs, consecutive spaces, newlines)
        if self.config.normalize_whitespace:
            # Replace non-breaking space with normal space
            text = text.replace("\u00A0", " ").replace("\u200B", "")
            # Collapse multiple spaces on a single line
            lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
            # Filter empty lines if desired or keep clean newlines
            text = "\n".join(line for line in lines if line)

        return text.strip()


default_normalizer = IndicNormalizer()
