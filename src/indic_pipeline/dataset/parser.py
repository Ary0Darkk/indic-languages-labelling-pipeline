"""
Ground truth parser supporting multiple JSON formats, TXT files, and tabular annotations.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from indic_pipeline.utils.logger import logger


class DocumentSample(BaseModel):
    image_path: Path
    ground_truth_text: str
    language: str = "kn"
    sample_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GroundTruthParser:
    """Parses ground truth reference text from varied formats."""

    @staticmethod
    def parse_from_file(gt_path: Path) -> str:
        """Extracts plain text ground truth from a JSON or TXT file."""
        gt_path = Path(gt_path)
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

        if gt_path.suffix.lower() == ".json":
            try:
                with open(gt_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return GroundTruthParser.extract_text_from_json(data)
            except Exception as e:
                logger.error(f"Failed to parse JSON ground truth {gt_path}: {e}")
                raise e
        else:
            # Assume text file
            with open(gt_path, "r", encoding="utf-8") as f:
                return f.read().strip()

    @staticmethod
    def extract_text_from_json(data: Any) -> str:
        """
        Extracts full text from multiple possible JSON structures:
        - String direct
        - Dict with 'text', 'transcription', 'ground_truth', 'transcript', 'content', 'label'
        - Dict with list of lines/annotations: 'lines', 'regions', 'boxes', 'annotations', 'items'
        - List of strings or dicts
        """
        if isinstance(data, str):
            return data.strip()

        if isinstance(data, list):
            parts = []
            for item in data:
                extracted = GroundTruthParser.extract_text_from_json(item)
                if extracted:
                    parts.append(extracted)
            return "\n".join(parts)

        if isinstance(data, dict):
            # Check direct top-level text fields
            for key in ["text", "transcript", "transcription", "ground_truth", "gt", "content", "label", "full_text"]:
                if key in data and isinstance(data[key], str) and data[key].strip():
                    return data[key].strip()

            # Check nested line/annotation lists
            for key in ["lines", "annotations", "regions", "boxes", "items", "paragraphs", "sentences"]:
                if key in data and isinstance(data[key], list):
                    parts = []
                    for item in data[key]:
                        extracted = GroundTruthParser.extract_text_from_json(item)
                        if extracted:
                            parts.append(extracted)
                    if parts:
                        return "\n".join(parts)

            # Fallback: if dict has values that are strings, join them
            str_values = [v.strip() for v in data.values() if isinstance(v, str) and v.strip()]
            if str_values:
                return "\n".join(str_values)

        return ""
