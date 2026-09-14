"""
Unit tests for GroundTruthParser and DatasetLoader.
"""

from pathlib import Path
import json
import pytest
from indic_pipeline.dataset.parser import GroundTruthParser
from indic_pipeline.dataset.loader import DatasetLoader


def test_parse_json_direct_string(tmp_path: Path):
    json_path = tmp_path / "sample.json"
    json_path.write_text(json.dumps({"text": "ಕರ್ನಾಟಕ ರಾಜ್ಯ"}), encoding="utf-8")

    text = GroundTruthParser.parse_from_file(json_path)
    assert text == "ಕರ್ನಾಟಕ ರಾಜ್ಯ"


def test_parse_json_nested_lines(tmp_path: Path):
    json_path = tmp_path / "sample_lines.json"
    data = {
        "lines": [
            {"text": "ಮೊದಲ ಸಾಲು"},
            {"text": "ಎರಡನೇ ಸಾಲು"}
        ]
    }
    json_path.write_text(json.dumps(data), encoding="utf-8")

    text = GroundTruthParser.parse_from_file(json_path)
    assert "ಮೊದಲ ಸಾಲು" in text
    assert "ಎರಡನೇ ಸಾಲು" in text


def test_dataset_loader(tmp_path: Path):
    import cv2
    import numpy as np

    # Create dummy image and json
    img_path = tmp_path / "kn_001.png"
    json_path = tmp_path / "kn_001.json"

    dummy_img = np.full((300, 300, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(img_path), dummy_img)
    json_path.write_text(json.dumps({"transcription": "ಪರೀಕ್ಷಾ ದಾಖಲೆ"}), encoding="utf-8")

    loader = DatasetLoader(data_dir=tmp_path, language="kn")
    samples = loader.load_samples()

    assert len(samples) == 1
    assert samples[0].sample_id == "kn_001"
    assert samples[0].ground_truth_text == "ಪರೀಕ್ಷಾ ದಾಖಲೆ"
