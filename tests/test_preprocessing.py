"""
Unit tests for image enhancement and deskewing.
"""

import numpy as np
from indic_pipeline.config import PreprocessingConfig
from indic_pipeline.preprocessing import ImageEnhancer, DocumentDeskewer, PreprocessingPipeline


def test_image_enhancer_clahe_and_resolution():
    config = PreprocessingConfig(enabled=True, clahe_contrast=True, min_dpi=300)
    enhancer = ImageEnhancer(config)

    # Create dummy low-contrast image (800x600)
    dummy_img = np.full((600, 800, 3), 180, dtype=np.uint8)
    dummy_img[200:400, 200:600] = 120  # Simulated text block

    enhanced = enhancer.enhance(dummy_img)

    # Check that image was upscaled to meet min dimension
    assert min(enhanced.shape[:2]) >= 1500
    assert enhanced.shape[2] == 3


def test_document_deskewer():
    deskewer = DocumentDeskewer()
    # Create an image with horizontal text lines
    img = np.full((1000, 1000, 3), 255, dtype=np.uint8)
    for y in range(200, 800, 60):
        img[y:y+10, 200:800] = 0

    # Rotate by 5 degrees
    rotated = deskewer.rotate_image(img, 5.0)
    assert rotated.shape[0] >= 1000
    assert rotated.shape[1] >= 1000


def test_preprocessing_pipeline():
    pipeline = PreprocessingPipeline()
    dummy_img = np.full((800, 800, 3), 240, dtype=np.uint8)
    processed, meta = pipeline.process(dummy_img)
    assert processed is not None
    assert "deskew_angle" in meta
