"""
Composable preprocessing pipeline for document images.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np
from indic_pipeline.config import PreprocessingConfig, settings
from indic_pipeline.preprocessing.enhancer import ImageEnhancer
from indic_pipeline.preprocessing.deskew import DocumentDeskewer
from indic_pipeline.utils.logger import logger


class PreprocessingPipeline:
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or settings.preprocessing
        self.enhancer = ImageEnhancer(self.config)
        self.deskewer = DocumentDeskewer()

    def process(
        self,
        image_input: Union[str, Path, np.ndarray],
        output_path: Optional[Path] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Executes the configured preprocessing steps on the image.
        Returns the processed image and metadata regarding applied transformations.
        """
        img = self.enhancer.load_image(image_input)
        metadata = {"original_shape": img.shape, "deskew_angle": 0.0}

        if not self.config.enabled:
            return img, metadata

        # 1. Deskew
        if self.config.deskew:
            img, angle = self.deskewer.deskew(img)
            metadata["deskew_angle"] = angle

        # 2. Enhance (illumination, contrast, denoise, resolution)
        img = self.enhancer.enhance(img)
        metadata["processed_shape"] = img.shape

        # 3. Save if requested
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), img)
            logger.debug(f"Saved preprocessed image to {output_path}")

        return img, metadata

    def process_to_temp_file(self, image_input: Union[str, Path, np.ndarray], cache_dir: Optional[Path] = None) -> Path:
        """Processes image and writes to a deterministic temporary/cache file, returning its path."""
        target_dir = cache_dir or settings.cache_dir / "preprocessed"
        target_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            out_file = target_dir / f"prep_{p.stem}.png"
        else:
            out_file = target_dir / "prep_temp.png"

        img, _ = self.process(image_input, output_path=out_file)
        return out_file
