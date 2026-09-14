"""
Dataset loader for pairing document images with reference ground truth.
"""

from pathlib import Path
from typing import List, Optional
import pandas as pd

from indic_pipeline.dataset.parser import DocumentSample, GroundTruthParser
from indic_pipeline.preprocessing.enhancer import ImageEnhancer
from indic_pipeline.utils.logger import logger

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".pdf"}


class DatasetLoader:
    def __init__(self, data_dir: Path, language: str = "kn"):
        self.data_dir = Path(data_dir)
        self.language = language

    def load_samples(
        self,
        max_samples: Optional[int] = None,
        gt_dir: Optional[Path] = None,
    ) -> List[DocumentSample]:
        """
        Loads document samples by discovering image and PDF files and pairing them
        with their corresponding ground truth files (.json or .txt).
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")

        samples = []

        # Find all raw files
        raw_files = sorted([
            p for p in self.data_dir.glob("**/*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ])

        image_files = []
        for rf in raw_files:
            if rf.suffix.lower() == ".pdf":
                extracted = ImageEnhancer.extract_pdf_pages(rf)
                image_files.extend(extracted)
            else:
                image_files.append(rf)

        if not image_files:
            logger.warning(f"No image files found in {self.data_dir}")
            return []

        for img_path in image_files:
            sample_id = img_path.stem
            gt_text = None

            # Strategy 1: Look for exact same name with .json or .txt in same folder
            json_same = img_path.with_suffix(".json")
            txt_same = img_path.with_suffix(".txt")

            if json_same.exists():
                gt_text = GroundTruthParser.parse_from_file(json_same)
            elif txt_same.exists():
                gt_text = GroundTruthParser.parse_from_file(txt_same)
            elif gt_dir and gt_dir.exists():
                # Strategy 2: Look in dedicated ground_truth directory
                json_gt = gt_dir / f"{sample_id}.json"
                txt_gt = gt_dir / f"{sample_id}.txt"
                if json_gt.exists():
                    gt_text = GroundTruthParser.parse_from_file(json_gt)
                elif txt_gt.exists():
                    gt_text = GroundTruthParser.parse_from_file(txt_gt)

            if gt_text is not None:
                samples.append(
                    DocumentSample(
                        image_path=img_path,
                        ground_truth_text=gt_text,
                        language=self.language,
                        sample_id=sample_id,
                    )
                )
            else:
                logger.debug(f"Skipping {img_path.name}: No matching .json/.txt ground truth found.")

            if max_samples and len(samples) >= max_samples:
                break

        logger.info(f"Loaded {len(samples)} paired image-groundtruth samples from {self.data_dir}")
        return samples

    def load_from_csv(self, csv_path: Path, image_col: str = "image_path", gt_col: str = "ground_truth") -> List[DocumentSample]:
        """Loads dataset from a CSV index file."""
        df = pd.read_csv(csv_path)
        samples = []
        for idx, row in df.iterrows():
            img_p = Path(row[image_col])
            if not img_p.is_absolute():
                img_p = self.data_dir / img_p

            if img_p.exists():
                samples.append(
                    DocumentSample(
                        image_path=img_p,
                        ground_truth_text=str(row[gt_col]),
                        language=self.language,
                        sample_id=img_p.stem,
                    )
                )
        return samples
