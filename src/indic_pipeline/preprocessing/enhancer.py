"""
Image enhancement techniques tailored for Indic documents (printed & handwritten).
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
from indic_pipeline.config import PreprocessingConfig
from indic_pipeline.utils.logger import logger


class ImageEnhancer:
    def __init__(self, config: PreprocessingConfig):
        self.config = config

    def load_image(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """Loads an image or first page of PDF into a standard BGR numpy array."""
        if isinstance(image_input, (str, Path)):
            path_str = str(image_input)
            if path_str.lower().endswith(".pdf"):
                pages = self.extract_pdf_pages(Path(path_str))
                if pages:
                    return self.load_image(pages[0])
                raise ValueError(f"No renderable pages found in PDF: {path_str}")

            img = cv2.imread(path_str)
            if img is None:
                # Try opening with PIL for wider format support (e.g. TIFF/WebP)
                pil_img = Image.open(path_str).convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return img
        elif isinstance(image_input, Image.Image):
            return cv2.cvtColor(np.array(image_input.convert("RGB")), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            return image_input.copy()
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

    @staticmethod
    def extract_pdf_pages(pdf_path: Path, output_dir: Optional[Path] = None, scale: float = 3.0) -> List[Path]:
        """Extracts and renders all pages of a PDF into high-res PNG images."""
        import pypdfium2 as pdfium

        pdf_path = Path(pdf_path)
        target_dir = output_dir or (pdf_path.parent / f"{pdf_path.stem}_pages")
        target_dir.mkdir(parents=True, exist_ok=True)

        pdf = pdfium.PdfDocument(str(pdf_path))
        page_paths = []

        for i, page in enumerate(pdf):
            # scale=3.0 renders at ~216 DPI, scale=4.0 renders at ~288 DPI
            pil_image = page.render(scale=scale).to_pil()
            page_file = target_dir / f"{pdf_path.stem}_p{i+1:03d}.png"
            pil_image.save(page_file, "PNG")
            page_paths.append(page_file)

        return page_paths

    def apply_clahe(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
        on the Luminance channel (LAB color space) to enhance faded ink
        without blowing out the background.
        """
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=(self.config.clahe_grid_size, self.config.clahe_grid_size),
        )
        cl = clahe.apply(l)

        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def remove_shadows_and_uneven_illumination(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Removes background shadows and uneven page lighting
        using morphological dilation and background division.
        """
        rgb_planes = cv2.split(img_bgr)
        result_planes = []

        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(
                diff_img,
                None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_8UC1,
            )
            result_planes.append(norm_img)

        return cv2.merge(result_planes)

    def denoise(self, img_bgr: np.ndarray) -> np.ndarray:
        """Denoise while preserving sharp stroke edges for Indic character loops."""
        return cv2.fastNlMeansDenoisingColored(
            img_bgr, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
        )

    def adaptive_binarize(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Converts to grayscale and applies adaptive Gaussian thresholding.
        Ideal for older paper with ink bleed or texture.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        binarized = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=8,
        )
        return cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)

    def ensure_resolution(self, img_bgr: np.ndarray, min_dimension: int = 1500) -> np.ndarray:
        """
        Upscales small images so that intricate Indic conjuncts and matras
        remain clear and distinct during OCR recognition.
        """
        h, w = img_bgr.shape[:2]
        if min(h, w) < min_dimension:
            scale = min_dimension / min(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            logger.debug(f"Upscaling document from {w}x{h} to {new_w}x{new_h} (scale={scale:.2f})")
            return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        return img_bgr

    def enhance(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
        """Runs the full enhancement pipeline according to configuration."""
        img = self.load_image(image_input)

        if not self.config.enabled:
            return img

        # 1. Ensure resolution for clear Indic character strokes
        img = self.ensure_resolution(img)

        # 2. Illumination correction / shadow removal
        img = self.remove_shadows_and_uneven_illumination(img)

        # 3. CLAHE contrast enhancement
        if self.config.clahe_contrast:
            img = self.apply_clahe(img)

        # 4. Denoise
        if self.config.denoise:
            # Quick bilateral filter for fast edge-preserving denoising
            img = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)

        # 5. Optional binarization
        if self.config.adaptive_binarize:
            img = self.adaptive_binarize(img)

        return img
