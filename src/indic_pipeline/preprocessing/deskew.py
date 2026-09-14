"""
Document deskewing and orientation correction.
"""

from typing import Tuple
import cv2
import numpy as np
from indic_pipeline.utils.logger import logger


class DocumentDeskewer:
    def __init__(self, max_angle: float = 45.0):
        self.max_angle = max_angle

    def estimate_skew_angle(self, img_bgr: np.ndarray) -> float:
        """
        Estimates skew angle of text lines using Hough Line Transform
        and contour bounding boxes.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Invert and threshold to get text pixels
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Use morphological operations to connect horizontal text line components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        # 1. Hough lines method
        edges = cv2.Canny(dilated, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=20
        )

        angles = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.ravel()[:4]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) <= self.max_angle:
                    angles.append(angle)

        if angles:
            median_angle = float(np.median(angles))
            if abs(median_angle) > 0.3:
                return median_angle

        # 2. Fallback: minAreaRect on non-zero pixels
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) > 50:
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            if abs(angle) <= self.max_angle and abs(angle) > 0.3:
                return float(angle)

        return 0.0

    def rotate_image(self, img_bgr: np.ndarray, angle: float) -> np.ndarray:
        """Rotates image by the given angle (in degrees) with white background padding."""
        if abs(angle) < 0.2:
            return img_bgr

        (h, w) = img_bgr.shape[:2]
        center = (w // 2, h // 2)

        # Compute rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # Compute new bounding dimensions
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # Adjust matrix to center
        M[0, 2] += (nW / 2) - center[0]
        M[1, 2] += (nH / 2) - center[1]

        # Rotate with white border fill (standard for document paper)
        rotated = cv2.warpAffine(
            img_bgr,
            M,
            (nW, nH),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        return rotated

    def deskew(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        """Deskews the image and returns (deskewed_image, angle_applied)."""
        angle = self.estimate_skew_angle(img_bgr)
        if abs(angle) >= 0.2:
            logger.debug(f"Detected skew angle: {angle:.2f} degrees. Correcting...")
            corrected = self.rotate_image(img_bgr, angle)
            return corrected, angle
        return img_bgr, 0.0
