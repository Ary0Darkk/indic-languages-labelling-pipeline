"""
Mock OCR engine for offline testing, dry-runs, and simulation.
"""

from pathlib import Path
import random
import time
from typing import Optional

from indic_pipeline.engines.base import BaseOCREngine, LineItem, OCRResult


class MockOCREngine(BaseOCREngine):
    def __init__(
        self,
        name: str = "mock_engine",
        simulated_cer: float = 0.05,
        simulated_latency: float = 0.4,
        enable_cache: bool = False,
    ):
        super().__init__(name=name, enable_cache=enable_cache)
        self.simulated_cer = simulated_cer
        self.simulated_latency = simulated_latency

    def _execute_transcribe(self, image_path: Path, language: str, prompt_hint: Optional[str] = None) -> OCRResult:
        time.sleep(self.simulated_latency)

        # Check if corresponding ground truth exists to create realistic noisy prediction
        gt_file_json = image_path.with_suffix(".json")
        gt_file_txt = image_path.with_suffix(".txt")

        base_text = "ಕರ್ನಾಟಕ ರಾಜ್ಯ ಸರ್ಕಾರ ಅಧಿಕೃತ ದಾಖಲೆ\nಹೆಸರು: ರಮೇಶ್ ಕುಮಾರ್\nದಿನಾಂಕ: 15-08-2024"
        if gt_file_txt.exists():
            base_text = gt_file_txt.read_text(encoding="utf-8")

        # Inject synthetic noise based on simulated_cer
        chars = list(base_text)
        num_errors = int(len(chars) * self.simulated_cer)
        for _ in range(num_errors):
            idx = random.randint(0, len(chars) - 1)
            if chars[idx] not in ("\n", " "):
                chars[idx] = random.choice(["ಅ", "ಆ", "ಇ", "ಕ", "ಗ", "ತ", "ನ", "ರ", "ಲ"])

        pred_text = "".join(chars)
        lines = [LineItem(text=line) for line in pred_text.splitlines() if line.strip()]

        return OCRResult(
            engine_name=self.name,
            image_path=str(image_path),
            language=language,
            raw_text=pred_text,
            lines=lines,
            metadata={"simulated_cer": self.simulated_cer},
        )
