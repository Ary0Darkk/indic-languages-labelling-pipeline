"""
Unit tests for BenchmarkRunner and BenchmarkReporter.
"""

from pathlib import Path
import cv2
import numpy as np

from indic_pipeline.dataset.loader import DocumentSample
from indic_pipeline.engines.mock_engine import MockOCREngine
from indic_pipeline.evaluation.benchmark import BenchmarkRunner
from indic_pipeline.evaluation.reporter import BenchmarkReporter


def test_benchmark_runner_and_reports(tmp_path: Path):
    # Setup dummy sample
    img_path = tmp_path / "test_sample.png"
    cv2.imwrite(str(img_path), np.full((300, 300, 3), 255, dtype=np.uint8))

    sample = DocumentSample(
        image_path=img_path,
        ground_truth_text="ಕರ್ನಾಟಕ ರಾಜ್ಯ ಸರ್ಕಾರ ಅಧಿಕೃತ ದಾಖಲೆ",
        language="kn",
        sample_id="test_sample",
    )

    out_dir = tmp_path / "outputs"
    reporter = BenchmarkReporter(output_dir=out_dir)

    engines = [
        MockOCREngine(name="mock_vlm", simulated_cer=0.03, simulated_latency=0.01),
        MockOCREngine(name="mock_ocr", simulated_cer=0.15, simulated_latency=0.01),
    ]

    runner = BenchmarkRunner(engines=engines, language="kn", reporter=reporter)
    summaries = runner.run([sample], apply_preprocessing=False, output_prefix="test_run")

    assert "mock_vlm" in summaries
    assert "mock_ocr" in summaries
    assert summaries["mock_vlm"]["mean_cer"] < summaries["mock_ocr"]["mean_cer"]

    # Verify generated report files
    assert (out_dir / "test_run_per_page.csv").exists()
    assert (out_dir / "test_run_summary.json").exists()
    assert (out_dir / "test_run_report.html").exists()
