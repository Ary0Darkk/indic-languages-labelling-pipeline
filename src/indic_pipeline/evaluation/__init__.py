"""
Evaluation, normalization, and metrics for Indic OCR.
"""

from .normalizer import IndicNormalizer
from .metrics import EvaluationMetrics, calculate_cer, calculate_wer, evaluate_prediction
from .reporter import BenchmarkReporter

__all__ = [
    "IndicNormalizer",
    "EvaluationMetrics",
    "calculate_cer",
    "calculate_wer",
    "evaluate_prediction",
    "BenchmarkReporter",
]
