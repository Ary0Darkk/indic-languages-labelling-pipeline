"""
Multi-engine benchmark runner for Indic OCR and HTR.
"""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm

from indic_pipeline.config import INDIC_LANGUAGES, settings
from indic_pipeline.dataset.loader import DatasetLoader, DocumentSample
from indic_pipeline.engines import BaseOCREngine, get_engine
from indic_pipeline.evaluation.metrics import evaluate_prediction
from indic_pipeline.evaluation.normalizer import IndicNormalizer
from indic_pipeline.evaluation.reporter import BenchmarkReporter
from indic_pipeline.preprocessing.pipeline import PreprocessingPipeline
from indic_pipeline.utils.logger import logger


class BenchmarkRunner:
    def __init__(
        self,
        engines: List[BaseOCREngine],
        language: str = "kn",
        normalizer: Optional[IndicNormalizer] = None,
        preprocessor: Optional[PreprocessingPipeline] = None,
        reporter: Optional[BenchmarkReporter] = None,
    ):
        self.engines = engines
        self.language = language
        self.normalizer = normalizer or IndicNormalizer()
        self.preprocessor = preprocessor or PreprocessingPipeline()
        self.reporter = reporter or BenchmarkReporter()

    def run(
        self,
        samples: List[DocumentSample],
        apply_preprocessing: bool = True,
        output_prefix: str = "benchmark",
    ) -> Dict[str, dict]:
        """
        Runs benchmarking across all configured engines and samples.
        """
        if not samples:
            raise ValueError("No samples provided for benchmarking.")

        page_results = []
        engine_stats: Dict[str, dict] = {
            eng.name: {
                "cers": [],
                "wers": [],
                "char_accs": [],
                "word_accs": [],
                "latencies": [],
                "total_subs": 0,
                "total_dels": 0,
                "total_ins": 0,
                "total_pages": 0,
            }
            for eng in self.engines
        }

        logger.info(f"Starting benchmark on {len(samples)} pages across {len(self.engines)} engine(s)...")

        for sample in tqdm(samples, desc="Benchmarking Pages"):
            # Preprocessing
            img_to_ocr = sample.image_path
            if apply_preprocessing:
                try:
                    img_to_ocr = self.preprocessor.process_to_temp_file(sample.image_path)
                except Exception as e:
                    logger.warning(f"Preprocessing failed for {sample.image_path.name}, using raw: {e}")
                    img_to_ocr = sample.image_path

            for eng in self.engines:
                try:
                    res = eng.transcribe(img_to_ocr, language=self.language)
                    pred_text = res.raw_text
                    latency = res.latency_seconds
                except Exception as e:
                    logger.error(f"Engine {eng.name} failed on {sample.sample_id}: {e}")
                    pred_text = ""
                    latency = 0.0

                metrics = evaluate_prediction(
                    reference=sample.ground_truth_text,
                    hypothesis=pred_text,
                    normalizer=self.normalizer,
                )

                # Record per-page result
                page_results.append({
                    "sample_id": sample.sample_id,
                    "image_name": sample.image_path.name,
                    "engine_name": eng.name,
                    "ground_truth": sample.ground_truth_text,
                    "ground_truth_len": len(sample.ground_truth_text),
                    "prediction": pred_text,
                    "prediction_len": len(pred_text),
                    "metrics": metrics,
                    "latency": latency,
                })

                # Update stats
                st = engine_stats[eng.name]
                st["cers"].append(metrics.cer)
                st["wers"].append(metrics.wer)
                st["char_accs"].append(metrics.char_accuracy)
                st["word_accs"].append(metrics.word_accuracy)
                st["latencies"].append(latency)
                st["total_subs"] += metrics.char_edits.substitutions
                st["total_dels"] += metrics.char_edits.deletions
                st["total_ins"] += metrics.char_edits.insertions
                st["total_pages"] += 1

        # Compute summary averages
        engine_summaries = {}
        for eng_name, st in engine_stats.items():
            if st["total_pages"] > 0:
                engine_summaries[eng_name] = {
                    "mean_cer": float(np.mean(st["cers"])),
                    "median_cer": float(np.median(st["cers"])),
                    "mean_wer": float(np.mean(st["wers"])),
                    "median_wer": float(np.median(st["wers"])),
                    "mean_char_acc": float(np.mean(st["char_accs"])),
                    "mean_word_acc": float(np.mean(st["word_accs"])),
                    "mean_latency": float(np.mean(st["latencies"])),
                    "total_subs": st["total_subs"],
                    "total_dels": st["total_dels"],
                    "total_ins": st["total_ins"],
                    "total_pages": st["total_pages"],
                }

        # Print console table
        self.reporter.print_summary_table(engine_summaries, language=self.language)

        # Export CSV and JSON
        self.reporter.export_csv_and_json(page_results, engine_summaries, prefix=output_prefix)

        # Generate HTML report
        lang_info = INDIC_LANGUAGES.get(self.language.lower(), {"name": self.language})
        self.reporter.generate_html_report(
            page_results=page_results,
            engine_summaries=engine_summaries,
            language_code=self.language,
            language_name=lang_info["name"],
            output_filename=f"{output_prefix}_report.html",
        )

        return engine_summaries
