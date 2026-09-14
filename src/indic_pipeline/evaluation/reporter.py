"""
Benchmark reporting and visualization (Console, CSV/JSON, and interactive HTML visual diffs).
"""

import html
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from rich.console import Console
from rich.table import Table
from jinja2 import Template

from indic_pipeline.evaluation.metrics import EvaluationMetrics
from indic_pipeline.utils.logger import logger

console = Console(force_terminal=True, legacy_windows=False)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Indic OCR & HTR Benchmark Report</title>
    <style>
        :root {
            --bg-main: #0f172a;
            --bg-card: #1e293b;
            --border: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #38bdf8;
            --success: #4ade80;
            --warning: #facc15;
            --danger: #f87171;
            --diff-del: #7f1d1d;
            --diff-ins: #14532d;
            --diff-sub: #713f12;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans", sans-serif;
            background-color: var(--bg-main);
            color: var(--text-main);
            padding: 2rem;
            line-height: 1.5;
        }
        .header {
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        h1 { font-size: 1.8rem; color: var(--accent); }
        .badge {
            background: #1e1b4b;
            color: #818cf8;
            padding: 0.3rem 0.8rem;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.2rem;
            margin-bottom: 2.5rem;
        }
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            padding: 1.25rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        .card-title { font-size: 0.9rem; color: var(--text-muted); margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.05em; }
        .card-value { font-size: 1.8rem; font-weight: 700; }
        .score-good { color: var(--success); }
        .score-medium { color: var(--warning); }
        .score-bad { color: var(--danger); }

        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 2rem;
            background: var(--bg-card);
            border-radius: 0.75rem;
            overflow: hidden;
        }
        th, td {
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        th { background: #1e293b; color: var(--text-muted); font-size: 0.85rem; font-weight: 600; }
        tr:hover { background: #33415540; }

        .details-section { margin-top: 3rem; }
        .page-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            margin-bottom: 1.5rem;
            overflow: hidden;
        }
        .page-header {
            padding: 1rem 1.25rem;
            background: #1e293b;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .page-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            padding: 1.25rem;
        }
        .text-box {
            background: #0f172a;
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            padding: 1rem;
            font-size: 0.95rem;
            white-space: pre-wrap;
            word-break: break-word;
            font-family: "Noto Sans", "Nirmala UI", sans-serif;
            max-height: 350px;
            overflow-y: auto;
        }
        .diff-del { background-color: var(--diff-del); text-decoration: line-through; padding: 0 2px; border-radius: 2px; }
        .diff-ins { background-color: var(--diff-ins); padding: 0 2px; border-radius: 2px; }
        .diff-sub { background-color: var(--diff-sub); padding: 0 2px; border-radius: 2px; }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>Indic OCR & Handwriting Benchmark Report</h1>
            <p style="color: var(--text-muted); font-size: 0.9rem; margin-top: 0.25rem;">Language: <strong>{{ language_name }} ({{ language_code }})</strong> | Total Pages Evaluated: <strong>{{ total_pages }}</strong></p>
        </div>
        <div class="badge">Indic Pipeline Benchmark</div>
    </div>

    <div class="summary-grid">
        {% for eng_name, stats in engine_summaries.items() %}
        <div class="card">
            <div class="card-title">{{ eng_name }}</div>
            <div class="card-value {% if stats.mean_cer < 0.08 %}score-good{% elif stats.mean_cer < 0.20 %}score-medium{% else %}score-bad{% endif %}">
                CER: {{ "%.2f"|format(stats.mean_cer * 100) }}%
            </div>
            <div style="font-size: 0.95rem; color: var(--text-muted); margin-top: 0.4rem;">
                WER: <strong>{{ "%.2f"|format(stats.mean_wer * 100) }}%</strong> | Latency: <strong>{{ "%.2f"|format(stats.mean_latency) }}s</strong>
            </div>
        </div>
        {% endfor %}
    </div>

    <h2 style="font-size: 1.25rem; margin-bottom: 1rem; color: var(--text-main);">Engine Comparison Summary</h2>
    <table>
        <thead>
            <tr>
                <th>Engine</th>
                <th>Mean CER (%)</th>
                <th>Mean WER (%)</th>
                <th>Char Accuracy (%)</th>
                <th>Word Accuracy (%)</th>
                <th>Avg Latency (s)</th>
                <th>Total Errors (S / D / I)</th>
            </tr>
        </thead>
        <tbody>
            {% for eng_name, stats in engine_summaries.items() %}
            <tr>
                <td><strong>{{ eng_name }}</strong></td>
                <td><span class="{% if stats.mean_cer < 0.08 %}score-good{% elif stats.mean_cer < 0.20 %}score-medium{% else %}score-bad{% endif %}">{{ "%.2f"|format(stats.mean_cer * 100) }}%</span></td>
                <td>{{ "%.2f"|format(stats.mean_wer * 100) }}%</td>
                <td>{{ "%.2f"|format(stats.mean_char_acc) }}%</td>
                <td>{{ "%.2f"|format(stats.mean_word_acc) }}%</td>
                <td>{{ "%.2f"|format(stats.mean_latency) }}s</td>
                <td>{{ stats.total_subs }} / {{ stats.total_dels }} / {{ stats.total_ins }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>

    <div class="details-section">
        <h2 style="font-size: 1.25rem; margin-bottom: 1rem; color: var(--text-main);">Per-Page Visual & Diff Inspection</h2>
        {% for item in page_results %}
        <div class="page-card">
            <div class="page-header">
                <div>
                    <strong>Page: {{ item.image_name }}</strong>
                    <span style="color: var(--text-muted); font-size: 0.85rem; margin-left: 0.75rem;">Ground Truth: {{ item.ground_truth_len }} chars</span>
                </div>
                <div style="font-size: 0.85rem; color: var(--accent);">
                    Engine: <strong>{{ item.engine_name }}</strong> | CER: <strong>{{ "%.2f"|format(item.metrics.cer * 100) }}%</strong> | WER: <strong>{{ "%.2f"|format(item.metrics.wer * 100) }}%</strong>
                </div>
            </div>
            <div class="page-content">
                <div>
                    <div style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.4rem; font-weight: 600;">GROUND TRUTH (REFERENCE)</div>
                    <div class="text-box">{{ item.ground_truth }}</div>
                </div>
                <div>
                    <div style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.4rem; font-weight: 600;">MODEL PREDICTION</div>
                    <div class="text-box">{{ item.prediction }}</div>
                </div>
            </div>
        </div>
        {% endfor %}
    </div>
</body>
</html>
"""


class BenchmarkReporter:
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("./data/outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def print_summary_table(self, engine_summaries: Dict[str, dict], language: str = "kn") -> None:
        """Prints a styled Rich table in the console."""
        table = Table(title=f"Indic OCR Benchmark Summary ({language.upper()})", show_header=True, header_style="bold magenta")
        table.add_column("Engine", style="cyan", no_wrap=True)
        table.add_column("Mean CER", justify="right")
        table.add_column("Mean WER", justify="right")
        table.add_column("Char Accuracy", justify="right")
        table.add_column("Word Accuracy", justify="right")
        table.add_column("Avg Latency (s)", justify="right")
        table.add_column("Pages", justify="center")

        for eng_name, stats in engine_summaries.items():
            cer_pct = stats["mean_cer"] * 100
            wer_pct = stats["mean_wer"] * 100
            cer_style = "bold green" if cer_pct < 8.0 else ("bold yellow" if cer_pct < 20.0 else "bold red")

            table.add_row(
                eng_name,
                f"[{cer_style}]{cer_pct:.2f}%[/{cer_style}]",
                f"{wer_pct:.2f}%",
                f"{stats['mean_char_acc']:.2f}%",
                f"{stats['mean_word_acc']:.2f}%",
                f"{stats['mean_latency']:.2f}s",
                str(stats["total_pages"]),
            )

        console.print("\n")
        console.print(table)
        console.print("\n")

    def export_csv_and_json(self, page_results: List[dict], engine_summaries: Dict[str, dict], prefix: str = "benchmark") -> Dict[str, Path]:
        """Exports raw records and summaries to CSV and JSON formats."""
        records = []
        for r in page_results:
            records.append({
                "image_name": r["image_name"],
                "engine_name": r["engine_name"],
                "cer": r["metrics"].cer,
                "wer": r["metrics"].wer,
                "char_accuracy": r["metrics"].char_accuracy,
                "word_accuracy": r["metrics"].word_accuracy,
                "latency_seconds": r.get("latency", 0.0),
                "substitutions": r["metrics"].char_edits.substitutions,
                "deletions": r["metrics"].char_edits.deletions,
                "insertions": r["metrics"].char_edits.insertions,
                "gt_chars": r["metrics"].reference_length,
                "pred_chars": r["metrics"].prediction_length,
            })

        df = pd.DataFrame(records)
        csv_path = self.output_dir / f"{prefix}_per_page.csv"
        df.to_csv(csv_path, index=False)

        summary_json_path = self.output_dir / f"{prefix}_summary.json"
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "summaries": engine_summaries,
                "page_results_count": len(page_results),
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported metrics to CSV: {csv_path} and JSON: {summary_json_path}")
        return {"csv": csv_path, "json": summary_json_path}

    def generate_html_report(
        self,
        page_results: List[dict],
        engine_summaries: Dict[str, dict],
        language_code: str = "kn",
        language_name: str = "Kannada",
        output_filename: str = "benchmark_report.html",
    ) -> Path:
        """Renders an interactive HTML report."""
        template = Template(HTML_TEMPLATE)
        rendered_html = template.render(
            language_code=language_code,
            language_name=language_name,
            total_pages=len(set(r["image_name"] for r in page_results)),
            engine_summaries=engine_summaries,
            page_results=page_results,
        )

        out_path = self.output_dir / output_filename
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(rendered_html)

        logger.info(f"Generated visual HTML benchmark report at: {out_path}")
        return out_path
