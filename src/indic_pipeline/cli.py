"""
Unified CLI for the Indic OCR & HTR pipeline.
"""

from pathlib import Path
from typing import List, Optional
import rich_click as click
from rich.console import Console

from indic_pipeline.config import INDIC_LANGUAGES, settings
from indic_pipeline.dataset.loader import DatasetLoader
from indic_pipeline.dataset.parser import GroundTruthParser
from indic_pipeline.engines import get_engine, MockOCREngine
from indic_pipeline.evaluation.benchmark import BenchmarkRunner
from indic_pipeline.evaluation.metrics import evaluate_prediction
from indic_pipeline.evaluation.normalizer import IndicNormalizer
from indic_pipeline.evaluation.reporter import BenchmarkReporter
from indic_pipeline.preprocessing.pipeline import PreprocessingPipeline
from indic_pipeline.utils.cache import disk_cache
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

console = Console(force_terminal=True, legacy_windows=False)
click.rich_click.USE_RICH_MARKUP = True


@click.group()
def cli():
    """Indic OCR & Handwriting (HTR) Benchmarking & Transcription Pipeline."""
    pass


@cli.command("benchmark")
@click.option("--data-dir", "-d", type=click.Path(exists=True, file_okay=False), default="./data/samples", help="Path to directory containing images and ground truth JSON/TXT.")
@click.option("--lang", "-l", default="kn", help="Target Indic language code (kn, hi, bn, ta, te, etc.).")
@click.option("--engines", "-e", default="gemini", help="Comma-separated list of engines (gemini, sarvam, gcv, surya, tesseract, mock).")
@click.option("--max-samples", "-n", type=int, default=None, help="Maximum number of sample pages to evaluate.")
@click.option("--output-prefix", "-o", default="benchmark", help="Output prefix for reports.")
@click.option("--no-prep", is_flag=True, help="Disable image preprocessing during benchmark.")
def benchmark_cmd(data_dir: str, lang: str, engines: str, max_samples: Optional[int], output_prefix: str, no_prep: bool):
    """Run comparative multi-engine benchmark on sample dataset."""
    lang_info = INDIC_LANGUAGES.get(lang.lower(), {"name": lang})
    console.print(f"[bold cyan]Benchmarking Indic OCR for Language: {lang_info['name']} ({lang})[/bold cyan]")

    # Load dataset
    loader = DatasetLoader(data_dir=Path(data_dir), language=lang)
    samples = loader.load_samples(max_samples=max_samples)

    if not samples:
        console.print(f"[bold red]No sample images paired with ground truth found in {data_dir}![/bold red]")
        console.print("Ensure your image files (.png/.jpg) have corresponding .json or .txt ground truth files.")
        return

    # Instantiate engines
    engine_names = [e.strip() for e in engines.split(",") if e.strip()]
    engine_instances = []
    for en in engine_names:
        try:
            if en == "mock":
                engine_instances.append(MockOCREngine(name="mock_gemini_sim", simulated_cer=0.04))
                engine_instances.append(MockOCREngine(name="mock_baseline_sim", simulated_cer=0.18))
            else:
                engine_instances.append(get_engine(en))
        except Exception as e:
            console.print(f"[yellow]Warning: Could not initialize engine '{en}': {e}[/yellow]")

    if not engine_instances:
        console.print("[bold red]No valid engines could be initialized.[/bold red]")
        return

    # Run Benchmark
    runner = BenchmarkRunner(
        engines=engine_instances,
        language=lang,
    )
    runner.run(samples, apply_preprocessing=not no_prep, output_prefix=output_prefix)


@cli.command("transcribe")
@click.option("--input", "-i", "image_path", required=True, type=click.Path(exists=True, dir_okay=False), help="Path to input document image or PDF.")
@click.option("--engine", "-e", default="gemini", help="OCR/VLM engine (gemini, sarvam, gcv, surya, tesseract, mock).")
@click.option("--lang", "-l", default="kn", help="Target Indic language code (kn, hi, bn, etc.).")
@click.option("--output", "-o", type=click.Path(dir_okay=False), default=None, help="Optional output text file path.")
@click.option("--hint", help="Optional transcription prompt hint.")
@click.option("--preprocess/--no-preprocess", default=True, help="Apply image enhancement and deskewing.")
def transcribe_cmd(image_path: str, engine: str, lang: str, output: Optional[str], hint: Optional[str], preprocess: bool):
    """Transcribe a document page image or multi-page PDF."""
    from indic_pipeline.preprocessing.enhancer import ImageEnhancer

    inp_p = Path(image_path)
    eng = get_engine(engine)
    prep = PreprocessingPipeline() if preprocess else None

    if inp_p.suffix.lower() == ".pdf":
        console.print(f"[cyan]Rendering pages from PDF: {inp_p.name}...[/cyan]")
        page_files = ImageEnhancer.extract_pdf_pages(inp_p)
        full_text_parts = []

        for idx, page_img in enumerate(page_files, 1):
            if prep:
                page_img = prep.process_to_temp_file(page_img)
            res = eng.transcribe(page_img, language=lang, prompt_hint=hint)
            full_text_parts.append(f"--- Page {idx} ---\n{res.raw_text}")
            console.print(f"\n[bold green]=== Page {idx} Transcription ({eng.name}) ===[/bold green]")
            console.print(res.raw_text)

        combined_text = "\n\n".join(full_text_parts)
        if output:
            out_p = Path(output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(combined_text, encoding="utf-8")
            console.print(f"[cyan]Saved full PDF transcription to {out_p}[/cyan]")
    else:
        img_p = prep.process_to_temp_file(inp_p) if prep else inp_p
        result = eng.transcribe(img_p, language=lang, prompt_hint=hint)

        console.print(f"\n[bold green]=== Transcription Result ({eng.name}) ===[/bold green]")
        console.print(result.raw_text)
        console.print(f"[dim]Latency: {result.latency_seconds:.2f}s[/dim]\n")

        if output:
            out_p = Path(output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(result.raw_text, encoding="utf-8")
            console.print(f"[cyan]Saved transcription to {out_p}[/cyan]")


@cli.command("evaluate")
@click.option("--gt", "-g", required=True, type=click.Path(exists=True, dir_okay=False), help="Path to ground truth JSON/TXT file.")
@click.option("--pred", "-p", required=True, type=click.Path(exists=True, dir_okay=False), help="Path to prediction text file.")
@click.option("--lang", "-l", default="kn", help="Target Indic language code.")
def evaluate_cmd(gt: str, pred: str, lang: str):
    """Calculate CER and WER between ground truth and prediction files."""
    gt_text = GroundTruthParser.parse_from_file(Path(gt))
    with open(pred, "r", encoding="utf-8") as f:
        pred_text = f.read().strip()

    metrics = evaluate_prediction(gt_text, pred_text)

    console.print("\n[bold cyan]=== Evaluation Metrics ===[/bold cyan]")
    console.print(f"Character Error Rate (CER): [bold]{metrics.cer * 100:.2f}%[/bold]")
    console.print(f"Word Error Rate (WER):      [bold]{metrics.wer * 100:.2f}%[/bold]")
    console.print(f"Char Accuracy:              [bold]{metrics.char_accuracy:.2f}%[/bold]")
    console.print(f"Word Accuracy:              [bold]{metrics.word_accuracy:.2f}%[/bold]")
    console.print(f"Edits (S/D/I):              {metrics.char_edits.substitutions} / {metrics.char_edits.deletions} / {metrics.char_edits.insertions}\n")


@cli.command("preprocess")
@click.option("--input", "-i", required=True, type=click.Path(exists=True), help="Input image file or directory.")
@click.option("--output", "-o", required=True, type=click.Path(), help="Output path or directory.")
def preprocess_cmd(input: str, output: str):
    """Preprocess document image(s) with illumination correction and deskewing."""
    pipeline = PreprocessingPipeline()
    inp_p = Path(input)
    out_p = Path(output)

    if inp_p.is_file():
        pipeline.process(inp_p, output_path=out_p)
        console.print(f"[green]Preprocessed {inp_p.name} -> {out_p}[/green]")
    else:
        out_p.mkdir(parents=True, exist_ok=True)
        count = 0
        for f in inp_p.glob("*.*"):
            if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
                pipeline.process(f, output_path=out_p / f.name)
                count += 1
        console.print(f"[green]Preprocessed {count} images to {out_p}[/green]")


@cli.command("clear-cache")
def clear_cache_cmd():
    """Clear disk cache of OCR engine responses."""
    disk_cache.clear()
    console.print("[green]Cache cleared successfully.[/green]")


if __name__ == "__main__":
    cli()
