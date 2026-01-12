from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer

from .pipeline import LabelingConfig, run_labeling


def run(
    input_path: Optional[Path] = typer.Option(
        None,
        "--input",
        help="Path to manifest.jsonl or manifest.parquet",
    ),
    db_path: Optional[Path] = typer.Option(
        None,
        "--db",
        help="Path to SQLite db.sqlite (used when --input is not provided)",
    ),
    outdir: Optional[Path] = typer.Option(
        None,
        "--outdir",
        help="Output directory for labels (default: input/db parent + /labels)",
    ),
    model: str = typer.Option(
        "Qwen/Qwen2-VL-2B-Instruct",
        "--model",
        help="Model ID",
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        help="Device string (auto, cpu, cuda, cuda:0)",
    ),
    precision: str = typer.Option(
        "auto",
        "--precision",
        help="auto|fp16|bf16|fp32",
    ),
    batch_size: int = typer.Option(8, "--batch-size", help="Batch size"),
    upscale: int = typer.Option(1, "--upscale", help="Upscale factor (e.g. 2 or 4)"),
    alpha_bg: str = typer.Option(
        "white",
        "--alpha-bg",
        help="Background for transparent PNGs: white|black|none",
    ),
    resume: bool = typer.Option(False, "--resume", help="Resume from existing labels.jsonl"),
    lang: str = typer.Option("ko", "--lang", help="ko|en|both"),
    only_source: str = typer.Option(
        "all",
        "--only-source",
        help="equipment_shape|cash|all",
    ),
    max_samples: Optional[int] = typer.Option(
        None,
        "--max-samples",
        help="Limit number of samples (for testing)",
    ),
    no_image: bool = typer.Option(False, "--no-image", help="Use metadata only"),
    no_metadata: bool = typer.Option(False, "--no-metadata", help="Use image only"),
    log_level: str = typer.Option("info", "--log-level", help="info|debug"),
    parquet: bool = typer.Option(False, "--parquet", help="Write labels.parquet"),
    load_4bit: bool = typer.Option(False, "--load-4bit", help="Enable 4-bit quantization"),
    max_new_tokens: int = typer.Option(384, "--max-new-tokens", help="Max new tokens"),
    quality_retry: bool = typer.Option(
        False,
        "--quality-retry/--no-quality-retry",
        help="Retry once with a stricter prompt when output is low quality",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="Filter DB inputs by run_id",
    ),
) -> None:
    """Generate CLIP-ready labels for MapleStory item icons."""

    logging.basicConfig(level=_parse_log_level(log_level), format="%(levelname)s: %(message)s")

    if not input_path and not db_path:
        typer.echo("Provide --input or --db")
        raise typer.Exit(code=1)
    if alpha_bg not in {"white", "black", "none"}:
        typer.echo("--alpha-bg must be white, black, or none")
        raise typer.Exit(code=1)
    if lang not in {"ko", "en", "both"}:
        typer.echo("--lang must be ko, en, or both")
        raise typer.Exit(code=1)
    if only_source not in {"equipment_shape", "cash", "all"}:
        typer.echo("--only-source must be equipment_shape, cash, or all")
        raise typer.Exit(code=1)
    if precision not in {"auto", "fp16", "bf16", "fp32"}:
        typer.echo("--precision must be auto, fp16, bf16, or fp32")
        raise typer.Exit(code=1)

    resolved_outdir = outdir
    if not resolved_outdir:
        if input_path:
            resolved_outdir = input_path.parent / "labels"
        else:
            resolved_outdir = db_path.parent / "labels"

    config = LabelingConfig(
        input_path=input_path,
        db_path=db_path,
        outdir=resolved_outdir,
        model_id=model,
        device=device,
        precision=precision,
        batch_size=batch_size,
        upscale=upscale,
        alpha_bg=alpha_bg,
        resume=resume,
        lang=lang,
        only_source=only_source,
        max_samples=max_samples,
        no_image=no_image,
        no_metadata=no_metadata,
        log_level=log_level,
        parquet=parquet,
        load_4bit=load_4bit,
        max_new_tokens=max_new_tokens,
        run_id=run_id,
        quality_retry=quality_retry,
    )

    run_labeling(config)


def _parse_log_level(value: str) -> int:
    value = value.lower()
    if value == "debug":
        return logging.DEBUG
    return logging.INFO
