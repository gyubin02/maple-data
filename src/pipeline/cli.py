from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import typer

from .pipeline import run_pipeline
from .utils import get_env_or_none, kst_yesterday_date, load_dotenv_if_available


def run(
    date: Optional[str] = typer.Option(
        None,
        "--date",
        help="Target date in YYYY-MM-DD (KST). Defaults to yesterday in KST.",
    ),
    top: int = typer.Option(100, "--top", help="Alias for --end-rank (default 100)."),
    start_rank: int = typer.Option(1, "--start-rank", help="Starting ranking (inclusive)."),
    end_rank: Optional[int] = typer.Option(
        None,
        "--end-rank",
        help="Ending ranking (inclusive). Defaults to --top.",
    ),
    download_icons: bool = typer.Option(
        True,
        "--download-icons/--no-download",
        help="Download icon assets locally.",
    ),
    concurrency: int = typer.Option(8, "--concurrency", help="Max concurrent requests."),
    rps: float = typer.Option(5.0, "--rps", help="Requests per second throttle."),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Base output directory for data storage.",
    ),
    db_path: Optional[Path] = typer.Option(
        None,
        "--db-path",
        help="Override SQLite DB path (default is output_dir/date/db.sqlite).",
    ),
    world_name: Optional[str] = typer.Option(None, "--world-name", help="Ranking world name filter."),
    world_type: Optional[int] = typer.Option(None, "--world-type", help="Ranking world type filter."),
    class_name: Optional[str] = typer.Option(None, "--class-name", help="Ranking class filter."),
    all_presets: bool = typer.Option(
        False,
        "--all-presets",
        help="Store all cash equipment presets instead of current/default.",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="Reuse an existing run_id to merge additional ranking ranges.",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="Override NEXON_API_KEY environment variable.",
    ),
) -> None:
    """Collect MapleStory ranking equipment and cash icons via Nexon Open API."""

    load_dotenv_if_available()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    resolved_key = api_key or get_env_or_none("NEXON_API_KEY")
    if not resolved_key:
        typer.echo("Missing NEXON_API_KEY. Set it in the environment or pass --api-key.")
        raise typer.Exit(code=1)

    resolved_date = date or kst_yesterday_date()
    resolved_output_dir = output_dir or Path(get_env_or_none("OUTPUT_DIR") or "data")

    resolved_end_rank = end_rank or top
    if start_rank < 1:
        typer.echo("--start-rank must be >= 1")
        raise typer.Exit(code=1)
    if resolved_end_rank < start_rank:
        typer.echo("--end-rank must be >= --start-rank")
        raise typer.Exit(code=1)

    report = asyncio.run(
        run_pipeline(
            api_key=resolved_key,
            target_date=resolved_date,
            start_rank=start_rank,
            end_rank=resolved_end_rank,
            download_icon_assets=download_icons,
            output_dir=resolved_output_dir,
            db_path=db_path,
            concurrency=concurrency,
            rps=rps,
            world_name=world_name,
            world_type=world_type,
            class_name=class_name,
            all_presets=all_presets,
            run_id_override=run_id,
        )
    )

    typer.echo("Run complete")
    typer.echo(report.to_markdown())
