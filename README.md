# MapleStory Ranking Icon Pipeline

Data based on NEXON Open API.

## Overview
Collects MapleStory overall ranking (1-100) characters and stores:
- Equipment shape icons (`item_shape_icon`) + metadata
- Cash item icons (`cash_item_icon`) + metadata

Output includes:
- Raw JSON responses for audit/replay
- SQLite database with idempotent upserts
- Optional icon downloads with SHA256 integrity tracking

## Requirements
- Python 3.11+
- Nexon Open API key (set `NEXON_API_KEY`)

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration
Create `.env` (see `.env.example`) or export env vars:
- `NEXON_API_KEY` (required)
- `OUTPUT_DIR` (optional, default `data/`)
- `DB_PATH` (optional, default `data/YYYY-MM-DD/db.sqlite`)

## Usage
```bash
python -m pipeline run --date 2026-01-10 --top 100 --download-icons --concurrency 8 --rps 5
python -m pipeline run --top 100 --no-download
python -m pipeline --top 100
python -m pipeline --start-rank 101 --end-rank 200 --date 2026-01-10
```

The `run` subcommand is optional.

Optional filters:
- `--world-name`
- `--world-type`
- `--class-name`

Rank range:
- `--start-rank` (default 1)
- `--end-rank` (default = `--top`)
- `--top` remains as an alias for `--end-rank`

Merge additional ranges into the same run:
- Use `--run-id` with a previous `Run ID` from `README_run.md`

Preset handling:
- Default: only current preset (or preset 1 if missing)
- `--all-presets` to store all presets

## Output Layout
```
project/
  src/
  data/
    YYYY-MM-DD/
      raw/
        ranking_overall.json
        ocid/{rank}_{character_name}.json
        item_equipment/{ocid}.json
        cashitem_equipment/{ocid}.json
      db.sqlite
      icons/
        equipment_shape/
        cash/
```

## Idempotency
Runs are keyed by a deterministic `run_id` derived from `target_date` and ranking parameters, so re-running with the same inputs updates existing rows instead of creating duplicates.

## Compliance
- Data based on NEXON Open API.
- Refresh data within 30 days to stay compliant; the CLI is scheduler-friendly (cron, etc.).

## Tests
```bash
pytest
```

## Notes
- Ranking pagination continues until the requested rank range is collected or pages are exhausted.
- Raw JSON is stored unmodified (for recovery if field names change).
