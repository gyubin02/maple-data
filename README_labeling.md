# Labeling Pipeline (CLIP Text Labels)

Data based on NEXON Open API.

## Overview
This pipeline generates CLIP-ready text labels for MapleStory item icons using Qwen2-VL.
It consumes either a manifest file or the SQLite DB and writes:
- `labels.jsonl` (one JSON record per image)
- `labels.parquet` (optional)

## Requirements
- Python 3.11+
- GPU recommended for Qwen2-VL inference

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional (for 4-bit quantization):
```bash
pip install bitsandbytes
```

## Input Adapters
You can use one of the following:

A) Manifest (recommended)
- `data/<DATE>/manifest.parquet` or `manifest.jsonl`
- Required columns: `image_path`, `item_name`, `source_type`

B) SQLite DB
- `data/<DATE>/db.sqlite`
- Joins `equipment_shape_items` / `cash_items` with `icon_assets`

## Run
```bash
python -m labeler run \
  --input data/2026-01-10/manifest.parquet \
  --outdir data/2026-01-10/labels \
  --model Qwen/Qwen2-VL-2B-Instruct \
  --device auto \
  --batch-size 8 \
  --upscale 2 \
  --resume
```

Using DB input:
```bash
python -m labeler run \
  --db data/2026-01-10/db.sqlite \
  --outdir data/2026-01-10/labels \
  --quality-retry
```

Range filter by run_id (optional):
```bash
python -m labeler run --db data/2026-01-10/db.sqlite --run-id <RUN_ID>
```

## Output Schema
Each line in `labels.jsonl` is a JSON object:

```json
{
  "image_path": "...",
  "image_sha256": "...",
  "source_type": "equipment_shape" | "cash",
  "item_name": "...",
  "item_description": "...",
  "label_ko": "...",
  "label_en": "...",
  "tags_ko": ["..."],
  "attributes": {
    "colors": ["..."],
    "theme": ["..."],
    "material": ["..."],
    "vibe": ["..."],
    "item_type_guess": "..."
  },
  "query_variants_ko": ["..."],
  "quality_flags": {
    "is_uncertain": true,
    "reasons": ["too_small", "ambiguous_icon"]
  },
  "model": "Qwen/Qwen2-VL-2B-Instruct",
  "prompt_version": "v1",
  "generated_at": "ISO-8601"
}
```

## Prompt Versioning
- Prompt version is stored as `prompt_version` in each record.
- Current version: `v2` (see `src/labeler/prompts.py`).

## Quality
- For higher-quality labels (more visual descriptors), use `--quality-retry`.

## Resume / Idempotency
- If `labels.jsonl` already exists, use `--resume`.
- The pipeline skips images already labeled by `image_path` or `image_sha256`.

## Comparisons
You can compare modes:
- `--no-image` (metadata only)
- `--no-metadata` (image only)

## Example Output (3 lines)
```json
{"image_path":"icons/equipment_shape/abc.png","image_sha256":"sha...","source_type":"equipment_shape","item_name":"Sample Hat","item_description":null,"label_ko":"샘플 모자 아이콘, 붉은 색감","label_en":null,"tags_ko":["모자","붉은","아이콘","장비","캐릭터"],"attributes":{"colors":["red"],"theme":["fantasy"],"material":["cloth"],"vibe":["cute"],"item_type_guess":"hat"},"query_variants_ko":["샘플 모자","붉은 모자 아이콘","메이플 모자"],"quality_flags":{"is_uncertain":false,"reasons":[]},"model":"Qwen/Qwen2-VL-2B-Instruct","prompt_version":"v2","generated_at":"2026-01-10T00:00:00Z"}
{"image_path":"icons/cash/def.png","image_sha256":"sha...","source_type":"cash","item_name":"Sample Cape","item_description":"Example","label_ko":"샘플 망토 아이콘, 푸른 계열","label_en":null,"tags_ko":["망토","푸른","코디","캐시","아이콘"],"attributes":{"colors":["blue"],"theme":["classic"],"material":["silk"],"vibe":["elegant"],"item_type_guess":"cape"},"query_variants_ko":["푸른 망토","샘플 망토 아이콘","메이플 캐시 망토"],"quality_flags":{"is_uncertain":false,"reasons":[]},"model":"Qwen/Qwen2-VL-2B-Instruct","prompt_version":"v2","generated_at":"2026-01-10T00:00:00Z"}
{"image_path":"icons/equipment_shape/ghi.png","image_sha256":"sha...","source_type":"equipment_shape","item_name":"Sample Sword","item_description":null,"label_ko":"샘플 검 아이콘, 금속 느낌","label_en":null,"tags_ko":["검","무기","금속","아이콘","장비"],"attributes":{"colors":["silver"],"theme":["fantasy"],"material":["metal"],"vibe":["sharp"],"item_type_guess":"sword"},"query_variants_ko":["샘플 검","메이플 검 아이콘","금속 검"],"quality_flags":{"is_uncertain":false,"reasons":[]},"model":"Qwen/Qwen2-VL-2B-Instruct","prompt_version":"v2","generated_at":"2026-01-10T00:00:00Z"}
```
