from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .adapters import LabelInput, iter_inputs
from .image_utils import load_image
from .model import LabelerModel, ModelConfig
from .prompts import (
    PROMPT_VERSION,
    PromptInputs,
    build_messages,
    build_quality_messages,
    build_quality_prompt,
    build_user_prompt,
)
from pipeline.utils import ensure_dir, utc_now_iso

logger = logging.getLogger("labeler")


@dataclass
class LabelingConfig:
    input_path: Optional[Path]
    db_path: Optional[Path]
    outdir: Path
    model_id: str
    device: str
    precision: str
    batch_size: int
    upscale: int
    alpha_bg: str
    resume: bool
    lang: str
    only_source: str
    max_samples: Optional[int]
    no_image: bool
    no_metadata: bool
    log_level: str
    parquet: bool
    load_4bit: bool
    max_new_tokens: int
    run_id: Optional[str]
    quality_retry: bool


def run_labeling(config: LabelingConfig) -> None:
    ensure_dir(config.outdir)
    output_jsonl = config.outdir / "labels.jsonl"
    output_parquet = config.outdir / "labels.parquet"
    error_log = config.outdir / "labels_errors.log"

    if output_jsonl.exists() and not config.resume:
        raise RuntimeError("labels.jsonl already exists; use --resume or remove the file")

    existing_paths: set[str] = set()
    existing_sha: set[str] = set()
    if config.resume and output_jsonl.exists():
        existing_paths, existing_sha = _load_existing(output_jsonl)

    model = LabelerModel(
        ModelConfig(
            model_id=config.model_id,
            device=config.device,
            precision=config.precision,
            max_new_tokens=config.max_new_tokens,
            load_4bit=config.load_4bit,
        )
    )

    records_for_parquet: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    seen_sha: set[str] = set()

    with output_jsonl.open("a", encoding="utf-8") as out_handle, error_log.open(
        "a", encoding="utf-8"
    ) as err_handle:
        batch: list[LabelInput] = []
        for sample in iter_inputs(
            input_path=config.input_path,
            db_path=config.db_path,
            only_source=config.only_source,
            max_samples=config.max_samples,
            run_id=config.run_id,
        ):
            if sample.image_path in existing_paths or sample.image_path in seen_paths:
                continue
            if sample.image_sha256 and (
                sample.image_sha256 in existing_sha or sample.image_sha256 in seen_sha
            ):
                continue
            if not config.no_image:
                if not sample.image_abspath or not sample.image_abspath.exists():
                    _log_error(err_handle, sample.image_path, "missing_image")
                    continue
            batch.append(sample)
            if len(batch) >= config.batch_size:
                _process_batch(
                    batch,
                    model,
                    config,
                    out_handle,
                    err_handle,
                    records_for_parquet,
                    seen_paths,
                    seen_sha,
                )
                batch = []

        if batch:
            _process_batch(
                batch,
                model,
                config,
                out_handle,
                err_handle,
                records_for_parquet,
                seen_paths,
                seen_sha,
            )

    if config.parquet:
        _write_parquet(output_parquet, records_for_parquet)


def _process_batch(
    batch: list[LabelInput],
    model: LabelerModel,
    config: LabelingConfig,
    out_handle,
    err_handle,
    parquet_buffer: list[dict[str, object]],
    seen_paths: set[str],
    seen_sha: set[str],
) -> None:
    include_image = not config.no_image
    include_metadata = not config.no_metadata

    messages_list = []
    images = []
    active_samples: list[LabelInput] = []
    for sample in batch:
        prompt_inputs = PromptInputs(
            item_name=sample.item_name,
            item_description=sample.item_description,
            item_part=sample.item_part,
            source_type=sample.source_type,
            include_image=include_image,
            include_metadata=include_metadata,
            lang=config.lang,
        )
        user_prompt = build_user_prompt(prompt_inputs)
        messages_list.append(build_messages(user_prompt, include_image, strict=False))
        if include_image:
            try:
                images.append(load_image(sample.image_abspath, config.upscale, config.alpha_bg))
            except Exception:
                _log_error(err_handle, sample.image_path, "image_load_failed")
                messages_list.pop()
                continue
        active_samples.append(sample)

    if not active_samples:
        return

    outputs = model.generate_texts(messages_list, images if include_image else None)

    for sample, raw_text in zip(active_samples, outputs):
        record = _parse_and_build(
            sample,
            raw_text,
            model,
            config,
            err_handle,
            include_image,
            include_metadata,
        )
        if not record:
            continue
        out_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_handle.flush()
        parquet_buffer.append(record)
        seen_paths.add(sample.image_path)
        if sample.image_sha256:
            seen_sha.add(sample.image_sha256)


def _parse_and_build(
    sample: LabelInput,
    raw_text: str,
    model: LabelerModel,
    config: LabelingConfig,
    err_handle,
    include_image: bool,
    include_metadata: bool,
) -> Optional[dict[str, object]]:
    parsed = _try_parse(raw_text)
    if parsed is None:
        parsed = _retry_strict(sample, model, config, include_image, include_metadata)
    if parsed is None:
        _log_error(err_handle, sample.image_path, "invalid_json")
        return None

    try:
        record = _normalize_record(parsed, sample, config)
    except ValueError as exc:
        _log_error(err_handle, sample.image_path, str(exc))
        return None

    if config.quality_retry and include_image and _needs_quality_retry(record, sample):
        refined = _retry_quality(sample, model, config, include_image, include_metadata)
        if refined is None:
            _append_quality_reason(record, "quality_retry_failed")
            return record
        try:
            record = _normalize_record(refined, sample, config)
        except ValueError as exc:
            _log_error(err_handle, sample.image_path, str(exc))
            _append_quality_reason(record, "quality_retry_invalid")
            return record
    return record


def _retry_strict(
    sample: LabelInput,
    model: LabelerModel,
    config: LabelingConfig,
    include_image: bool,
    include_metadata: bool,
) -> Optional[dict[str, object]]:
    prompt_inputs = PromptInputs(
        item_name=sample.item_name,
        item_description=sample.item_description,
        item_part=sample.item_part,
        source_type=sample.source_type,
        include_image=include_image,
        include_metadata=include_metadata,
        lang=config.lang,
    )
    user_prompt = build_user_prompt(prompt_inputs)
    messages = [build_messages(user_prompt, include_image, strict=True)]
    images = None
    if include_image:
        try:
            images = [load_image(sample.image_abspath, config.upscale, config.alpha_bg)]
        except Exception:
            return None
    output = model.generate_texts(messages, images)
    if not output:
        return None
    return _try_parse(output[0])


def _retry_quality(
    sample: LabelInput,
    model: LabelerModel,
    config: LabelingConfig,
    include_image: bool,
    include_metadata: bool,
) -> Optional[dict[str, object]]:
    prompt_inputs = PromptInputs(
        item_name=sample.item_name,
        item_description=sample.item_description,
        item_part=sample.item_part,
        source_type=sample.source_type,
        include_image=include_image,
        include_metadata=include_metadata,
        lang=config.lang,
    )
    user_prompt = build_quality_prompt(prompt_inputs)
    messages = [build_quality_messages(user_prompt, include_image)]
    images = None
    if include_image:
        try:
            images = [load_image(sample.image_abspath, config.upscale, config.alpha_bg)]
        except Exception:
            return None
    output = model.generate_texts(messages, images)
    if not output:
        return None
    return _try_parse(output[0])


def _try_parse(raw_text: str) -> Optional[dict[str, object]]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None


def _normalize_record(
    parsed: dict[str, object],
    sample: LabelInput,
    config: LabelingConfig,
) -> dict[str, object]:
    label_ko = _clean_text(parsed.get("label_ko"))
    if not label_ko:
        raise ValueError("label_ko_missing")

    label_en = _clean_text(parsed.get("label_en"))
    if config.lang == "ko":
        label_en = None

    tags = _normalize_list(parsed.get("tags_ko"), max_items=15)
    queries = _normalize_list(parsed.get("query_variants_ko"), max_items=8)

    attributes = parsed.get("attributes") if isinstance(parsed.get("attributes"), dict) else {}
    normalized_attributes = {
        "colors": _normalize_list(attributes.get("colors"), max_items=10),
        "theme": _normalize_list(attributes.get("theme"), max_items=10),
        "material": _normalize_list(attributes.get("material"), max_items=10),
        "vibe": _normalize_list(attributes.get("vibe"), max_items=10),
        "item_type_guess": _clean_text(attributes.get("item_type_guess")),
    }

    quality = parsed.get("quality_flags") if isinstance(parsed.get("quality_flags"), dict) else {}
    is_uncertain = bool(quality.get("is_uncertain", False))
    reasons = _normalize_list(quality.get("reasons"))

    if len(tags) < 5:
        is_uncertain = True
        reasons.append("few_tags")
    if len(queries) < 3:
        is_uncertain = True
        reasons.append("few_queries")
    if _label_matches_item_name(label_ko, sample.item_name):
        is_uncertain = True
        reasons.append("label_equals_item_name")
    if _attributes_empty(normalized_attributes):
        is_uncertain = True
        reasons.append("attributes_missing")

    reasons = _unique_list(reasons)

    return {
        "image_path": sample.image_path,
        "image_sha256": sample.image_sha256,
        "source_type": sample.source_type,
        "item_name": sample.item_name,
        "item_description": sample.item_description,
        "label_ko": label_ko,
        "label_en": label_en,
        "tags_ko": tags,
        "attributes": normalized_attributes,
        "query_variants_ko": queries,
        "quality_flags": {"is_uncertain": is_uncertain, "reasons": reasons},
        "model": config.model_id,
        "prompt_version": PROMPT_VERSION,
        "generated_at": utc_now_iso(),
    }


def _normalize_list(value: object, max_items: Optional[int] = None) -> list[str]:
    if value is None:
        items: list[str] = []
    elif isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        items = [item.strip() for item in str(value).split(",") if item.strip()]
    if max_items:
        items = items[:max_items]
    return items


def _clean_text(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _needs_quality_retry(record: dict[str, object], sample: LabelInput) -> bool:
    label = record.get("label_ko") or ""
    if _label_matches_item_name(str(label), sample.item_name):
        return True
    if _attributes_empty(record.get("attributes")):
        return True
    tags = record.get("tags_ko")
    queries = record.get("query_variants_ko")
    if not isinstance(tags, list) or len(tags) < 5:
        return True
    if not isinstance(queries, list) or len(queries) < 3:
        return True
    return False


def _attributes_empty(attributes: object) -> bool:
    if not isinstance(attributes, dict):
        return True
    for key in ("colors", "theme", "material", "vibe"):
        values = attributes.get(key)
        if isinstance(values, list) and values:
            return False
    item_type = attributes.get("item_type_guess")
    if isinstance(item_type, str) and item_type.strip():
        return False
    return True


def _label_matches_item_name(label: str, item_name: str) -> bool:
    return _normalize_text_key(label) == _normalize_text_key(item_name)


def _normalize_text_key(value: str) -> str:
    return re.sub(r"[\W_]+", "", value).lower()


def _unique_list(values: Iterable[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _append_quality_reason(record: dict[str, object], reason: str) -> None:
    quality = record.get("quality_flags")
    if not isinstance(quality, dict):
        quality = {}
    reasons = quality.get("reasons")
    if not isinstance(reasons, list):
        reasons = []
    cleaned = [str(item).strip() for item in reasons if str(item).strip()]
    reasons = _unique_list([*cleaned, reason])
    quality["reasons"] = reasons
    quality["is_uncertain"] = True
    record["quality_flags"] = quality


def _log_error(handle, image_path: str, message: str) -> None:
    handle.write(f"{image_path}\t{message}\n")
    handle.flush()


def _load_existing(path: Path) -> tuple[set[str], set[str]]:
    paths: set[str] = set()
    shas: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            image_path = record.get("image_path")
            image_sha = record.get("image_sha256")
            if isinstance(image_path, str):
                paths.add(image_path)
            if isinstance(image_sha, str):
                shas.add(image_sha)
    return paths, shas


def _write_parquet(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        return
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        logger.warning("pyarrow not installed; skipping parquet output")
        return
    table = pa.Table.from_pylist(records)
    pq.write_table(table, path)
