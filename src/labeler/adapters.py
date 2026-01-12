from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger("labeler")


@dataclass
class LabelInput:
    image_path: str
    image_abspath: Optional[Path]
    image_url: Optional[str]
    image_sha256: Optional[str]
    item_name: str
    item_description: Optional[str]
    item_part: Optional[str]
    source_type: str
    ocid: Optional[str]
    ranking: Optional[int]


def iter_inputs(
    *,
    input_path: Optional[Path],
    db_path: Optional[Path],
    only_source: str,
    max_samples: Optional[int],
    run_id: Optional[str],
) -> Iterable[LabelInput]:
    if input_path:
        if input_path.suffix.lower() in {".jsonl", ".json"}:
            yield from _iter_manifest_jsonl(input_path, only_source, max_samples)
        elif input_path.suffix.lower() == ".parquet":
            yield from _iter_manifest_parquet(input_path, only_source, max_samples)
        else:
            raise ValueError(f"Unsupported input format: {input_path}")
        return

    if not db_path:
        raise ValueError("Provide --input or --db")

    yield from _iter_db(db_path, only_source, max_samples, run_id)


def _iter_manifest_jsonl(
    path: Path,
    only_source: str,
    max_samples: Optional[int],
) -> Iterable[LabelInput]:
    base_dir = path.parent
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON line in %s", path)
                continue
            sample = _build_from_record(record, base_dir, only_source)
            if not sample:
                continue
            yield sample
            count += 1
            if max_samples and count >= max_samples:
                break


def _iter_manifest_parquet(
    path: Path,
    only_source: str,
    max_samples: Optional[int],
) -> Iterable[LabelInput]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for parquet input") from exc

    base_dir = path.parent
    table = pq.read_table(path)
    rows = table.to_pylist()
    count = 0
    for record in rows:
        sample = _build_from_record(record, base_dir, only_source)
        if not sample:
            continue
        yield sample
        count += 1
        if max_samples and count >= max_samples:
            break


def _build_from_record(
    record: dict[str, object],
    base_dir: Path,
    only_source: str,
) -> Optional[LabelInput]:
    source_type = str(record.get("source_type") or "")
    if not _source_allowed(source_type, only_source):
        return None

    image_path = str(record.get("image_path") or "").strip()
    if not image_path:
        logger.warning("Missing image_path in manifest record")
        return None

    image_abspath = Path(image_path)
    if not image_abspath.is_absolute():
        image_abspath = (base_dir / image_abspath).resolve()

    item_name = str(record.get("item_name") or "").strip()
    if not item_name:
        logger.warning("Missing item_name in manifest record")
        return None

    return LabelInput(
        image_path=image_path,
        image_abspath=image_abspath,
        image_url=_optional_str(record.get("image_url")),
        image_sha256=_optional_str(record.get("image_sha256")),
        item_name=item_name,
        item_description=_optional_str(record.get("item_description")),
        item_part=_optional_str(record.get("item_part")),
        source_type=source_type,
        ocid=_optional_str(record.get("ocid")),
        ranking=_optional_int(record.get("ranking")),
    )


def _iter_db(
    db_path: Path,
    only_source: str,
    max_samples: Optional[int],
    run_id: Optional[str],
) -> Iterable[LabelInput]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    base_dir = db_path.parent

    def stream(query: str, params: tuple[object, ...], source_type: str) -> Iterable[LabelInput]:
        cursor = conn.execute(query, params)
        for row in cursor:
            local_path = row["local_path"]
            image_path = local_path or ""
            image_abspath = (base_dir / local_path).resolve() if local_path else None
            item_name = row["item_name"] or ""
            if not item_name:
                continue
            yield LabelInput(
                image_path=image_path,
                image_abspath=image_abspath,
                image_url=row["image_url"],
                image_sha256=row["sha256"],
                item_name=item_name,
                item_description=row["item_description"],
                item_part=_build_item_part(row["item_part"], row["item_slot"]),
                source_type=source_type,
                ocid=row["ocid"],
                ranking=None,
            )

    count = 0
    if only_source in ("equipment_shape", "all"):
        query, params = _equipment_query(run_id)
        for sample in stream(query, params, "equipment_shape"):
            yield sample
            count += 1
            if max_samples and count >= max_samples:
                conn.close()
                return

    if only_source in ("cash", "all"):
        query, params = _cash_query(run_id)
        for sample in stream(query, params, "cash"):
            yield sample
            count += 1
            if max_samples and count >= max_samples:
                conn.close()
                return

    conn.close()


def _equipment_query(run_id: Optional[str]) -> tuple[str, tuple[object, ...]]:
    query = (
        "SELECT e.item_shape_icon_url AS image_url, a.sha256 AS sha256, a.local_path AS local_path, "
        "e.item_name AS item_name, e.item_description AS item_description, "
        "e.item_equipment_part AS item_part, e.equipment_slot AS item_slot, e.ocid AS ocid "
        "FROM equipment_shape_items e "
        "LEFT JOIN icon_assets a ON a.url = e.item_shape_icon_url "
        "WHERE e.item_shape_icon_url IS NOT NULL AND e.item_shape_icon_url != ''"
    )
    if run_id:
        query += " AND e.run_id = ?"
        return query, (run_id,)
    return query, ()


def _cash_query(run_id: Optional[str]) -> tuple[str, tuple[object, ...]]:
    query = (
        "SELECT c.cash_item_icon_url AS image_url, a.sha256 AS sha256, a.local_path AS local_path, "
        "c.cash_item_name AS item_name, c.cash_item_description AS item_description, "
        "c.cash_item_equipment_part AS item_part, c.cash_item_equipment_slot AS item_slot, c.ocid AS ocid "
        "FROM cash_items c "
        "LEFT JOIN icon_assets a ON a.url = c.cash_item_icon_url "
        "WHERE c.cash_item_icon_url IS NOT NULL AND c.cash_item_icon_url != ''"
    )
    if run_id:
        query += " AND c.run_id = ?"
        return query, (run_id,)
    return query, ()


def _source_allowed(source_type: str, only_source: str) -> bool:
    if only_source == "all":
        return source_type in {"equipment_shape", "cash"}
    return source_type == only_source


def _build_item_part(part: Optional[str], slot: Optional[str]) -> Optional[str]:
    part = (part or "").strip()
    slot = (slot or "").strip()
    if part and slot and part != slot:
        return f"{part}/{slot}"
    return part or slot or None


def _optional_str(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
