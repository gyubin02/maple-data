from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Iterable

from . import db
from .api import ApiClient
from .downloader import download_icons
from .parsers import extract_cash_items, extract_equipment_items
from .utils import (
    PipelineReport,
    compute_run_id,
    ensure_dir,
    get_env_or_none,
    kst_yesterday_date,
    safe_filename,
    to_int,
    utc_now_iso,
    write_json,
)

logger = logging.getLogger("pipeline")


async def run_pipeline(
    *,
    api_key: str,
    target_date: str | None,
    start_rank: int,
    end_rank: int,
    download_icon_assets: bool,
    output_dir: Path,
    db_path: Path | None,
    concurrency: int,
    rps: float,
    world_name: str | None,
    world_type: int | None,
    class_name: str | None,
    all_presets: bool,
    run_id_override: str | None,
) -> PipelineReport:
    start = time.monotonic()
    resolved_date = target_date or kst_yesterday_date()

    output_root = output_dir / resolved_date
    raw_root = output_root / "raw"
    raw_ocid = raw_root / "ocid"
    raw_item = raw_root / "item_equipment"
    raw_cash = raw_root / "cashitem_equipment"
    icons_root = output_root / "icons"

    ensure_dir(raw_ocid)
    ensure_dir(raw_item)
    ensure_dir(raw_cash)
    ensure_dir(icons_root / "equipment_shape")
    ensure_dir(icons_root / "cash")

    resolved_db_path = db_path or Path(get_env_or_none("DB_PATH") or output_root / "db.sqlite")
    conn = db.connect(resolved_db_path)
    db.init_db(conn)

    run_params = {
        "start_rank": start_rank,
        "end_rank": end_rank,
        "world_name": world_name,
        "world_type": world_type,
        "class_name": class_name,
        "all_presets": all_presets,
    }
    run_id = run_id_override or compute_run_id(resolved_date, run_params)
    existing_run = db.fetch_run(conn, run_id)
    if existing_run:
        if existing_run["target_date"] != resolved_date:
            raise ValueError(
                f"run_id {run_id} target_date mismatch ({existing_run['target_date']} != {resolved_date})"
            )
    else:
        db.insert_run(conn, run_id, resolved_date, utc_now_iso(), json.dumps(run_params, ensure_ascii=False))
        conn.commit()

    equipment_items: list[dict[str, Any]] = []
    cash_items: list[dict[str, Any]] = []
    ocid_results: list[dict[str, Any]] = []

    async with ApiClient(api_key=api_key, concurrency=concurrency, rps=rps) as api:
        ranking_entries, ranking_raw = await _fetch_ranking_entries(
            api=api,
            target_date=resolved_date,
            start_rank=start_rank,
            end_rank=end_rank,
            world_name=world_name,
            world_type=world_type,
            class_name=class_name,
        )
        write_json(raw_root / "ranking_overall.json", ranking_raw)
        db.upsert_ranking_entries(conn, run_id, ranking_entries)
        conn.commit()

        ocid_results = await _fetch_ocids(api, ranking_entries, raw_ocid)
        now_iso = utc_now_iso()
        character_rows = [
            {
                "ocid": row["ocid"],
                "character_name": row["character_name"],
                "first_seen_at": now_iso,
                "last_seen_at": now_iso,
            }
            for row in ocid_results
            if row.get("ocid")
        ]
        if character_rows:
            db.upsert_characters(conn, character_rows)
            conn.commit()

        ocids = [row["ocid"] for row in ocid_results if row.get("ocid")]
        equipment_items = await _fetch_equipment(api, ocids, resolved_date, raw_item)
        if equipment_items:
            db.upsert_equipment_items(conn, run_id, equipment_items)
            conn.commit()

        cash_items = await _fetch_cash_items(api, ocids, resolved_date, raw_cash, all_presets)
        if cash_items:
            db.upsert_cash_items(conn, run_id, cash_items)
            conn.commit()

        metrics = api.metrics

    icon_results = None
    download_counts = None
    if download_icon_assets:
        urls_by_category = _collect_icon_urls(equipment_items, cash_items)
        existing = db.fetch_icon_assets(conn, list(urls_by_category.keys()))
        icon_results, download_counts = await download_icons(
            urls_by_category=urls_by_category,
            output_root=output_root,
            icons_root=icons_root,
            existing_assets=existing,
            rps=rps,
            concurrency=concurrency,
        )
        for record in icon_results:
            db.upsert_icon_asset(conn, record.__dict__)
        conn.commit()

    elapsed = time.monotonic() - start
    report = PipelineReport(
        run_id=run_id,
        target_date=resolved_date,
        start_rank=start_rank,
        end_rank=end_rank,
        ranking_count=len(ranking_entries),
        ocid_count=len(ocid_results),
        equipment_items_count=len(equipment_items),
        cash_items_count=len(cash_items),
        icons_downloaded=download_counts.downloaded if download_counts else 0,
        icons_skipped=download_counts.skipped if download_counts else 0,
        icons_failed=download_counts.failed if download_counts else 0,
        rate_limit_hits=metrics.rate_limit_hits,
        server_errors=metrics.server_errors,
        data_preparing_hits=metrics.data_preparing_hits,
        elapsed_seconds=elapsed,
    )

    report_path = output_root / "README_run.md"
    report_path.write_text(report.to_markdown(), encoding="utf-8")
    return report


async def _fetch_ranking_entries(
    *,
    api: ApiClient,
    target_date: str,
    start_rank: int,
    end_rank: int,
    world_name: str | None,
    world_type: int | None,
    class_name: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    page = 1
    collected: dict[int, dict[str, Any]] = {}
    pages: list[dict[str, Any]] = []
    max_pages = 50
    target_count = end_rank - start_rank + 1

    while len(collected) < target_count and page <= max_pages:
        params = {"date": target_date, "page": page}
        if world_name:
            params["world_name"] = world_name
        if world_type is not None:
            params["world_type"] = world_type
        if class_name:
            params["character_class"] = class_name

        data = await api.get("/maplestory/v1/ranking/overall", params=params)
        pages.append({"page": page, "data": data})
        ranking_list = data.get("ranking") or []
        if not ranking_list:
            break

        new_count = 0
        max_rank_in_page = None
        for entry in ranking_list:
            ranking = to_int(entry.get("ranking"))
            if ranking is None:
                continue
            max_rank_in_page = ranking if max_rank_in_page is None else max(max_rank_in_page, ranking)
            if ranking < start_rank or ranking > end_rank:
                continue
            if ranking not in collected:
                collected[ranking] = {
                    "ranking": ranking,
                    "character_name": entry.get("character_name"),
                    "world_name": entry.get("world_name"),
                    "class_name": entry.get("class_name"),
                    "sub_class_name": entry.get("sub_class_name"),
                    "character_level": to_int(entry.get("character_level")),
                    "character_exp": to_int(entry.get("character_exp")),
                    "character_popularity": to_int(entry.get("character_popularity")),
                    "character_guildname": entry.get("character_guildname"),
                }
                new_count += 1
        if len(collected) >= target_count:
            break
        if max_rank_in_page is not None and max_rank_in_page < start_rank:
            page += 1
            continue
        if new_count == 0:
            logger.warning("No new ranking entries found on page %s", page)
            break
        page += 1

    entries = [collected[key] for key in sorted(collected.keys())]
    raw = {
        "target_date": target_date,
        "start_rank": start_rank,
        "end_rank": end_rank,
        "pages": pages,
        "collected_count": len(entries),
    }
    if len(entries) < target_count:
        logger.warning("Ranking entries collected %s < requested %s", len(entries), target_count)
    return entries, raw


async def _fetch_ocids(
    api: ApiClient,
    ranking_entries: Iterable[dict[str, Any]],
    raw_dir: Path,
) -> list[dict[str, Any]]:
    tasks = []
    for entry in ranking_entries:
        character_name = entry.get("character_name")
        rank = entry.get("ranking")
        if not character_name:
            continue
        tasks.append(
            asyncio.create_task(_fetch_single_ocid(api, character_name, rank, raw_dir))
        )

    results: list[dict[str, Any]] = []
    if tasks:
        completed = await asyncio.gather(*tasks, return_exceptions=True)
        for item in completed:
            if isinstance(item, Exception):
                logger.warning("OCID fetch failed: %s", item)
                continue
            if item:
                results.append(item)
    return results


async def _fetch_single_ocid(
    api: ApiClient,
    character_name: str,
    rank: int | None,
    raw_dir: Path,
) -> dict[str, Any] | None:
    data = await api.get("/maplestory/v1/id", params={"character_name": character_name})
    ocid = data.get("ocid")
    filename = f"{rank:03d}" if rank is not None else "unknown"
    filename = f"{filename}_{safe_filename(character_name)}.json"
    write_json(raw_dir / filename, data)
    if not ocid:
        logger.warning("No OCID for %s", character_name)
        return None
    return {"ocid": ocid, "character_name": character_name}


async def _fetch_equipment(
    api: ApiClient,
    ocids: list[str],
    target_date: str,
    raw_dir: Path,
) -> list[dict[str, Any]]:
    tasks = [
        asyncio.create_task(_fetch_single_equipment(api, ocid, target_date, raw_dir))
        for ocid in ocids
    ]
    results: list[dict[str, Any]] = []
    if tasks:
        completed = await asyncio.gather(*tasks, return_exceptions=True)
        for item in completed:
            if isinstance(item, Exception):
                logger.warning("Equipment fetch failed: %s", item)
                continue
            results.extend(item)
    return results


async def _fetch_single_equipment(
    api: ApiClient,
    ocid: str,
    target_date: str,
    raw_dir: Path,
) -> list[dict[str, Any]]:
    data = await api.get(
        "/maplestory/v1/character/item-equipment",
        params={"ocid": ocid, "date": target_date},
    )
    write_json(raw_dir / f"{ocid}.json", data)
    return extract_equipment_items(data, ocid)


async def _fetch_cash_items(
    api: ApiClient,
    ocids: list[str],
    target_date: str,
    raw_dir: Path,
    all_presets: bool,
) -> list[dict[str, Any]]:
    tasks = [
        asyncio.create_task(_fetch_single_cash_item(api, ocid, target_date, raw_dir, all_presets))
        for ocid in ocids
    ]
    results: list[dict[str, Any]] = []
    if tasks:
        completed = await asyncio.gather(*tasks, return_exceptions=True)
        for item in completed:
            if isinstance(item, Exception):
                logger.warning("Cash item fetch failed: %s", item)
                continue
            results.extend(item)
    return results


async def _fetch_single_cash_item(
    api: ApiClient,
    ocid: str,
    target_date: str,
    raw_dir: Path,
    all_presets: bool,
) -> list[dict[str, Any]]:
    data = await api.get(
        "/maplestory/v1/character/cashitem-equipment",
        params={"ocid": ocid, "date": target_date},
    )
    write_json(raw_dir / f"{ocid}.json", data)
    return extract_cash_items(data, ocid, all_presets=all_presets)


def _collect_icon_urls(
    equipment_items: Iterable[dict[str, Any]],
    cash_items: Iterable[dict[str, Any]],
) -> dict[str, str]:
    urls: dict[str, str] = {}
    for item in equipment_items:
        url = item.get("item_shape_icon_url")
        if url and url not in urls:
            urls[url] = "equipment_shape"
    for item in cash_items:
        url = item.get("cash_item_icon_url")
        if url and url not in urls:
            urls[url] = "cash"
    return urls
