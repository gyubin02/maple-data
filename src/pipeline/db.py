from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            target_date TEXT NOT NULL,
            created_at TEXT NOT NULL,
            params_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS ranking_entries (
            run_id TEXT NOT NULL,
            ranking INTEGER NOT NULL,
            character_name TEXT,
            world_name TEXT,
            class_name TEXT,
            sub_class_name TEXT,
            character_level INTEGER,
            character_exp INTEGER,
            character_popularity INTEGER,
            character_guildname TEXT,
            UNIQUE(run_id, ranking)
        );

        CREATE TABLE IF NOT EXISTS characters (
            ocid TEXT PRIMARY KEY,
            character_name TEXT,
            first_seen_at TEXT,
            last_seen_at TEXT
        );

        CREATE TABLE IF NOT EXISTS equipment_shape_items (
            run_id TEXT NOT NULL,
            ocid TEXT NOT NULL,
            item_equipment_part TEXT,
            equipment_slot TEXT,
            item_name TEXT,
            item_icon_url TEXT,
            item_description TEXT,
            item_shape_name TEXT,
            item_shape_icon_url TEXT,
            raw_json TEXT,
            UNIQUE(run_id, ocid, item_equipment_part, equipment_slot)
        );

        CREATE TABLE IF NOT EXISTS cash_items (
            run_id TEXT NOT NULL,
            ocid TEXT NOT NULL,
            preset_no INTEGER,
            cash_item_equipment_part TEXT,
            cash_item_equipment_slot TEXT,
            cash_item_name TEXT,
            cash_item_icon_url TEXT,
            cash_item_description TEXT,
            cash_item_label TEXT,
            date_expire TEXT,
            date_option_expire TEXT,
            raw_json TEXT,
            UNIQUE(run_id, ocid, preset_no, cash_item_equipment_part, cash_item_equipment_slot)
        );

        CREATE TABLE IF NOT EXISTS icon_assets (
            url TEXT PRIMARY KEY,
            sha256 TEXT,
            local_path TEXT,
            content_type TEXT,
            byte_size INTEGER,
            fetched_at TEXT,
            error TEXT
        );
        """
    )


def fetch_run(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    cursor = conn.execute(
        "SELECT run_id, target_date, created_at, params_json FROM runs WHERE run_id = ?",
        (run_id,),
    )
    row = cursor.fetchone()
    if not row:
        return None
    return {
        "run_id": row[0],
        "target_date": row[1],
        "created_at": row[2],
        "params_json": row[3],
    }


def insert_run(
    conn: sqlite3.Connection,
    run_id: str,
    target_date: str,
    created_at: str,
    params_json: str,
) -> None:
    conn.execute(
        """
        INSERT INTO runs (run_id, target_date, created_at, params_json)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            target_date = excluded.target_date,
            created_at = excluded.created_at,
            params_json = excluded.params_json
        """,
        (run_id, target_date, created_at, params_json),
    )


def upsert_ranking_entries(
    conn: sqlite3.Connection,
    run_id: str,
    entries: Iterable[dict[str, Any]],
) -> None:
    rows = [
        (
            run_id,
            entry.get("ranking"),
            entry.get("character_name"),
            entry.get("world_name"),
            entry.get("class_name"),
            entry.get("sub_class_name"),
            entry.get("character_level"),
            entry.get("character_exp"),
            entry.get("character_popularity"),
            entry.get("character_guildname"),
        )
        for entry in entries
    ]
    conn.executemany(
        """
        INSERT INTO ranking_entries (
            run_id,
            ranking,
            character_name,
            world_name,
            class_name,
            sub_class_name,
            character_level,
            character_exp,
            character_popularity,
            character_guildname
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, ranking) DO UPDATE SET
            character_name = excluded.character_name,
            world_name = excluded.world_name,
            class_name = excluded.class_name,
            sub_class_name = excluded.sub_class_name,
            character_level = excluded.character_level,
            character_exp = excluded.character_exp,
            character_popularity = excluded.character_popularity,
            character_guildname = excluded.character_guildname
        """,
        rows,
    )


def upsert_characters(
    conn: sqlite3.Connection,
    rows: Iterable[dict[str, Any]],
) -> None:
    prepared = [
        (
            row.get("ocid"),
            row.get("character_name"),
            row.get("first_seen_at"),
            row.get("last_seen_at"),
        )
        for row in rows
    ]
    conn.executemany(
        """
        INSERT INTO characters (ocid, character_name, first_seen_at, last_seen_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(ocid) DO UPDATE SET
            character_name = excluded.character_name,
            last_seen_at = excluded.last_seen_at
        """,
        prepared,
    )


def upsert_equipment_items(
    conn: sqlite3.Connection,
    run_id: str,
    rows: Iterable[dict[str, Any]],
) -> None:
    prepared = [
        (
            run_id,
            row.get("ocid"),
            row.get("item_equipment_part"),
            row.get("equipment_slot"),
            row.get("item_name"),
            row.get("item_icon_url"),
            row.get("item_description"),
            row.get("item_shape_name"),
            row.get("item_shape_icon_url"),
            row.get("raw_json"),
        )
        for row in rows
    ]
    conn.executemany(
        """
        INSERT INTO equipment_shape_items (
            run_id,
            ocid,
            item_equipment_part,
            equipment_slot,
            item_name,
            item_icon_url,
            item_description,
            item_shape_name,
            item_shape_icon_url,
            raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, ocid, item_equipment_part, equipment_slot) DO UPDATE SET
            item_name = excluded.item_name,
            item_icon_url = excluded.item_icon_url,
            item_description = excluded.item_description,
            item_shape_name = excluded.item_shape_name,
            item_shape_icon_url = excluded.item_shape_icon_url,
            raw_json = excluded.raw_json
        """,
        prepared,
    )


def upsert_cash_items(
    conn: sqlite3.Connection,
    run_id: str,
    rows: Iterable[dict[str, Any]],
) -> None:
    prepared = [
        (
            run_id,
            row.get("ocid"),
            row.get("preset_no"),
            row.get("cash_item_equipment_part"),
            row.get("cash_item_equipment_slot"),
            row.get("cash_item_name"),
            row.get("cash_item_icon_url"),
            row.get("cash_item_description"),
            row.get("cash_item_label"),
            row.get("date_expire"),
            row.get("date_option_expire"),
            row.get("raw_json"),
        )
        for row in rows
    ]
    conn.executemany(
        """
        INSERT INTO cash_items (
            run_id,
            ocid,
            preset_no,
            cash_item_equipment_part,
            cash_item_equipment_slot,
            cash_item_name,
            cash_item_icon_url,
            cash_item_description,
            cash_item_label,
            date_expire,
            date_option_expire,
            raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, ocid, preset_no, cash_item_equipment_part, cash_item_equipment_slot)
        DO UPDATE SET
            cash_item_name = excluded.cash_item_name,
            cash_item_icon_url = excluded.cash_item_icon_url,
            cash_item_description = excluded.cash_item_description,
            cash_item_label = excluded.cash_item_label,
            date_expire = excluded.date_expire,
            date_option_expire = excluded.date_option_expire,
            raw_json = excluded.raw_json
        """,
        prepared,
    )


def fetch_icon_assets(
    conn: sqlite3.Connection,
    urls: list[str],
) -> dict[str, dict[str, Any]]:
    if not urls:
        return {}
    placeholders = ",".join(["?"] * len(urls))
    query = f"SELECT url, sha256, local_path, error FROM icon_assets WHERE url IN ({placeholders})"
    cursor = conn.execute(query, urls)
    result = {}
    for row in cursor.fetchall():
        result[row[0]] = {"sha256": row[1], "local_path": row[2], "error": row[3]}
    return result


def upsert_icon_asset(conn: sqlite3.Connection, record: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO icon_assets (
            url,
            sha256,
            local_path,
            content_type,
            byte_size,
            fetched_at,
            error
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(url) DO UPDATE SET
            sha256 = excluded.sha256,
            local_path = excluded.local_path,
            content_type = excluded.content_type,
            byte_size = excluded.byte_size,
            fetched_at = excluded.fetched_at,
            error = excluded.error
        """,
        (
            record.get("url"),
            record.get("sha256"),
            record.get("local_path"),
            record.get("content_type"),
            record.get("byte_size"),
            record.get("fetched_at"),
            record.get("error"),
        ),
    )
