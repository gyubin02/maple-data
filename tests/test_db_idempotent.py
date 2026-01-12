from pathlib import Path

from pipeline import db


def test_db_idempotent_inserts(tmp_path: Path) -> None:
    db_path = tmp_path / "test.sqlite"
    conn = db.connect(db_path)
    db.init_db(conn)

    run_id = "2025-01-01-acde"
    db.insert_run(conn, run_id, "2025-01-01", "2025-01-02T00:00:00Z", "{}")

    equipment = [
        {
            "ocid": "ocid-1",
            "item_equipment_part": "head",
            "equipment_slot": "slot",
            "item_name": "Hat",
            "item_icon_url": "http://example.com/hat.png",
            "item_description": "desc",
            "item_shape_name": "Shape Hat",
            "item_shape_icon_url": "http://example.com/shape.png",
            "raw_json": "{}",
        }
    ]

    cash = [
        {
            "ocid": "ocid-1",
            "preset_no": 1,
            "cash_item_equipment_part": "hat",
            "cash_item_equipment_slot": "slot",
            "cash_item_name": "Cash Hat",
            "cash_item_icon_url": "http://example.com/cash.png",
            "cash_item_description": "desc",
            "cash_item_label": "label",
            "date_expire": None,
            "date_option_expire": None,
            "raw_json": "{}",
        }
    ]

    db.upsert_equipment_items(conn, run_id, equipment)
    db.upsert_equipment_items(conn, run_id, equipment)
    db.upsert_cash_items(conn, run_id, cash)
    db.upsert_cash_items(conn, run_id, cash)
    conn.commit()

    eq_count = conn.execute("SELECT COUNT(*) FROM equipment_shape_items").fetchone()[0]
    cash_count = conn.execute("SELECT COUNT(*) FROM cash_items").fetchone()[0]

    assert eq_count == 1
    assert cash_count == 1
