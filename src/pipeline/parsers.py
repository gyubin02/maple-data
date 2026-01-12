from __future__ import annotations

from typing import Any

from .utils import json_dumps, to_int


def extract_equipment_items(data: dict[str, Any], ocid: str) -> list[dict[str, Any]]:
    items = []
    for item in data.get("item_equipment", []) or []:
        items.append(
            {
                "ocid": ocid,
                "item_equipment_part": _coalesce(item.get("item_equipment_part")),
                "equipment_slot": _coalesce(item.get("equipment_slot")),
                "item_name": item.get("item_name"),
                "item_icon_url": item.get("item_icon"),
                "item_description": item.get("item_description"),
                "item_shape_name": item.get("item_shape_name"),
                "item_shape_icon_url": item.get("item_shape_icon"),
                "raw_json": json_dumps(item),
            }
        )
    return items


def extract_cash_items(
    data: dict[str, Any],
    ocid: str,
    all_presets: bool,
) -> list[dict[str, Any]]:
    presets = _extract_presets(data)
    if not presets:
        return []

    if all_presets:
        selected = presets
    else:
        selected = _select_current_or_default_presets(data, presets)

    items: list[dict[str, Any]] = []
    for preset_no, preset_items in selected:
        for item in preset_items:
            items.append(
                {
                    "ocid": ocid,
                    "preset_no": preset_no,
                    "cash_item_equipment_part": _coalesce(item.get("cash_item_equipment_part")),
                    "cash_item_equipment_slot": _coalesce(item.get("cash_item_equipment_slot")),
                    "cash_item_name": item.get("cash_item_name"),
                    "cash_item_icon_url": item.get("cash_item_icon"),
                    "cash_item_description": item.get("cash_item_description"),
                    "cash_item_label": item.get("cash_item_label"),
                    "date_expire": item.get("date_expire"),
                    "date_option_expire": item.get("date_option_expire"),
                    "raw_json": json_dumps(item),
                }
            )
    return items


def _extract_presets(data: dict[str, Any]) -> list[tuple[int, list[dict[str, Any]]]]:
    presets: list[tuple[int, list[dict[str, Any]]]] = []
    for key, value in data.items():
        if not key.startswith("cash_item_equipment_preset_"):
            continue
        if not isinstance(value, list):
            continue
        preset_no = _parse_preset_no(key)
        if preset_no is None:
            continue
        presets.append((preset_no, value))
    presets.sort(key=lambda item: item[0])
    return presets


def _parse_preset_no(key: str) -> int | None:
    try:
        return int(key.rsplit("_", 1)[-1])
    except (IndexError, ValueError):
        return None


def _coalesce(value: Any) -> str:
    return value if value is not None else ""


def _select_current_or_default_presets(
    data: dict[str, Any],
    presets: list[tuple[int, list[dict[str, Any]]]],
) -> list[tuple[int, list[dict[str, Any]]]]:
    current = to_int(data.get("preset_no"))
    if current is not None:
        for preset_no, items in presets:
            if preset_no == current:
                return [(preset_no, items)]

    for preset_no, items in presets:
        if preset_no == 1:
            return [(preset_no, items)]

    return presets
