from pipeline.parsers import extract_cash_items, extract_equipment_items


def test_extract_equipment_items_maps_shape_icon():
    data = {
        "item_equipment": [
            {
                "item_equipment_part": "head",
                "equipment_slot": "slot1",
                "item_name": "Test Hat",
                "item_icon": "http://example.com/icon.png",
                "item_description": "desc",
                "item_shape_name": "Shape Hat",
                "item_shape_icon": "http://example.com/shape.png",
            }
        ]
    }
    items = extract_equipment_items(data, "ocid-1")
    assert len(items) == 1
    assert items[0]["item_shape_icon_url"] == "http://example.com/shape.png"
    assert items[0]["item_icon_url"] == "http://example.com/icon.png"
    assert items[0]["ocid"] == "ocid-1"


def test_extract_cash_items_selects_current_preset():
    data = {
        "preset_no": 2,
        "cash_item_equipment_preset_1": [
            {
                "cash_item_equipment_part": "hat",
                "cash_item_equipment_slot": "slot1",
                "cash_item_name": "Hat 1",
                "cash_item_icon": "http://example.com/hat1.png",
            }
        ],
        "cash_item_equipment_preset_2": [
            {
                "cash_item_equipment_part": "hat",
                "cash_item_equipment_slot": "slot2",
                "cash_item_name": "Hat 2",
                "cash_item_icon": "http://example.com/hat2.png",
            }
        ],
    }
    items = extract_cash_items(data, "ocid-1", all_presets=False)
    assert len(items) == 1
    assert items[0]["preset_no"] == 2
    assert items[0]["cash_item_icon_url"] == "http://example.com/hat2.png"


def test_extract_cash_items_defaults_to_preset1():
    data = {
        "cash_item_equipment_preset_1": [
            {
                "cash_item_equipment_part": "hat",
                "cash_item_equipment_slot": "slot1",
                "cash_item_name": "Hat 1",
                "cash_item_icon": "http://example.com/hat1.png",
            }
        ],
        "cash_item_equipment_preset_2": [
            {
                "cash_item_equipment_part": "hat",
                "cash_item_equipment_slot": "slot2",
                "cash_item_name": "Hat 2",
                "cash_item_icon": "http://example.com/hat2.png",
            }
        ],
    }
    items = extract_cash_items(data, "ocid-1", all_presets=False)
    assert len(items) == 1
    assert items[0]["preset_no"] == 1


def test_extract_cash_items_all_presets():
    data = {
        "cash_item_equipment_preset_1": [
            {
                "cash_item_equipment_part": "hat",
                "cash_item_equipment_slot": "slot1",
                "cash_item_name": "Hat 1",
                "cash_item_icon": "http://example.com/hat1.png",
            }
        ],
        "cash_item_equipment_preset_2": [
            {
                "cash_item_equipment_part": "hat",
                "cash_item_equipment_slot": "slot2",
                "cash_item_name": "Hat 2",
                "cash_item_icon": "http://example.com/hat2.png",
            }
        ],
    }
    items = extract_cash_items(data, "ocid-1", all_presets=True)
    assert len(items) == 2
    assert {item["preset_no"] for item in items} == {1, 2}
