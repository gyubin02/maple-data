from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

PROMPT_VERSION = "v1"

SYSTEM_PROMPT_BASE = (
    "You are generating labels for MapleStory item icons for CLIP training. "
    "Return a single JSON object only. Do not output markdown or extra text. "
    "Use item_name as the primary identifier; keep it intact. "
    "Use metadata if provided and avoid guessing. "
    "If uncertain, set quality_flags.is_uncertain=true and add reasons. "
    "label_ko must be one short Korean sentence with key visual keywords. "
    "tags_ko must be 5-15 short Korean keywords. "
    "query_variants_ko must be 3-8 natural Korean search queries. "
    "attributes must include colors/theme/material/vibe lists and item_type_guess string or null. "
    "If label_en is not requested, set it to null."
)

SYSTEM_PROMPT_STRICT = (
    SYSTEM_PROMPT_BASE
    + " Output must be valid JSON with double quotes and no trailing commas."
)


@dataclass
class PromptInputs:
    item_name: str
    item_description: Optional[str]
    item_part: Optional[str]
    source_type: str
    include_image: bool
    include_metadata: bool
    lang: str


def build_user_prompt(inputs: PromptInputs) -> str:
    lines = []
    if inputs.include_metadata:
        lines.append(f"item_name: {inputs.item_name}")
        if inputs.item_description:
            lines.append(f"item_description: {inputs.item_description}")
        else:
            lines.append("item_description: (none)")
        if inputs.item_part:
            lines.append(f"item_part: {inputs.item_part}")
        else:
            lines.append("item_part: (none)")
        lines.append(f"source_type: {inputs.source_type}")
    else:
        lines.append("metadata: (not provided)")
        lines.append(f"source_type: {inputs.source_type}")

    if inputs.include_image:
        lines.append("image: provided")
    else:
        lines.append("image: not provided (metadata-only)")

    lines.append(f"language: {inputs.lang}")
    lines.append(
        "Return JSON with keys: label_ko, label_en, tags_ko, attributes, "
        "query_variants_ko, quality_flags."
    )
    return "\n".join(lines)


def build_messages(user_prompt: str, include_image: bool, strict: bool) -> list[dict[str, object]]:
    system_prompt = SYSTEM_PROMPT_STRICT if strict else SYSTEM_PROMPT_BASE
    if include_image:
        content: list[dict[str, object]] = [
            {"type": "image"},
            {"type": "text", "text": user_prompt},
        ]
    else:
        content = [{"type": "text", "text": user_prompt}]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
