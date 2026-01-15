#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, TypeVar

import chromadb
import torch
import torch.nn.functional as F
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import SiglipModel, SiglipProcessor

from keyword_filters import (
    CATEGORY_SYNONYMS,
    COLOR_SYNONYMS,
    VIBE_SYNONYMS,
    extract_keywords,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
T = TypeVar("T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index images with SigLIP + LoRA embeddings into ChromaDB."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/2026-01-11"),
        help="Root directory containing images to index.",
    )
    parser.add_argument(
        "--model-id",
        default="google/siglip-base-patch16-256-multilingual",
        help="Base SigLIP model ID.",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=Path("outputs/ko-clip-lora"),
        help="Path to the LoRA adapter directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for image embedding.",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=Path("chroma_db"),
        help="Directory for ChromaDB persistent storage.",
    )
    parser.add_argument(
        "--collection",
        default="maple_items",
        help="ChromaDB collection name.",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=None,
        help="Path to labels.jsonl (defaults to data-dir/labels/labels.jsonl).",
    )
    return parser.parse_args()


def resolve_adapter_path(adapter_path: Path) -> Path:
    if (adapter_path / "adapter_config.json").exists():
        return adapter_path
    candidate = adapter_path / "best_model"
    if (candidate / "adapter_config.json").exists():
        return candidate
    return adapter_path


def find_images(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")
    images = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images)


def batch_iter(items: List[T], batch_size: int) -> Iterable[List[T]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_ids(paths: List[Path]) -> List[str]:
    counts = {}
    ids: List[str] = []
    for path in paths:
        stem = path.stem
        count = counts.get(stem, 0)
        ids.append(stem if count == 0 else f"{stem}_{count}")
        counts[stem] = count + 1
    return ids


def load_images(
    paths: List[Path], ids: List[str]
) -> Tuple[List[Image.Image], List[Path], List[str]]:
    images: List[Image.Image] = []
    valid_paths: List[Path] = []
    valid_ids: List[str] = []
    for path, item_id in zip(paths, ids):
        try:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
            valid_paths.append(path)
            valid_ids.append(item_id)
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping unreadable image: {path} ({exc})")
    return images, valid_paths, valid_ids


def normalize_label(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed or None
    return str(value)


def detect_category(texts: List[str]) -> Optional[str]:
    lowered_texts = [text.lower() for text in texts if text]
    for category, keywords in CATEGORY_SYNONYMS.items():
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if any(keyword_lower in text for text in lowered_texts):
                return category
    return None


def collect_label_texts(
    item_name: Optional[str],
    label_ko: Optional[str],
    tags: List[str],
    query_variants: List[str],
    attributes: Dict[str, object],
    item_type_guess: Optional[str],
) -> List[str]:
    texts: List[str] = []
    for value in (item_name, label_ko, item_type_guess):
        if value:
            texts.append(value)
    texts.extend(tag for tag in tags if tag)
    texts.extend(variant for variant in query_variants if variant)
    for value in attributes.values():
        if isinstance(value, list):
            for entry in value:
                entry_norm = normalize_label(entry)
                if entry_norm:
                    texts.append(entry_norm)
        else:
            entry_norm = normalize_label(value)
            if entry_norm:
                texts.append(entry_norm)
    return texts


def load_labels(labels_path: Path) -> Dict[str, Dict[str, object]]:
    if not labels_path.exists():
        print(f"Labels file not found, continuing without labels: {labels_path}")
        return {}

    label_map: Dict[str, Dict[str, object]] = {}
    with labels_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skipping label line {line_no}: {exc}")
                continue

            image_path = record.get("image_path")
            if not image_path:
                continue

            item_name = normalize_label(record.get("item_name"))
            label_ko = normalize_label(record.get("label_ko"))
            tags = record.get("tags_ko") or []
            tag_texts = [normalize_label(tag) for tag in tags if tag is not None]
            tag_texts = [tag for tag in tag_texts if tag]
            query_variants = record.get("query_variants_ko") or []
            variant_texts = [
                normalize_label(variant)
                for variant in query_variants
                if variant is not None
            ]
            variant_texts = [variant for variant in variant_texts if variant]
            attributes = record.get("attributes") or {}
            item_type_guess = normalize_label(attributes.get("item_type_guess"))
            if not item_name and not label_ko and not tag_texts:
                continue

            normalized_path = Path(str(image_path)).as_posix().lstrip("./")
            label_map[normalized_path] = {}
            if item_name:
                label_map[normalized_path]["item_name"] = item_name
            if label_ko:
                label_map[normalized_path]["label_ko"] = label_ko
                label_map[normalized_path]["label"] = label_ko
            texts = collect_label_texts(
                item_name,
                label_ko,
                tag_texts,
                variant_texts,
                attributes,
                item_type_guess,
            )
            category = detect_category(texts)
            if category:
                label_map[normalized_path]["category"] = category
            colors = extract_keywords(texts, COLOR_SYNONYMS)
            if colors:
                label_map[normalized_path]["colors"] = colors
                for color in colors:
                    label_map[normalized_path][f"color_{color}"] = True
            vibes = extract_keywords(texts, VIBE_SYNONYMS)
            if vibes:
                label_map[normalized_path]["vibes"] = vibes
                for vibe in vibes:
                    label_map[normalized_path][f"vibe_{vibe}"] = True

    print(f"Loaded labels for {len(label_map)} images from {labels_path}")
    return label_map


def main() -> None:
    args = parse_args()

    image_paths = find_images(args.data_dir)
    if not image_paths:
        print(f"No images found under {args.data_dir}")
        return

    ids = build_ids(image_paths)
    adapter_path = resolve_adapter_path(args.adapter_path)
    labels_path = args.labels_path or args.data_dir / "labels/labels.jsonl"
    label_map = load_labels(labels_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model and processor...")
    base_model = SiglipModel.from_pretrained(args.model_id)
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    processor = SiglipProcessor.from_pretrained(args.model_id)

    model.to(device)
    model.eval()

    client = chromadb.PersistentClient(path=str(args.chroma_path))
    collection = client.get_or_create_collection(
        name=args.collection,
        metadata={"hnsw:space": "cosine"},
    )

    total_images = len(image_paths)
    progress = tqdm(total=total_images, desc="Indexing images", unit="img")
    indexed_count = 0

    with torch.no_grad():
        for batch_paths, batch_ids in zip(
            batch_iter(image_paths, args.batch_size),
            batch_iter(ids, args.batch_size),
        ):
            images, valid_paths, valid_ids = load_images(batch_paths, batch_ids)
            if not images:
                progress.update(len(batch_paths))
                continue

            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}

            embeds = model.get_image_features(**inputs)
            embeds = F.normalize(embeds, dim=-1)

            embeddings = embeds.detach().cpu().tolist()
            metadatas = []
            for path in valid_paths:
                rel_path = path.relative_to(args.data_dir).as_posix()
                metadata = {"filepath": rel_path}
                label_data = label_map.get(rel_path)
                if label_data:
                    metadata.update(label_data)
                metadatas.append(metadata)

            collection.upsert(
                ids=valid_ids,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            indexed_count += len(valid_paths)
            progress.update(len(batch_paths))

    progress.close()
    print(
        f"Indexed {indexed_count} images (scanned {total_images}) into collection "
        f"'{args.collection}'."
    )


if __name__ == "__main__":
    main()
