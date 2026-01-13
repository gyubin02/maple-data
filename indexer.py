#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple, TypeVar

import chromadb
import torch
import torch.nn.functional as F
from peft import PeftModel
from PIL import Image
from tqdm import tqdm
from transformers import SiglipModel, SiglipProcessor

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


def main() -> None:
    args = parse_args()

    image_paths = find_images(args.data_dir)
    if not image_paths:
        print(f"No images found under {args.data_dir}")
        return

    ids = build_ids(image_paths)
    adapter_path = resolve_adapter_path(args.adapter_path)

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
            metadatas = [
                {"filepath": str(path.relative_to(args.data_dir).as_posix())}
                for path in valid_paths
            ]

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
