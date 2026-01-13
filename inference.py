#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import SiglipModel, SiglipProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SigLIP inference with LoRA adapter.")
    parser.add_argument("--model-id", default="google/siglip-base-patch16-256-multilingual")
    parser.add_argument("--adapter-path", default="outputs/ko-clip-lora/best_model")
    parser.add_argument("--image-path", required=True, type=Path)
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=[
            "레인보우 스타",
            "블랙과 흰색의 별 모양 무기",
            "흰색 티셔츠",
            "파란색 모자",
            "관련 없는 이미지",
        ],
        help="List of text candidates (Korean recommended).",
    )
    return parser.parse_args()


def resolve_adapter_path(adapter_path: Path) -> Path:
    if (adapter_path / "adapter_config.json").exists():
        return adapter_path
    candidate = adapter_path / "best_model"
    if (candidate / "adapter_config.json").exists():
        return candidate
    return adapter_path


def main() -> None:
    args = parse_args()

    if not args.image_path.exists():
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    print("Loading model...")
    base_model = SiglipModel.from_pretrained(args.model_id)
    adapter_path = resolve_adapter_path(Path(args.adapter_path))
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    processor = SiglipProcessor.from_pretrained(args.model_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    image = Image.open(args.image_path).convert("RGB")
    image_inputs = processor(images=image, return_tensors="pt").to(device)
    text_inputs = processor(text=args.candidates, return_tensors="pt", padding=True).to(device)

    print(f"\nTarget Image: {args.image_path}")
    print("-" * 30)

    with torch.no_grad():
        image_embeds = model.get_image_features(**image_inputs)
        text_embeds = model.get_text_features(**text_inputs)

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        similarities = image_embeds @ text_embeds.t()

    for text, similarity in zip(args.candidates, similarities[0]):
        sim_value = similarity.item()
        distance = 1.0 - sim_value
        print(f"{text} Similarity: {sim_value:.4f} | Distance: {distance:.4f}")


if __name__ == "__main__":
    main()
