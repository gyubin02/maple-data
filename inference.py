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
            "하얀 모자",
            "눈",
            "관련 없는 이미지",
        ],
        help="List of text candidates (Korean recommended).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.image_path.exists():
        raise FileNotFoundError(f"Image not found: {args.image_path}")

    print("Loading model...")
    base_model = SiglipModel.from_pretrained(args.model_id)
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
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

        logits = image_embeds @ text_embeds.t()
        logit_scale = model.logit_scale.exp()
        logits = logits * logit_scale
        probs = logits.softmax(dim=1)

    for text, prob in zip(args.candidates, probs[0]):
        print(f"{text}: {prob.item() * 100:.2f}%")


if __name__ == "__main__":
    main()
