#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import SiglipModel, SiglipProcessor


class CustomDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], processor: SiglipProcessor, max_length: int) -> None:
        self.records = records
        self.image_processor = processor.image_processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.records[idx]
        image_path = record["image_path"]
        label = record["label_ko"]
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        image_inputs = self.image_processor(images=image, return_tensors="pt")
        text_inputs = self.tokenizer(
            label,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )

        input_ids = text_inputs["input_ids"][0]
        if "attention_mask" in text_inputs:
            attention_mask = text_inputs["attention_mask"][0]
        else:
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            attention_mask = (input_ids != pad_id).long()

        return {
            "pixel_values": image_inputs["pixel_values"][0],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def load_records(data_file: Path, data_root: Path) -> list[dict[str, Any]]:
    text = data_file.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty data file: {data_file}")

    if text.lstrip().startswith("["):
        raw_records = json.loads(text)
    else:
        raw_records = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            raw_records.append(json.loads(line))

    records: list[dict[str, Any]] = []
    missing = 0
    for rec in raw_records:
        image_path = rec.get("image_path")
        label = rec.get("label_ko")
        if not image_path or not label:
            continue
        path = Path(image_path)
        if not path.is_absolute():
            path = (data_root / path).resolve()
        if not path.exists():
            missing += 1
            continue
        label_text = str(label).strip()
        if not label_text:
            continue
        records.append({"image_path": path, "label_ko": label_text})

    if missing:
        print(f"Skipped {missing} records with missing images.")
    if not records:
        raise ValueError("No valid records found after filtering.")
    return records


def prepare_model_and_processor(
    model_id: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> tuple[SiglipModel, SiglipProcessor]:
    processor = SiglipProcessor.from_pretrained(model_id)
    base_model = SiglipModel.from_pretrained(model_id)
    for param in base_model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model, processor


def clip_contrastive_loss(
    model: SiglipModel,
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
) -> torch.Tensor:
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    logit_scale = model.logit_scale.exp().clamp(max=100)
    logits_per_text = logit_scale * text_embeds @ image_embeds.t()
    logits_per_image = logits_per_text.t()
    labels = torch.arange(logits_per_text.size(0), device=logits_per_text.device)
    loss_t = F.cross_entropy(logits_per_text, labels)
    loss_i = F.cross_entropy(logits_per_image, labels)
    return (loss_t + loss_i) / 2


@torch.no_grad()
def evaluate(
    model: SiglipModel,
    data_loader: DataLoader,
    device: torch.device,
    autocast_context,
) -> float:
    model.eval()
    total_loss = 0.0
    steps = 0
    for batch in data_loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with autocast_context:
            image_embeds = model.get_image_features(pixel_values=batch["pixel_values"])
            text_embeds = model.get_text_features(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            loss = clip_contrastive_loss(model, image_embeds, text_embeds)
        total_loss += loss.item()
        steps += 1
    return total_loss / max(steps, 1)


@torch.no_grad()
def run_similarity_test(
    model: SiglipModel,
    processor: SiglipProcessor,
    sample: dict[str, Any],
    device: torch.device,
    autocast_context,
) -> None:
    model.eval()
    image_path = sample["image_path"]
    label = sample["label_ko"]
    queries = [label, "unrelated item icon"]

    with Image.open(image_path) as img:
        image = img.convert("RGB")

    image_inputs = processor.image_processor(images=image, return_tensors="pt")
    text_inputs = processor.tokenizer(
        queries,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=processor.tokenizer.model_max_length,
    )

    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

    with autocast_context:
        image_features = model.get_image_features(**image_inputs)
        text_features = model.get_text_features(**text_inputs)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    scores = (text_features @ image_features.T).squeeze(-1).cpu().tolist()

    print("Similarity test (higher is better):")
    for query, score in zip(queries, scores):
        print(f"- {query}: {score:.4f}")


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for KoCLIP image-text retrieval.")
    parser.add_argument("--data-file", type=Path, required=True, help="Path to JSONL or JSON data file.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.cwd(),
        help="Root directory for relative image paths.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ko-clip-lora"))
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/siglip-base-patch16-256-multilingual",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Per-device batch size (64-128 typical on 24GB; reduce if OOM).",
    )
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not 0.1 <= args.val_ratio <= 0.15:
        raise ValueError("--val-ratio must be between 0.10 and 0.15")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA not available; using fp32. bf16 requires GPU.")

    torch.backends.cuda.matmul.allow_tf32 = True

    records = load_records(args.data_file, args.data_root)
    train_records, val_records = train_test_split(
        records,
        test_size=args.val_ratio,
        random_state=args.seed,
        shuffle=True,
    )
    print(f"Loaded {len(records)} samples (train={len(train_records)}, val={len(val_records)}).")

    model, processor = prepare_model_and_processor(
        args.model_id,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model.to(device)

    max_length = processor.tokenizer.model_max_length
    train_dataset = CustomDataset(train_records, processor, max_length)
    val_dataset = CustomDataset(val_records, processor, max_length)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found. Check LoRA target_modules.")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    if device.type == "cuda":
        autocast_context = torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        autocast_context = nullcontext()

    best_val = float("inf")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with autocast_context:
                image_embeds = model.get_image_features(pixel_values=batch["pixel_values"])
                text_embeds = model.get_text_features(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                loss = clip_contrastive_loss(model, image_embeds, text_embeds)
            total_loss += loss.item()
            loss = loss / args.grad_accum_steps
            loss.backward()

            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            steps += 1

        train_loss = total_loss / max(steps, 1)
        val_loss = evaluate(model, val_loader, device, autocast_context)
        print(f"Epoch {epoch:02d} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_dir = output_dir / "best_model"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)

    if val_records:
        run_similarity_test(model, processor, val_records[0], device, autocast_context)


if __name__ == "__main__":
    main()
