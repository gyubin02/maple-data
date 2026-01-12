from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

logger = logging.getLogger("labeler")


@dataclass
class ModelConfig:
    model_id: str
    device: str
    precision: str
    max_new_tokens: int
    load_4bit: bool


class LabelerModel:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.device = _resolve_device(config.device)
        self.dtype = _resolve_dtype(config.precision, self.device)
        quantization_config = None
        load_kwargs: dict[str, object] = {}
        if config.load_4bit:
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise RuntimeError("bitsandbytes is required for 4-bit loading") from exc
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.dtype,
            )
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
        elif self.device.startswith("cuda"):
            load_kwargs["device_map"] = "auto"

        self.processor = AutoProcessor.from_pretrained(config.model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            **load_kwargs,
        )
        if not load_kwargs.get("device_map"):
            self.model.to(self.device)
        self.model.eval()

    def generate_texts(
        self,
        messages_list: list[list[dict[str, object]]],
        images: Optional[list[object]],
    ) -> list[str]:
        prompts = [
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in messages_list
        ]

        if images is None:
            inputs = self.processor(
                text=prompts,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=prompts,
                images=images,
                padding=True,
                return_tensors="pt",
            )
        inputs = _move_to_device(inputs, self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )
        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, prompt_length:]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _resolve_dtype(precision: str, device: str) -> torch.dtype:
    if precision == "fp32":
        return torch.float32
    if precision == "bf16":
        if device.startswith("cuda") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if precision == "fp16":
        return torch.float16
    if device.startswith("cuda"):
        return torch.float16
    return torch.float32


def _move_to_device(inputs: dict[str, object], device: torch.device | str) -> dict[str, object]:
    moved = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved
