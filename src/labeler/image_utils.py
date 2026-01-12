from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image


def load_image(
    path: Path,
    upscale: int,
    alpha_background: str,
) -> Image.Image:
    image = Image.open(path)
    image = _apply_alpha(image, alpha_background)
    if upscale and upscale > 1:
        image = image.resize(
            (image.width * upscale, image.height * upscale),
            resample=Image.BICUBIC,
        )
    return image


def _apply_alpha(image: Image.Image, alpha_background: str) -> Image.Image:
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        if alpha_background == "none":
            return image.convert("RGBA")
        color = (255, 255, 255, 255) if alpha_background == "white" else (0, 0, 0, 255)
        background = Image.new("RGBA", image.size, color)
        foreground = image.convert("RGBA")
        return Image.alpha_composite(background, foreground).convert("RGB")
    return image.convert("RGB")
