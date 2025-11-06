# utils/vis_simple.py
import os
import torch
from typing import Iterable, Tuple
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid, save_image

_to_tensor = transforms.ToTensor()

def _img_tensor(pil: Image.Image, size: Tuple[int,int]) -> torch.Tensor:
    # [3,H,W] in [0,1]
    return _to_tensor(pil.resize(size, Image.BICUBIC)).clamp(0, 1)

def _mask_tensor(pil_mask: Image.Image, size: Tuple[int,int]) -> torch.Tensor:
    # [1,H,W] in [0,1] (nearest to keep hard edges)
    t = _to_tensor(pil_mask.resize(size, Image.NEAREST)).clamp(0, 1)
    if t.ndim == 3 and t.shape[0] > 1:   # if mask came as 3ch, keep first
        t = t[:1]
    return t

@torch.no_grad()
def save_diff_example(
    diff_aug,                          # your ControlNetAug
    pil_img: Image.Image,
    pil_mask: Image.Image,
    image_size: Tuple[int,int] = (384,384),
    out_path: str = "./outputs/diff_example.png",
) -> str:
    """Saves a 3-column grid: [original | mask | infused]."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    img_t = _img_tensor(pil_img, image_size)         # [3,H,W]
    msk_t = _mask_tensor(pil_mask, image_size)       # [1,H,W]

    # Force a single visualization regardless of prob gate
    orig_prob, orig_alpha = getattr(diff_aug, "prob_value", 1.0), getattr(diff_aug, "alpha", 0.9)
    try:
        if hasattr(diff_aug, "prob_value"):
            diff_aug.prob_value = 1.0
        infused = diff_aug(img_t, msk_t)             # [3,H,W] in [0,1]
    finally:
        if hasattr(diff_aug, "prob_value"):
            diff_aug.prob_value = orig_prob
        if hasattr(diff_aug, "alpha"):
            diff_aug.alpha = orig_alpha

    # Make a 3-channel mask image for display
    mask_rgb = msk_t.expand(3, *msk_t.shape[-2:])    # [3,H,W]

    grid = make_grid([img_t, mask_rgb, infused], nrow=3, padding=8)
    save_image(grid, out_path)
    return out_path

@torch.no_grad()
def save_diff_batch(
    diff_aug,
    samples: Iterable[Tuple[Image.Image, Image.Image]],  # (img_pil, mask_pil)
    image_size: Tuple[int,int] = (384,384),
    out_dir: str = "./outputs/diff_viz",
    prefix: str = "viz",
    max_examples: int = 8,
) -> list[str]:
    """Saves one grid per sample: [original | mask | infused]."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, (img_pil, mask_pil) in enumerate(samples):
        if i >= max_examples:
            break
        p = os.path.join(out_dir, f"{prefix}_{i:03d}.png")
        paths.append(save_diff_example(diff_aug, img_pil, mask_pil, image_size, p))
    return paths
