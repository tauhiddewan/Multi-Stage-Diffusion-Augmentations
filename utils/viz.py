# utils/vis_diffusion.py
import os
import math
import torch
from typing import Iterable, Sequence, Tuple, Optional
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid, save_image

# Lightweight tensor<->PIL helpers
_to_tensor = transforms.ToTensor()                 # [0,1] CHW
def _resize_tensor(t: torch.Tensor, size: Tuple[int,int]) -> torch.Tensor:
    return transforms.functional.resize(t, size, antialias=True)

def _pil_to_tensor_resized(pil: Image.Image, size: Tuple[int,int]) -> torch.Tensor:
    return _to_tensor(pil.resize(size, Image.BICUBIC)).clamp(0,1)

@torch.no_grad()
def _img_at_strength(
    diff_aug,
    img_t_01: torch.Tensor,          # [C,H,W] in [0,1]
    strength: float,
    *,
    steps: Optional[int] = None,
    guidance: Optional[float] = None,
    target_size: Optional[int] = None,
    seed: int = 12345,
) -> torch.Tensor:
    
    device = diff_aug.device
    pipe = diff_aug.pipe
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass

    orig_steps = diff_aug.num_inference_steps
    orig_guidance = diff_aug.guidance_scale
    orig_target = diff_aug.target_size

    if steps is not None:
        diff_aug.num_inference_steps = int(steps)
    if guidance is not None:
        diff_aug.guidance_scale = float(guidance)
    if target_size is not None:
        diff_aug.target_size = int(target_size)

    steps_eff = max(1, int(diff_aug.num_inference_steps))
    strength_eff = max(float(strength), 1.0 / steps_eff + 1e-6)

    H, W = img_t_01.shape[1], img_t_01.shape[2]
    pil_in = transforms.ToPILImage()(img_t_01.cpu())
    pil_in_resized = pil_in.resize((diff_aug.target_size, diff_aug.target_size), Image.BICUBIC)

    g = torch.Generator(device=device)
    g.manual_seed(seed)

    with torch.autocast(device if device != "cpu" else "cpu", enabled=(device!="cpu")):
        out = pipe(
            prompt=diff_aug.prompt or "",
            image=pil_in_resized,
            strength=strength_eff,
            guidance_scale=diff_aug.guidance_scale,
            num_inference_steps=steps_eff,
            generator=g,
        )
    pil_out_resized = out.images[0]
    pil_out = pil_out_resized.resize((W, H), Image.BICUBIC)
    out_t = _to_tensor(pil_out).clamp(0,1)

    diff_aug.num_inference_steps = orig_steps
    diff_aug.guidance_scale = orig_guidance
    diff_aug.target_size = orig_target
    return out_t

@torch.no_grad()
def save_stagewise_examples(
    diff_aug,
    samples: Iterable[Tuple[Image.Image, Image.Image]],  # iterable of (PIL_image, PIL_mask)
    *,
    image_size: Tuple[int,int] = (384,384),
    strengths: Sequence[float] = (0.05, 0.10, 0.15, 0.20),
    steps: Optional[int] = None,
    guidance: Optional[float] = None,
    target_size: Optional[int] = None,
    out_dir: str = "./outputs/results",
    prefix: str = "stagewise",
    max_examples: int = 4,
):
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    to_tensor_resized = lambda pil: _pil_to_tensor_resized(pil, image_size)

    for idx, (pil_img, _pil_msk) in enumerate(samples):
        if idx >= max_examples:
            break
        base_t = to_tensor_resized(pil_img)  # [C,H,W] in [0,1]
        cols = [base_t]
        # make deterministic but different per column
        for j, s in enumerate(strengths):
            out_t = _img_at_strength(
                diff_aug,
                base_t,
                strength=s,
                steps=steps,
                guidance=guidance,
                target_size=target_size,
                seed=12345 + 13 * j + 101 * idx,
            )
            cols.append(out_t)

        grid = make_grid(torch.stack(cols, dim=0), nrow=len(cols), padding=8)
        save_path = os.path.join(out_dir, f"{prefix}_ex{idx}_W{image_size[0]}xH{image_size[1]}.png")
        save_image(grid, save_path)
        saved.append(save_path)
    return saved
