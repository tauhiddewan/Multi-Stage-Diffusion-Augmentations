# utils/diffusion_aug.py
from dataclasses import dataclass
from typing import Optional, Tuple
import random
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

def _tensor_to_pil(img_t: torch.Tensor) -> Image.Image:
    # img_t: [C,H,W] in [0,1]
    x = img_t.detach().clamp(0,1).mul(255).byte().permute(1,2,0).cpu().numpy()
    if x.ndim == 2:
        x = np.repeat(x[...,None], 3, axis=2)
    return Image.fromarray(x)

def _pil_to_tensor(img_pil: Image.Image) -> torch.Tensor:
    arr = np.array(img_pil)
    if arr.ndim == 2:
        arr = np.repeat(arr[...,None], 3, axis=2)
    t = torch.from_numpy(arr).permute(2,0,1).float() / 255.0
    return t

@dataclass
class Stage:
    start_epoch: int
    strength: float 

class DiffusionImg2ImgAug:
    def __init__(
        self,
        model_id: str = "stabilityai/sd-turbo",  # or "runwayml/stable-diffusion-v1-5"
        stages: Tuple[Stage, ...] = (
            Stage(0,  0.03),
            Stage(10, 0.07),
            Stage(25, 0.12),
            Stage(40, 0.18),
        ),
        guidance_scale: float = 1.5,
        num_inference_steps: int = 6,
        p: float = 0.5,
        target_size: int = 512,
        prompt: Optional[str] = "",
        seed: Optional[int] = 123,
        dtype: str = "fp16",
        device: Optional[str] = None,
    ):
        self.stages = tuple(sorted(stages, key=lambda s: s.start_epoch))
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.p = float(p)
        self.prompt = prompt or ""
        self.target_size = int(target_size)
        self._epoch = 0
        self.rng = random.Random(seed)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16 if (dtype == "fp16" and self.device == "cuda") else torch.float32

        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
            self.pipe.enable_vae_tiling()

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def _current_strength(self) -> float:
        s = self.stages[0].strength
        for st in self.stages:
            if self._epoch >= st.start_epoch:
                s = st.strength
            else:
                break
        return float(np.clip(s, 0.0, 1.0))

    def __call__(self, img_t: torch.Tensor) -> torch.Tensor:
        if self.rng.random() > self.p:
            return img_t

        strength = self._current_strength()

        pil = _tensor_to_pil(img_t)
        ow, oh = pil.size
        pil_resized = pil.resize((self.target_size, self.target_size), Image.BICUBIC)

        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.rng.randrange(0, 2**31 - 1))

        with torch.autocast(self.device if self.device != "cpu" else "cpu", enabled=(self.device!="cpu")):
            out = self.pipe(
                prompt=self.prompt,
                image=pil_resized,
                strength=strength,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                generator=gen,
            )
        aug = out.images[0].resize((ow, oh), Image.BICUBIC)
        return _pil_to_tensor(aug).clamp(0,1)
