import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline


def pil_to_tensor(img):
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    return torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0


class ControlNetAug:
    def __init__(
            self, 
            alpha, 
            prob_value, 
            model_id = "runwayml/stable-diffusion-v1-5", 
            controlnet_id = "lllyasviel/control_v11p_sd15_seg", 
            prompt = "endoscopy image, realistic lighting", 
            neg_prompt = None, 
            guidance_scale = 3.5, 
            condn_scale = 1, 
            num_inference_steps = 20, 
            target_img_size = 384, 
            random_seed = 123, 
            dtype = "fp16", 
            device = "cuda", 
            alpha_schedule=None, 
            total_epochs=None
    ):
        self.alpha = alpha 
        self.prob_value = prob_value
        self.model_id = model_id
        self.controlnet_id = controlnet_id
        self.prompt = prompt
        self.neg_prompt = neg_prompt
        self.guidance_scale = guidance_scale
        self.condn_scale = condn_scale
        self.num_inference_steps = num_inference_steps
        self.target_img_size = target_img_size
        self.rng = random.Random(random_seed)
        self.dtype = dtype
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        ### Curriculum settings
        self.alpha_schedule = alpha_schedule
        self.total_epochs = total_epochs

        torch_dtype = torch.float16 if (dtype=="fp16" and self.device=="cuda") else torch.float32
        controlnet = ControlNetModel.from_pretrained(
            pretrained_model_name_or_path=controlnet_id, 
            torch_dtype=torch_dtype, 
            use_safetensors=True
        )

        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            pretrained_model_name_or_path=model_id, 
            controlnet=controlnet, 
            torch_dtype=torch_dtype, 
            safety_checker=None, 
            use_safetensors=True
        ).to(self.device)
        self.pipeline.set_progress_bar_config(disable=True)


    def set_epoch(self, epoch: int):
        if self.alpha_schedule is not None:
            T = self.total_epochs or 1
            e = max(0, min(epoch, T - 1))
            self.alpha = float(self.alpha_schedule(e, T))

    
    @torch.inference_mode()
    def __call__(self, img_tensor, msk_tensor):
        if self.alpha >= 0.999 or self.rng.random() > self.prob_value:
            return img_tensor

        c, h, w = img_tensor.shape

        # -----------------------------
        # 1) Prepare binary masks
        # -----------------------------
        # Full-resolution binary mask (for blending)
        msk_full = (msk_tensor > 0.5).float()          # [1,H,W]

        # Downsampled / resized mask for ControlNet template
        msk_resized = F.interpolate(
            input=msk_full.unsqueeze(0),               # [1,1,H,W]
            size=(self.target_img_size, self.target_img_size),
            mode="nearest"
        ).squeeze(0).squeeze(0)                        # [Hc,Wc]

        # Build 3-channel control image (mask in red channel)
        blank_mask_map = np.zeros(
            (self.target_img_size, self.target_img_size, 3),
            dtype=np.uint8
        )
        blank_mask_map[..., 0] = (
            msk_resized.detach().cpu().numpy() * 255
        ).astype(np.uint8)
        blank_mask_map = Image.fromarray(blank_mask_map)

        # -----------------------------
        # 2) Run diffusion (ControlNet)
        # -----------------------------
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.rng.randrange(0, 2**31 - 1))

        output = self.pipeline(
            prompt=self.prompt,
            negative_prompt=self.neg_prompt,
            image=blank_mask_map,
            height=self.target_img_size,
            width=self.target_img_size,
            guidance_scale=self.guidance_scale,
            controlnet_conditioning_scale=self.condn_scale,
            num_inference_steps=self.num_inference_steps,
            generator=generator
        )

        diff_aug_img = output.images[0]  # PIL

        # Resize to match original tensor spatial size
        if diff_aug_img.size != (w, h):
            diff_aug_img = diff_aug_img.resize(
                size=(w, h),
                resample=Image.BICUBIC
            )

        diff_aug_img = pil_to_tensor(diff_aug_img).clamp(0, 1)  # [3,H,W]

        # -----------------------------
        # 3) Background-only blending
        # -----------------------------
        # Ensure mask size matches tensor size
        if msk_full.shape[-2:] != (h, w):
            msk_full = F.interpolate(
                msk_full.unsqueeze(0), size=(h, w), mode="nearest"
            ).squeeze(0)

        msk_full = msk_full.to(img_tensor.device)      # [1,H,W]
        msk_full_3 = msk_full.expand_as(img_tensor)    # [3,H,W]
        bg_mask_3 = 1.0 - msk_full_3                   # [3,H,W]

        # Blend ONLY in the background:
        #   - inside polyp: keep original image
        #   - outside polyp: alpha-blend original and diffused
        bg_blended = self.alpha * img_tensor + (1.0 - self.alpha) * diff_aug_img
        infused_img = msk_full_3 * img_tensor + bg_mask_3 * bg_blended

        return infused_img.clamp(0, 1)

