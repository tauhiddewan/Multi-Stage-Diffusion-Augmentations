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
            device = "cuda"
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

        torch_dtype = torch.float16 if (dtype=="fp16" and self.device=="cuda") else torch.float32

        controlnet = ControlNetModel.from_pretrained(
            pretrained_model_name_or_path=controlnet_id, 
            torch_dtype=torch_dtype, 
            use_safetensors=True
        )

        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            model_id=model_id, 
            controlnet=controlnet, 
            torch_dtype=torch_dtype, 
            safety_checker=None, 
            use_safetensors=True
        ).to(self.device)
        self.pipeline.set_progress_bar_config(disable=True)


    @torch.inference_mode()
    def __call__(self, img_tensor, msk_tensor):
        if self.alpha >= 0.999 or self.rng.random() > self.prob_value: #probability gate + alpha short-circuit
            return img_tensor  
        
        c, h, w = img_tensor.shape 

        ## building a RGB mask template
        msk_tensor_binarized = (msk_tensor > 0.5).float() 
        msk_tensor_resized  = F.interpolate(
            input=msk_tensor_binarized.unsqueeze(0), 
            size=(self.target_img_size, self.target_img_size),
            mode="nearest"
            ).squeeze(0).squeeze(0)
        
        blank_mask_map = np.zeros((self.target_img_size, self.target_img_size, 3), dtype=np.uint8)
        blank_mask_map[..., 0] = (msk_tensor_resized.detach().cpu().numpy() * 255).astype(np.uint8)
        blank_mask_map = Image.fromarray(blank_mask_map) 

        ## generating a synthetic image that matches templates layout 
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.rng.randrange(0, 2**31 - 1))
        output = self.pipeline(
            prompt=self.prompt, 
            negative_prompt=self.neg_prompt, 
            control_image=blank_mask_map, 
            height=self.target_img_size, 
            width=self.target_img_size, 
            guidance_scale=self.guidance_scale, 
            controlnet_conditioning_scale=self.condn_scale, 
            num_inference_steps=self.num_inference_steps, 
            generator=generator
        )
        diff_aug_img = output.images[0] #first generated image

        ## mixing 
        if diff_aug_img.size != (w, h):
            diff_aug_img = diff_aug_img.resize(
                size=(w, h),
                resample=Image.BICUBIC
            )
        diff_aug_img = pil_to_tensor(diff_aug_img).clamp(0, 1)  #Blending can sometimes push values slightly outside due to float precision

        infused_img = self.alpha * img_tensor + (1-self.alpha)*diff_aug_img
        return infused_img.clamp(0, 1)

