#!/usr/bin/env python

import os
import math
import argparse
import pickle
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor


# -----------------------------
# Config dataclass
# -----------------------------
@dataclass
class TrainConfig:
    pickle_path: str = "./data/dataset.pkl"             # your pickle file
    output_dir: str = "./kvasir_lora_weights"

    model_id: str = "runwayml/stable-diffusion-v1-5"
    resolution: int = 512
    train_batch_size: int = 4
    num_epochs: int = 10
    learning_rate: float = 1e-4
    mixed_precision: str = "fp16"                  # "no", "fp16", "bf16"
    gradient_accumulation_steps: int = 1
    seed: int = 123

    lora_rank: int = 4                             # LoRA rank
    text_prompt: str = "colonoscopy image of colon mucosa with possible polyp"


# -----------------------------
# Dataset using pickle train_data
# -----------------------------
class KvasirLoRADataset(Dataset):
    """
    Expects a list of (img_path, mask_path) pairs.
    Uses only img_path for LoRA training.
    """

    def __init__(self, train_pairs, resolution=512, prompt="colonoscopy image"):
        self.train_pairs = train_pairs
        self.prompt = prompt

        self.transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
                transforms.CenterCrop((resolution, resolution)),
                transforms.ToTensor(),
                # SD expects [-1, 1]
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.train_pairs)

    def __getitem__(self, idx):
        img_path, _ = self.train_pairs[idx]  # ignore mask path
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return {"pixel_values": img, "prompt": self.prompt}


# -----------------------------
# Create proper LoRA processors for UNet
# -----------------------------
def create_lora_attn_procs(unet, rank=4):
    """
    Create LoRA attention processors for all attention modules in the UNet,
    following the pattern used by diffusers examples.
    """
    attn_procs = {}
    for name in unet.attn_processors.keys():
        # figure out the hidden_size for this block
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name.split(".")[1])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            # shouldn't really happen for SD1.5 UNet
            continue

        # cross-attention dim: only for attn2 (context) branches
        if name.endswith("attn2.processor"):
            cross_attention_dim = unet.config.cross_attention_dim
        else:
            cross_attention_dim = None

        attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank,
        )

    return attn_procs


# -----------------------------
# Training loop
# -----------------------------
def train(config: TrainConfig):
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
    )
    device = accelerator.device

    if accelerator.is_local_main_process:
        os.makedirs(config.output_dir, exist_ok=True)

    # Seed
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # -----------------------------
    # Load pickle & extract train_data
    # -----------------------------
    if accelerator.is_local_main_process:
        print(f"Loading dataset pickle from: {config.pickle_path}")
    with open(config.pickle_path, "rb") as f:
        data = pickle.load(f)

    # Expecting keys like: "train_data", "val_data", "test_data_id", ...
    if "train_data" not in data:
        raise KeyError("Expected key 'train_data' in pickle. Found keys: "
                       f"{list(data.keys())}")

    train_pairs = data["train_data"]  # list of (img_path, mask_path)

    if accelerator.is_local_main_process:
        print(f"Found {len(train_pairs)} training samples in train_data")

    dataset = KvasirLoRADataset(
        train_pairs=train_pairs,
        resolution=config.resolution,
        prompt=config.text_prompt,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # -----------------------------
    # Load base SD1.5 pipeline
    # -----------------------------
    if accelerator.is_local_main_process:
        print(f"Loading base model: {config.model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        config.model_id,
        torch_dtype=torch.float16 if config.mixed_precision == "fp16" else torch.float32,
        safety_checker=None,
    )
    # use DDPM scheduler for training
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet

    # Freeze everything except LoRA layers
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # -----------------------------
    # Create and attach LoRA processors
    # -----------------------------
    if accelerator.is_local_main_process:
        print("Creating LoRA attention processors...")
    lora_attn_procs = create_lora_attn_procs(unet, rank=config.lora_rank)
    unet.set_attn_processor(lora_attn_procs)

    # Wrap them in AttnProcsLayers so we can optimize them easily
    lora_layers = AttnProcsLayers(unet.attn_processors)
    lora_layers.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(lora_layers.parameters(), lr=config.learning_rate)

    # Prepare with accelerate
    lora_layers, optimizer, dataloader = accelerator.prepare(
        lora_layers, optimizer, dataloader
    )

    num_update_steps_per_epoch = math.ceil(len(dataloader) / 1)
    max_train_steps = config.num_epochs * num_update_steps_per_epoch

    if accelerator.is_local_main_process:
        print(f"Training for {config.num_epochs} epochs "
              f"({max_train_steps} steps total approx.)")

    global_step = 0
    for epoch in range(config.num_epochs):
        if accelerator.is_local_main_process:
            print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        unet.train()

        prog_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        for step, batch in enumerate(prog_bar):
            with accelerator.accumulate(lora_layers):
                # 1) Encode text
                prompts = batch["prompt"]
                tokenized = tokenizer(
                    list(prompts),
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = tokenized.input_ids.to(device)

                with torch.no_grad():
                    encoder_hidden_states = text_encoder(input_ids)[0]

                # 2) Image -> latent
                pixel_values = batch["pixel_values"].to(device)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # 3) Sample noise and timesteps
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, pipe.scheduler.num_train_timesteps,
                    (latents.shape[0],),
                    device=device,
                    dtype=torch.long,
                )

                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

                # 4) Predict noise with UNet (with LoRA)
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                # 5) Loss: MSE between predicted noise and true noise
                loss = torch.nn.functional.mse_loss(model_pred, noise)
                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

                if accelerator.is_local_main_process:
                    prog_bar.set_postfix({"loss": loss.item()})

        if accelerator.is_local_main_process:
            print(f"Epoch {epoch+1} done. Last loss: {loss.item():.4f}")

    # -----------------------------
    # Save LoRA weights
    # -----------------------------
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        # CPU copy of LoRA layers
        lora_state_dict = accelerator.unwrap_model(lora_layers).state_dict()
        save_path = os.path.join(config.output_dir, "kvasir_lora.pt")
        torch.save(lora_state_dict, save_path)
        print(f"\nSaved LoRA weights to: {save_path}")


# -----------------------------
# CLI entry point
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_path", type=str, default="./dataset.pkl",
                        help="Path to your dataset pickle.")
    parser.add_argument("--output_dir", type=str, default="./kvasir_lora_weights")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument(
        "--text_prompt",
        type=str,
        default="colonoscopy image of colon mucosa with possible polyp",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = TrainConfig(
        pickle_path=args.pickle_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        train_batch_size=args.train_batch_size,
        learning_rate=args.learning_rate,
        resolution=args.resolution,
        lora_rank=args.lora_rank,
        text_prompt=args.text_prompt,
    )
    train(cfg)
