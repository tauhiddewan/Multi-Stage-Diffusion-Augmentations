#!/usr/bin/env python

import os
import math
import argparse
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

from diffusers import StableDiffusionPipeline, DDPMScheduler


# -----------------------------
# Dataset from your pickle
# -----------------------------
class KvasirSDDataset(Dataset):
    """
    Expects train_pairs as a list of (img_path, mask_path).
    Uses only img_path for SD fine-tuning.
    """

    def __init__(self, train_pairs, resolution=512, prompt=None):
        self.train_pairs = train_pairs
        self.prompt = prompt or "colonoscopy image of colon mucosa with possible polyp"

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
        img_path, _ = self.train_pairs[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return {"pixel_values": img, "prompt": self.prompt}


# -----------------------------
# Training function
# -----------------------------
def train(
    pickle_path: str,
    output_dir: str,
    model_id: str = "runwayml/stable-diffusion-v1-5",
    resolution: int = 512,
    train_batch_size: int = 1,
    num_epochs: int = 5,
    learning_rate: float = 1e-5,
    seed: int = 123,
):
    os.makedirs(output_dir, exist_ok=True)

    # ---- Seed & device ----
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load pickle & train_data ----
    print(f"Loading dataset pickle from: {pickle_path}")
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    if "train_data" not in data:
        raise KeyError(
            f"Expected key 'train_data' in pickle. Found keys: {list(data.keys())}"
        )

    train_pairs = data["train_data"]  # list of (img_path, mask_path)
    print(f"Found {len(train_pairs)} training samples in train_data")

    dataset = KvasirSDDataset(
        train_pairs=train_pairs,
        resolution=resolution,
        prompt="colonoscopy image of colon mucosa with possible polyp",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # ---- Load base SD1.5 pipeline in FP32 ----
    print(f"Loading base model: {model_id}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,   # keep everything in fp32 to avoid dtype issues
        safety_checker=None,
    )
    # Use DDPM scheduler for training
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet

    # Freeze VAE & text encoder; train UNet only
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(True)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()

    num_update_steps_per_epoch = math.ceil(len(dataloader))
    max_train_steps = num_epochs * num_update_steps_per_epoch
    print(f"Training for {num_epochs} epochs (~{max_train_steps} steps)")

    global_step = 0
    unet.train()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        pbar = tqdm(dataloader)
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device=device, dtype=torch.float32)

            # 1) Encode text prompts
            prompts = batch["prompt"]
            tokens = tokenizer(
                list(prompts),
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            )
            input_ids = tokens.input_ids.to(device)

            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]

            # 2) Encode images to latents
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # 3) Sample noise & timesteps
            noise = torch.randn_like(latents, device=device)
            timesteps = torch.randint(
                0,
                pipe.scheduler.config.num_train_timesteps,  # use config to avoid warning
                (latents.shape[0],),
                device=device,
                dtype=torch.long,
            )

            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            optimizer.zero_grad(set_to_none=True)

            # 4) Predict noise with UNet (all fp32)
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
            ).sample
            loss = mse_loss(model_pred, noise)

            # 5) Backward & step
            loss.backward()
            optimizer.step()

            global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        print(f"Epoch {epoch + 1} finished. Last loss: {loss.item():.4f}")

    # ---- Save fine-tuned pipeline ----
    save_dir = os.path.join(output_dir, "sd15_kvasir_finetuned")
    print(f"\nSaving fine-tuned model to: {save_dir}")
    pipe.save_pretrained(save_dir)
    print("Done.")


# -----------------------------
# CLI entry
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_path", type=str, default="./data/dataset.pkl",
                        help="Path to your dataset pickle.")
    parser.add_argument("--output_dir", type=str, default="./sd_finetuned")
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        pickle_path=args.pickle_path,
        output_dir=args.output_dir,
        model_id=args.model_id,
        resolution=args.resolution,
        train_batch_size=args.train_batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
    )
