import os
import ast
import pickle
import random

import torch
import numpy as np
from PIL import Image
from dotenv import dotenv_values

from utils.diff_aug import ControlNetAug
from utils.viz import save_alpha_sweep_two_rows


def load_env_and_data():
    """Load env, device, dataset and basic paths once."""
    env = dotenv_values(dotenv_path="./.env")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sizes from env (strings -> tuples)
    image_size = ast.literal_eval(env.get("image_size", "(384, 384)"))
    mask_size = ast.literal_eval(env.get("mask_size", "(96, 96)"))  # unused here but kept

    data_folder_path = env.get("data_folder_path", "./data")
    pickle_filename = env.get("dataset_pkl_fname", "dataset.pkl")
    output_path = env.get("output_folder_path", "./outputs")
    os.makedirs(output_path, exist_ok=True)

    # Load dataset
    pkl_path = os.path.join(data_folder_path, pickle_filename)
    print(f"Loading dataset pickle from: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if "train_data" not in data:
        raise KeyError(f"'train_data' key not found in {pkl_path}. Found: {list(data.keys())}")

    train_data = data["train_data"]  # list of (img_path, mask_path)
    print(f"Found {len(train_data)} training samples")

    # Fine-tuned SD path
    finetuned_model_dir = env.get(
        "finetuned_model_id",
        "./sd_finetuned/sd15_kvasir_finetuned",
    )
    if not os.path.isdir(finetuned_model_dir):
        raise FileNotFoundError(
            f"Fine-tuned model directory not found: {finetuned_model_dir}\n"
            "Make sure finetune_sd1.5.py finished and saved there, or update the path."
        )

    print(f"Using fine-tuned SD model from: {finetuned_model_dir}")

    return env, device, image_size, train_data, output_path, finetuned_model_dir


def generate_alpha_sweep(
    img_pil: Image.Image,
    mask_pil: Image.Image,
    env,
    device,
    image_size,
    output_path: str,
    finetuned_model_dir: str,
):
    """
    Build a ControlNetAug once, then generate a 2-row alpha sweep:
        Row 1: real image repeated
        Row 2: generated images for different alphas
    """
    print("\n=== Generating curriculum noise alpha sweep ===")

    diff_aug = ControlNetAug(
        alpha=float(env.get("alpha", 0.95)),   # initial alpha, will be overridden inside sweep
        prob_value=float(env.get("prob_value", 0.4)),
        model_id=finetuned_model_dir,
        controlnet_id=env.get("controlnet_id", "lllyasviel/control_v11p_sd15_seg"),
        prompt=env.get(
            "prompt",
            "colonoscopy image, natural endoscopic view, realistic mucosal texture, minimal lighting changes",
        ),
        neg_prompt=env.get(
            "neg_prompt",
            "tiles, mosaic, grid, patterns, texture map, letters, symbols, unrealistic, synthetic, cartoon, illustration, distortion, artifacts",
        ),
        guidance_scale=float(env.get("guidance_scale", 7.5)),
        condn_scale=float(env.get("condn_scale", 1.2)),
        num_inference_steps=int(env.get("num_inference_steps", 20)),
        target_img_size=image_size[0],
        random_seed=int(env.get("random_seed", 123)),
        dtype=env.get("dtype", "fp16"),
        device=device,
        alpha_schedule=None,
        total_epochs=None,
    )

    # 25 alphas from 0.0 → 1.0
    alphas = [float(a) for a in np.linspace(0.0, 1.0, 10+1)]

    viz_dir = os.path.join(output_path, "diff_aug_sample_images")
    os.makedirs(viz_dir, exist_ok=True)

    out_path = os.path.join(viz_dir, "curriculum_noise.png")  # use PNG; PDF can be flaky
    print(f"Saving 2-row alpha sweep to: {out_path}")

    save_alpha_sweep_two_rows(
        diff_aug=diff_aug,
        pil_img=img_pil,
        pil_mask=mask_pil,
        alphas=alphas,
        image_size=tuple(image_size),
        out_path=out_path,
    )

    print("Done.")


def main():
    env, device, image_size, train_data, output_path, finetuned_model_dir = load_env_and_data()

    # Choose a single sample image (random, or fix index via env)
    idx = int(env.get("viz_sample_index", -1))
    if idx < 0 or idx >= len(train_data):
        idx = random.randint(0, len(train_data) - 1)

    img_path, mask_path = train_data[idx]
    print(f"\nUsing sample index {idx}:")
    print(f"  image: {img_path}")
    print(f"  mask : {mask_path}")

    img_pil = Image.open(img_path).convert("RGB")
    mask_pil = Image.open(mask_path).convert("L")

    generate_alpha_sweep(
        img_pil=img_pil,
        mask_pil=mask_pil,
        env=env,
        device=device,
        image_size=image_size,
        output_path=output_path,
        finetuned_model_dir=finetuned_model_dir,
    )


if __name__ == "__main__":
    main()
