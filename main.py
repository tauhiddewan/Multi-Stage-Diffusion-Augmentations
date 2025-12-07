import os
import ast
import json
import math
import pickle
import torch
import warnings
import logging
from glob import glob
from PIL import Image
from dotenv import dotenv_values
from torch.utils.data import DataLoader
from torch.optim import AdamW

from utils.dataset import DiffusedDataset
from utils.diff_aug import ControlNetAug
from utils.model import select_model, count_params
from utils.loss import select_criterion
from utils.train import training_loop, test_loop, get_lr_scheduler
from utils.plots import save_training_curves
from utils.misc import create_logger
from utils.viz import save_diff_example, save_diff_batch
from diffusers.utils import logging as diffusers_logging

from diffusers.utils import logging as diffusers_logging
diffusers_logging.disable_progress_bar()
logging.getLogger("diffusers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")


def alpha_curriculum(epoch, T, a0=1.0, a1=0.8):
    if T <= 1:
        return a1
    t = epoch / (T - 1)
    return a0 * (1 - t) + a1 * t


def str2bool(x):
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    return str(x).lower() in {"1", "true", "yes", "y"}


def main():
    env_vars = dotenv_values(dotenv_path="./.env")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = create_logger(log_filename="controlnet_alpha_blend", env_vars=env_vars)

    # --------------------------
    # ENV VARIABLES
    # --------------------------
    aug_mode = str(env_vars.get("aug_mode", "none"))

    batch_size = int(env_vars.get("batch_size", 4))
    num_epochs = int(env_vars.get("num_epochs", 100))
    model_name = str(env_vars.get("model_name", "segformer"))
    model_config = str(env_vars.get("model_config", "b0"))
    learning_rate = float(env_vars.get("learning_rate", 1e-4))
    image_size = ast.literal_eval(env_vars.get("image_size", "(384, 384)"))
    mask_size = ast.literal_eval(env_vars.get("mask_size", "(96, 96)"))
    cnvrg_threshold = float(env_vars.get("threshold", 0.005))
    ma_window = int(env_vars.get("ma_window", 10))
    max_epochs = int(env_vars.get("max_epochs", 100))
    min_epochs = int(env_vars.get("min_epochs", 20))

    use_scheduler = str2bool(env_vars.get("use_scheduler", "False"))
    save_model = str2bool(env_vars.get("save_model", "True"))

    data_folder_path = str(env_vars.get("data_folder_path", "./data"))
    output_path = env_vars.get("output_folder_path", "./outputs")
    os.makedirs(output_path, exist_ok=True)

    pickle_filename = str(env_vars.get("dataset_pkl_fname", "dataset.pkl"))

    # --------------------------
    # LOAD SPLITS
    # --------------------------
    pkl_path = os.path.join(data_folder_path, pickle_filename)
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Make this robust to either naming scheme:
    #   - "test_data_id"/"test_data_ood1"/"test_data_ood2"
    #   - or "test_data"/"test_ood1"/"test_ood2"
    train_data = data["train_data"]
    val_data = data["val_data"]
    test_data_id = data["test_data_id"]
    test_data_ood1 = data["test_data_ood1"]
    test_data_ood2 = data["test_data_ood2"]

    # --------------------------
    # DIFFUSION AUG
    # --------------------------
    diff_aug = ControlNetAug(
        alpha=float(env_vars.get("start_alpha", 0.5)),
        prob_value=float(env_vars.get("prob_value", 1.0)),
        model_id=str(env_vars.get("model_id")),
        controlnet_id=env_vars.get("controlnet_id"),
        prompt=str(env_vars.get("prompt", "")),
        neg_prompt=str(env_vars.get("neg_prompt", "")),
        guidance_scale=float(env_vars.get("guidance_scale", 7.5)),
        condn_scale=float(env_vars.get("condn_scale", 1.2)),
        num_inference_steps=int(env_vars.get("num_inference_steps", 20)),
        target_img_size=image_size[0],
        random_seed=123,
        dtype="fp16",
        device=device,
        alpha_schedule=alpha_curriculum,
        total_epochs=num_epochs
    )

    # Sanity-check a few diffusion examples (OPTIONAL)
    # samples = []
    # for i in range(min(6, len(train_data))):
    #     img_path, mask_path = train_data[i]
    #     img_pil = Image.open(img_path).convert("RGB")
    #     mask_pil = Image.open(mask_path).convert("L")
    #     samples.append((img_pil, mask_pil))

    # _ = save_diff_batch(
    #     diff_aug,
    #     samples=samples,
    #     image_size=image_size,
    #     out_dir=output_path,
    #     prefix="diff",
    # )
    # --------------------------
    # DATASETS & LOADERS
    # --------------------------
    train_ds = DiffusedDataset(
        data=train_data,
        mode="train",
        aug_mode=aug_mode,
        image_size=image_size,
        mask_size=mask_size,
        diff_aug=diff_aug,
    )

    valid_ds = DiffusedDataset(
        data=val_data,
        mode="val",
        aug_mode=aug_mode,
        image_size=image_size,
        mask_size=mask_size,
        diff_aug=None,
    )

    test_ds_id = DiffusedDataset(
        data=test_data_id,
        mode="test",
        aug_mode=aug_mode,
        image_size=image_size,
        mask_size=mask_size,
        diff_aug=None,
    )

    test_ds_ood1 = DiffusedDataset(
        data=test_data_ood1,
        mode="test",
        aug_mode=aug_mode,
        image_size=image_size,
        mask_size=mask_size,
        diff_aug=None,
    )

    test_ds_ood2 = DiffusedDataset(
        data=test_data_ood2,
        mode="test",
        aug_mode=aug_mode,
        image_size=image_size,
        mask_size=mask_size,
        diff_aug=None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader_id = DataLoader(
        test_ds_id,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader_ood1 = DataLoader(
        test_ds_ood1,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader_ood2 = DataLoader(
        test_ds_ood2,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # --------------------------
    # MODEL, LOSS, OPTIMIZER, SCHEDULER
    # --------------------------
    model = select_model(model_name, model_config)
    logger.info(
        f"Model: {model_name}-{model_config} | Params: {count_params(model):.2f}M"
    )

    criterion = select_criterion(model_name)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    scheduler = get_lr_scheduler(
        optimizer,
        num_epochs=num_epochs,
        warmup_epochs=5,
        min_lr=1e-6,
    )

    best_model_save_path = os.path.join(output_path, f"{model_name}_{model_config}")

    # --------------------------
    # TRAINING (with validation using test_loop)
    # --------------------------
    best_model, losses, ma_losses, dice_scores, miou_scores = training_loop(
        dataloader=train_loader,
        model=model,
        model_name=model_name,
        train_data_size=len(train_loader.dataset),
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        threshold=cnvrg_threshold,  # convergence tolerance
        ma_window=ma_window,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        best_model_save_path=best_model_save_path,
        logger=logger,
        use_scheduler=use_scheduler,
        save_model=save_model,
        val_dataloader=valid_loader
    )

    paths = save_training_curves(
        losses=losses,
        ma_losses=ma_losses,
        dice_scores=dice_scores,
        miou_scores=miou_scores,
        out_dir=output_path,
        prefix="train",
    )
    logger.info(f"Saved training curves: {paths}")

    # --------------------------
    # TESTING (ID + OOD)
    # --------------------------
    test_loss_id, test_dice_id, test_iou_id = test_loop(
        model=best_model,
        model_name=model_name,
        test_dataloader=test_loader_id,
        criterion=criterion,
        device=device,
        num_repeat=1,
    )

    test_loss_ood1, test_dice_ood1, test_iou_ood1 = test_loop(
        model=best_model,
        model_name=model_name,
        test_dataloader=test_loader_ood1,
        criterion=criterion,
        device=device,
        num_repeat=1,
    )

    test_loss_ood2, test_dice_ood2, test_iou_ood2 = test_loop(
        model=best_model,
        model_name=model_name,
        test_dataloader=test_loader_ood2,
        criterion=criterion,
        device=device,
        num_repeat=1,
    )

    logger.info(f"Test (ID)   - Loss: {test_loss_id:.4f}, Dice: {test_dice_id:.4f}, IoU: {test_iou_id:.4f}")
    logger.info(f"Test (OOD1) - Loss: {test_loss_ood1:.4f}, Dice: {test_dice_ood1:.4f}, IoU: {test_iou_ood1:.4f}")
    logger.info(f"Test (OOD2) - Loss: {test_loss_ood2:.4f}, Dice: {test_dice_ood2:.4f}, IoU: {test_iou_ood2:.4f}")

    results = {
        "in_dstbn": {
            "test_loss": test_loss_id,
            "test_dice": test_dice_id,
            "test_iou": test_iou_id,
        },
        "ood1": {
            "test_loss": test_loss_ood1,
            "test_dice": test_dice_ood1,
            "test_iou": test_iou_ood1,
        },
        "ood2": {
            "test_loss": test_loss_ood2,
            "test_dice": test_dice_ood2,
            "test_iou": test_iou_ood2,
        },
    }

    with open(os.path.join(output_path, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
