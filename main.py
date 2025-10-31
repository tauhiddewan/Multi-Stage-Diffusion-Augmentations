import os
import torch
from glob import glob
from PIL import Image
from dotenv import dotenv_values
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from utils.dataset import DiffusedKvasirDataset
from utils.dif_aug import DiffusionImg2ImgAug, Stage
from utils.model import select_model, count_params
from utils.loss import select_criterion
from utils.train import training_loop, test_loop, get_lr_scheduler
from utils.misc import create_logger



def main():
    # --- Environment and device ---
    env_vars = dotenv_values(dotenv_path="./.env")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = create_logger(log_filename="curriculam_diff", env_vars=env_vars)
    
    root = env_vars.get("data_folder_path", "./data")
    img_files = sorted(glob(os.path.join(root, "images", "*.jpg")))
    msk_files = sorted(glob(os.path.join(root, "masks", "*.jpg")))
    data = [(Image.open(i).convert("RGB"), Image.open(m).convert("L"))
            for i, m in zip(img_files, msk_files)]
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_data, val_data = random_split(data, [train_size, val_size])

    image_size = (384, 384)
    mask_size = (96, 96)
    output_path = env_vars.get("output_folder_path", "./outputs" )

    # diff_aug = DiffusionImg2ImgAug(
    #     model_id="runwayml/stable-diffusion-v1-5",
    #     stages=(
    #         Stage(0,  0.03),
    #         Stage(10, 0.07),
    #         Stage(25, 0.12),
    #         Stage(40, 0.18),
    #     ),
    #     guidance_scale=2.0,
    #     num_inference_steps=15,
    #     p=0.6,
    #     target_size=512,
    #     prompt="",       # try "endoscopy image" if you want style bias
    #     seed=123,
    #     dtype="fp16",
    # )

    diff_aug = DiffusionImg2ImgAug(
        model_id="stabilityai/sd-turbo",
        stages=(
            Stage(0,  0.28),
            Stage(10, 0.32),
            Stage(25, 0.36),
            Stage(40, 0.40),
            ),
        guidance_scale=1.0,
        num_inference_steps=4,   
        p=0.3,                  
        target_size=384,
        prompt="",
        seed=123,
        dtype="fp16"
        )
    

    train_ds = DiffusedKvasirDataset(
        data=train_data,
        mode="train",
        image_size=image_size,
        mask_size=mask_size,
        diff_aug=diff_aug,
    )

    val_ds = DiffusedKvasirDataset(
        data=val_data,
        mode="val",
        image_size=image_size,
        mask_size=mask_size,
        diff_aug=None,
    )

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

    # ============================================================
    #  Model, loss, optimizer, scheduler
    # ============================================================
    model_name = env_vars.get("model_name", "segformer")
    model_config = env_vars.get("model_config", "b0")

    model = select_model(model_name, model_config)
    logger.info(f"Model: {model_name}-{model_config} | Params: {count_params(model):.2f}M")

    criterion = select_criterion(model_name)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = get_lr_scheduler(optimizer, num_epochs=60, warmup_epochs=5, min_lr=1e-6)

    # ============================================================
    #  Train
    # ============================================================
    best_model_save_path = os.path.join(output_path, f"{model_name}_{model_config}")
    os.makedirs(output_path, exist_ok=True)

    best_model, losses, ma_losses, dice_scores, miou_scores = training_loop(
        dataloader=train_loader,
        model=model,
        model_name=model_name,
        train_data_size=len(train_loader.dataset),
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        threshold=0.005,        # convergence tolerance
        ma_window=5,
        max_epochs=60,
        min_epochs=10,
        best_model_save_path=best_model_save_path,
        logger=logger,
        use_scheduler=True,
        save_model=True,
    )

    # ============================================================
    #  Validation
    # ============================================================
    logger.info("Running validation...")
    val_loss, val_dice, val_miou = test_loop(
        model=best_model,
        model_name=model_name,
        test_dataloader=val_loader,
        criterion=criterion,
        device=device,
        num_repeat=1,
    )
    logger.info(f"Validation - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, mIoU: {val_miou:.4f}")


if __name__ == "__main__":
    main()
