import os
import math
import pickle
import torch
from glob import glob
from PIL import Image
from dotenv import dotenv_values
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from utils.dataset import DiffusedKvasirDataset
from utils.diff_aug import ControlNetAug
from utils.model import select_model, count_params
from utils.loss import select_criterion
from utils.train import training_loop, test_loop, get_lr_scheduler
from utils.plots import save_training_curves
from utils.misc import create_logger
from utils.viz import save_diff_example, save_diff_batch

def alpha_linear(epoch, T, a0=0.95, a1=0.40):
    t = 0 if T is None or T <= 1 else epoch / (T - 1)
    return a0 * (1 - t) + a1 * t

def alpha_cosine(epoch, T, a0=0.95, a1=0.40):
    t = 0 if T is None or T <= 1 else epoch / (T - 1)
    w = 0.5 * (1 - math.cos(math.pi * t))
    return a0 * (1 - w) + a1 * w


def main():
    env_vars = dotenv_values(dotenv_path="./.env")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = create_logger(log_filename="controlnet_alpha_blend", env_vars=env_vars)
    
    with open(f'{env_vars["data_folder_path"]}/{env_vars["split_fname"]}', "rb") as f:
        data = pickle.load(f)

    train_data = data["train_data"]
    test_data = data["test_data"]

    image_size = (384, 384)   # model input size
    mask_size = (96, 96)      # mask for model head (downsampled)
    output_path = env_vars.get("output_folder_path", "./outputs")

    # diff_aug = ControlNetAug(
    #     alpha = float(env_vars.get("alpha", 0.95)), # start mostly real; constant for now
    #     prob_value=float(env_vars.get("prob_value", 0.4)), # augment ~40% of training samples
    #     model_id="runwayml/stable-diffusion-v1-5",
    #     controlnet_id="lllyasviel/control_v11p_sd15_seg",
    #     prompt=str(env_vars.get("prompt", "colonscopic image, realistic lighting, different hospital scanner")),
    #     neg_prompt=None,
    #     guidance_scale=1.0,
    #     condn_scale=0.5,
    #     num_inference_steps=10,
    #     target_img_size=image_size[0], # match to avoid extra resize
    #     random_seed=123,
    #     dtype="fp16",
    #     device=device,                   
    #     alpha_schedule=None, # None = curriculum inactive
    #     total_epochs=None, # unused while schedule is None
    # )

    diff_aug = ControlNetAug(
        alpha = float(env_vars.get("alpha")),
        prob_value = float(env_vars.get("prob_value")),
        model_id = str(env_vars.get("model_id")),
        controlnet_id = env_vars.get("controlnet_id"),
        prompt = str(env_vars.get("prompt")),
        neg_prompt =  str(env_vars.get("neg_prompt")),
        guidance_scale = float(env_vars.get("guidance_scale")),
        condn_scale = float(env_vars.get("guidance_scale")),
        num_inference_steps = int(env_vars.get("num_inference_steps")),
        target_img_size = image_size[0],
        random_seed = 123,
        dtype="fp16",
        device=device
    )


    _ = save_diff_batch(diff_aug,
                        samples=[train_data[i] for i in range(min(6, len(train_data)))],
                        image_size=(384,384),
                        out_dir=output_path,
                        prefix="diff")
    

    train_ds = DiffusedKvasirDataset(
        data=train_data,
        mode="train",
        image_size=image_size,
        mask_size=mask_size,
        diff_aug=diff_aug,
    )
    test_ds = DiffusedKvasirDataset(
        data=test_data,
        mode="test",
        image_size=image_size,
        mask_size=mask_size,
        diff_aug=None,
    )

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False,
                             num_workers=0, pin_memory=True)


    model_name = env_vars.get("model_name", "segformer")
    model_config = env_vars.get("model_config", "b0")

    model = select_model(model_name, model_config)
    logger.info(f"Model: {model_name}-{model_config} | Params: {count_params(model):.2f}M")

    criterion = select_criterion(model_name)
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    num_epochs = int(env_vars.get("num_epochs", 100))
    scheduler = get_lr_scheduler(optimizer, num_epochs=num_epochs,
                                 warmup_epochs=5, min_lr=1e-6)

    best_model_save_path = os.path.join(output_path, f"{model_name}_{model_config}")
    os.makedirs(output_path, exist_ok=True)

    # NOTE (for later curriculum):
    #   IMPLEMENT
    #   IMPLEMENT
    #   IMPLEMENT
    #   IMPLEMENT

    best_model, losses, ma_losses, dice_scores, miou_scores = training_loop(
        dataloader=train_loader,
        model=model,
        model_name=model_name,
        train_data_size=len(train_loader.dataset),
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        threshold=float(env_vars.get("threshold", 0.005)),        # convergence tolerance
        ma_window=int(env_vars.get("ma_window", 10)),
        max_epochs=int(env_vars.get("max_epochs", 100)),
        min_epochs=int(env_vars.get("min_epochs", 20)),
        best_model_save_path=best_model_save_path,
        logger=logger,
        use_scheduler=bool(env_vars.get("use_scheduler", False)),
        save_model=bool(env_vars.get("save_model", True))
    )

    paths = save_training_curves(
        losses=losses,
        ma_losses=ma_losses,
        dice_scores=dice_scores,
        miou_scores=miou_scores,
        out_dir=output_path,
        prefix="train"
    )
    logger.info(f"Saved training curves: {paths}")

    test_loss, test_dice, test_miou = test_loop(
        model=best_model,
        model_name=model_name,
        test_dataloader=test_loader,
        criterion=criterion,
        device=device,
        num_repeat=1,
    )
    logger.info(f"Test - Loss: {test_loss:.4f}, Dice: {test_dice:.4f}, mIoU: {test_miou:.4f}")


if __name__ == "__main__":
    main()
