# utils/plots.py
import os
import matplotlib.pyplot as plt

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_training_curves(
    losses,
    ma_losses,
    dice_scores,
    miou_scores,
    out_dir: str = "./outputs",
    prefix: str = "curves"
):
    _ensure_dir(out_dir)
    epochs = range(1, len(losses) + 1)

    plt.figure()
    plt.plot(epochs, losses, label="Loss")
    if ma_losses and len(ma_losses) == len(losses):
        plt.plot(epochs, ma_losses, label="MA Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    loss_path = os.path.join(out_dir, f"{prefix}_loss.png")
    plt.savefig(loss_path, bbox_inches="tight", dpi=150)
    plt.close()

    # 2) Dice
    if dice_scores:
        plt.figure()
        plt.plot(epochs, dice_scores, label="Dice", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Dice")
        plt.title("Dice Score")
        plt.grid(True, alpha=0.3)
        dice_path = os.path.join(out_dir, f"{prefix}_dice.png")
        plt.savefig(dice_path, bbox_inches="tight", dpi=150)
        plt.close()

    # 3) mIoU
    if miou_scores:
        plt.figure()
        plt.plot(epochs, miou_scores, label="mIoU", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("mIoU")
        plt.title("Mean IoU")
        plt.grid(True, alpha=0.3)
        miou_path = os.path.join(out_dir, f"{prefix}_miou.png")
        plt.savefig(miou_path, bbox_inches="tight", dpi=150)
        plt.close()

    return {
        "loss": loss_path,
        "dice": dice_path if dice_scores else None,
        "miou": miou_path if miou_scores else None,
    }
