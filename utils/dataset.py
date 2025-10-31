# utils/dataset_diffusion_only.py
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class DiffusedKvasirDataset(Dataset):
    def __init__(
        self,
        data,
        mode,                          # "train" | "val" | "test"
        image_size=(512, 512),
        mask_size=(128, 128),
        diff_aug=None
    ):
        self.data = data
        self.mode = mode
        self.image_size = image_size
        self.mask_size = mask_size
        self.diff_aug = diff_aug

        self.base_img = transforms.Compose([
            transforms.Resize(self.image_size, transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),                     # [0,1], CHW
        ])
        self.base_mask = transforms.Compose([
            transforms.Resize(self.mask_size, transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),                     # [0,1], 1xHxW
        ])

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def set_epoch(self, epoch: int):
        if self.diff_aug is not None and hasattr(self.diff_aug, "set_epoch"):
            self.diff_aug.set_epoch(epoch)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_pil, mask_pil = self.data[idx]

        img = self.base_img(image_pil)     # [3,H,W], [0,1]
        msk = self.base_mask(mask_pil)     # [1,Hm, Wm], [0,1]

        if self.mode in ("train") and self.diff_aug is not None:
            img = self.diff_aug(img)       # keep [0,1]

        # normalize + binarize mask
        img = self.normalize(img)
        msk = (msk > 0.5).float()

        return img, msk
