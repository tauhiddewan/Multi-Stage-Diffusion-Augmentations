# utils/dataset_diffusion_only.py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class DiffusedDataset(Dataset):
    def __init__(
        self,
        data,
        mode,                      # "train", "val", "test"
        image_size=(512, 512),
        mask_size=(128, 128),
        diff_aug=None,             # callable or None
    ):
        
        self.data = data
        self.mode = mode
        self.image_size = image_size
        self.mask_size = mask_size
        self.diff_aug = diff_aug

        self.base_img = transforms.Compose([
            transforms.Resize(self.image_size, transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),       # [0,1], shape [3,H,W]
        ])

        self.base_mask = transforms.Compose([
            transforms.Resize(self.mask_size, transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),       # [0,1], shape [1,h,w] if grayscale
        ])

        self.base_mask_full = transforms.Compose([
            transforms.Resize(self.image_size, transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),       # [0,1], shape [1,H,W]
        ])

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_path = self.data[idx]
        image_pil = Image.open(img_path).convert("RGB")
        mask_pil  = Image.open(mask_path).convert("L")   # single channel

        # Basic transforms
        img = self.base_img(image_pil)           # [3,H,W] in [0,1]
        msk_small = self.base_mask(mask_pil)     # [1,h,w] in [0,1]
        msk_full  = self.base_mask_full(mask_pil)

        # Apply diffusion augmentation ONLY during training
        if self.mode == "train" and self.diff_aug is not None:
            img = self.diff_aug(img, msk_full)   # should return [3,H,W] in [0,1]

        img = self.normalize(img)
        msk_small = (msk_small > 0.5).float()    # [0,1] -> {0,1}

        return img, msk_small
