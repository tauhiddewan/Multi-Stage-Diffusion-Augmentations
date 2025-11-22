# utils/dataset_diffusion_only.py

import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image


class DiffusedDataset(Dataset):
    def __init__(
        self,
        data,
        mode,                       # "train", "val", "test"
        aug_mode="none",            # "none", "just_classical", "just_diffusion", "classical+diffusion"
        image_size=(512, 512),
        mask_size=(128, 128),
        diff_aug=None,              # callable or None
    ):
        self.data = data
        self.mode = mode
        self.aug_mode = aug_mode.lower()
        self.image_size = image_size
        self.mask_size = mask_size
        self.diff_aug = diff_aug

        # Base resizing + tensor
        self.base_img_transforms = transforms.Compose([
            transforms.Resize(self.image_size, transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),                    # [0,1], [3,H,W]
        ])

        # For model mask (small)
        self.base_mask_transforms = transforms.Compose([
            transforms.Resize(self.mask_size, transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),                    # [0,1], [1,h,w]
        ])

        # For diffusion mask (full res)
        self.base_mask_full_transforms = transforms.Compose([
            transforms.Resize(self.image_size, transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),                    # [0,1], [1,H,W]
        ])

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # ---- Classical augmentations (copied in spirit from your old Dataset) ----
        if self.mode == 'train':
            # Image augmentations (spatial + intensity)
            self.aug_img_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(20, fill=0),
                transforms.RandomAffine(
                    degrees=5,
                    translate=(0.1, 0.1),
                    scale=(0.95, 1.05),
                    fill=0
                ),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2
                ),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                transforms.RandomGrayscale(p=0.2),
            ])

            # Mask augmentations (spatial only; NEAREST)
            self.aug_mask_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(
                    20,
                    fill=0,
                    interpolation=transforms.InterpolationMode.NEAREST
                ),
                transforms.RandomAffine(
                    degrees=5,
                    translate=(0.1, 0.1),
                    scale=(0.95, 1.05),
                    fill=0,
                    interpolation=transforms.InterpolationMode.NEAREST
                ),
                transforms.RandomPerspective(
                    distortion_scale=0.2,
                    p=0.5,
                    interpolation=transforms.InterpolationMode.NEAREST
                ),
            ])

        valid_aug_modes = {"none", "just_classical", "just_diffusion", "classical+diffusion"}
        if self.aug_mode not in valid_aug_modes:
            raise ValueError(f"aug_mode must be one of {valid_aug_modes}, got {self.aug_mode}")


    def _base_no_aug(self, image_pil, mask_pil):
        """
        Apply only base resize + ToTensor + normalize + binarize.
        Used for: aug_mode == 'none' OR for val/test in any mode.
        """
        img = self.base_img_transforms(image_pil)          # [3,H,W]
        msk_small = self.base_mask_transforms(mask_pil)    # [1,h,w]
        msk_full = self.base_mask_full_transforms(mask_pil)

        img = self.normalize(img)
        msk_small = (msk_small > 0.5).float()

        return img, msk_small, msk_full

    def _apply_classical_only(self, image_pil, mask_pil):
        img = self.base_img_transforms(image_pil)          # [3,H,W]
        msk_small = self.base_mask_transforms(mask_pil)    # [1,h,w]
        msk_full = self.base_mask_full_transforms(mask_pil)

        # classical aug only in train mode
        if self.mode == 'train':
            seed = random.randint(0, 2**32 - 1)

            # image aug
            torch.manual_seed(seed)
            random.seed(seed)
            img = self.aug_img_transforms(img)

            # mask aug (same seed -> same spatial ops)
            torch.manual_seed(seed)
            random.seed(seed)
            msk_small = self.aug_mask_transforms(msk_small)

        img = self.normalize(img)
        msk_small = (msk_small > 0.5).float()

        return img, msk_small, msk_full

    def _apply_diffusion_only(self, image_pil, mask_pil):
        img = self.base_img_transforms(image_pil)          # [3,H,W]
        msk_small = self.base_mask_transforms(mask_pil)    # [1,h,w]
        msk_full = self.base_mask_full_transforms(mask_pil)

        if self.mode == 'train' and self.diff_aug is not None:
            img = self.diff_aug(img, msk_full)             # expects [3,H,W] in [0,1]

        img = self.normalize(img)
        msk_small = (msk_small > 0.5).float()

        return img, msk_small, msk_full

    def _apply_classical_plus_diffusion(self, image_pil, mask_pil):
        """
        Classical + diffusion:
        - same classical pipeline as _apply_classical_only
        - THEN diffusion applied on the classical-augmented img (and full mask).
        """
        # First: classical (reusing previous logic)
        img = self.base_img_transforms(image_pil)
        msk_small = self.base_mask_transforms(mask_pil)
        msk_full = self.base_mask_full_transforms(mask_pil)

        if self.mode == 'train':
            seed = random.randint(0, 2**32 - 1)

            # classical img aug
            torch.manual_seed(seed)
            random.seed(seed)
            img = self.aug_img_transforms(img)

            # classical mask aug (small)
            torch.manual_seed(seed)
            random.seed(seed)
            msk_small = self.aug_mask_transforms(msk_small)

            # (Optional) classical mask aug (full) – if you need exact match with img for diffusion:
            torch.manual_seed(seed)
            random.seed(seed)
            msk_full = self.aug_mask_transforms(msk_full)

            # Now: diffusion on top
            if self.diff_aug is not None:
                img = self.diff_aug(img, msk_full)

        img = self.normalize(img)
        msk_small = (msk_small > 0.5).float()

        return img, msk_small, msk_full

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            img_path, mask_path = self.data[idx]
            image_pil = Image.open(img_path).convert("RGB")
            mask_pil = Image.open(mask_path).convert("L")

            # For val/test, ignore aug_mode: always clean pipeline
            if self.mode != "train":
                img, msk_small, _ = self._base_no_aug(image_pil, mask_pil)
                return img, msk_small

            # TRAIN: route by aug_mode
            if self.aug_mode == "none":
                img, msk_small, _ = self._base_no_aug(image_pil, mask_pil)

            elif self.aug_mode == "just_classical":
                img, msk_small, _ = self._apply_classical_only(image_pil, mask_pil)

            elif self.aug_mode == "just_diffusion":
                img, msk_small, _ = self._apply_diffusion_only(image_pil, mask_pil)

            elif self.aug_mode == "classical+diffusion":
                img, msk_small, _ = self._apply_classical_plus_diffusion(image_pil, mask_pil)

            else:
                raise ValueError(f"Unknown aug_mode: {self.aug_mode}")

            return img, msk_small

        except Exception as e:
            print(f"Error loading item {idx}: {str(e)}")
            raise
