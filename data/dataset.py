"""
Inpainting Dataset

Expects the following directory layout:

    data/
      train/
        images/   ← clean ground-truth images (jpg/png)
        masks/    ← binary masks; filename prefix before first '_' must match image stem (jpg/png)
                    white (255) = hole to inpaint, black (0) = keep
      val/
        images/
        masks/

The dataset returns a dict with:
    "masked_image"  : [3, H, W]  float32 in [-1, 1]  (image with hole zeroed out)
    "mask"          : [1, H, W]  float32, 1 = hole, 0 = known
    "ground_truth"  : [3, H, W]  float32 in [-1, 1]
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


_IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _find_images(directory: str) -> list:
    directory = Path(directory)
    return sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in _IMG_EXTENSIONS
    )


class InpaintingDataset(Dataset):
    """Paired image + mask dataset for inpainting.

    Args:
        data_dir  (str): Path to split directory (e.g. "data/train").
                         Must contain "images/" and "masks/" subdirectories
                         with matching filenames.
        image_size (int): Both H and W are resized to this value.
        augment   (bool): Apply random horizontal flip when True.
    """

    def __init__(self, data_dir: str, image_size: int = 256, augment: bool = False):
        self.image_size = image_size
        self.augment = augment

        img_dir  = Path(data_dir) / "images"
        mask_dir = Path(data_dir) / "masks"

        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if not mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

        self.image_paths = _find_images(img_dir)
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {img_dir}")

        # Build mask lookup keyed by the part before the first '_' so that
        # mask filenames like "abc123_m09j2d_ec3c2c21.png" match image "abc123.jpg"
        mask_by_prefix = {p.stem.split("_")[0]: p for p in _find_images(mask_dir)}
        self.mask_paths = []
        for img_path in self.image_paths:
            mask_path = mask_by_prefix.get(img_path.stem)
            if mask_path is None:
                raise FileNotFoundError(
                    f"No matching mask for image '{img_path.name}' in {mask_dir}"
                )
            self.mask_paths.append(mask_path)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        # ---- Load image (BGR → RGB) ---------------------------------------
        image = cv2.imread(str(self.image_paths[idx]))
        if image is None:
            raise IOError(f"Failed to load image: {self.image_paths[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size),
                           interpolation=cv2.INTER_LINEAR)

        # ---- Load mask (grayscale) ----------------------------------------
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Failed to load mask: {self.mask_paths[idx]}")
        mask = cv2.resize(mask, (self.image_size, self.image_size),
                          interpolation=cv2.INTER_NEAREST)

        # ---- Augmentation ------------------------------------------------
        if self.augment and np.random.rand() > 0.5:
            image = np.fliplr(image).copy()
            mask  = np.fliplr(mask).copy()

        # ---- To tensors --------------------------------------------------
        # Image: uint8 [0,255] → float32 [-1,1]
        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 127.5 - 1.0

        # Mask: threshold at 127 → binary float32 {0, 1}, shape [1, H, W]
        mask_t = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)

        # Masked image: zero out hole region (mask=1 means hole)
        masked_image_t = image_t * (1.0 - mask_t)

        return {
            "masked_image": masked_image_t,   # [3, H, W] in [-1, 1]
            "mask":         mask_t,            # [1, H, W] in {0, 1}
            "ground_truth": image_t,           # [3, H, W] in [-1, 1]
        }
