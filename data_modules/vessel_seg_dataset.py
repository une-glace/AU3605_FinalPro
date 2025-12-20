import os
import cv2
import torch
from torch.utils.data import Dataset
from utils import color_normalization


def match_mask(img_name, mask_dir):
    base = os.path.splitext(img_name)[0]

    # Adam Hoover: im0001.ppm -> im0001.ah.ppm
    if base.startswith("im") and base[2:].isdigit():
        candidate = f"{base}.ah.ppm"
        p = os.path.join(mask_dir, candidate)
        if os.path.exists(p):
            return p

    # DRIVE: 01_test -> 01_manual1.gif
    if base.endswith("_test"):
        base_drive = base.replace("_test", "")
        candidate = f"{base_drive}_manual1.gif"
        p = os.path.join(mask_dir, candidate)
        if os.path.exists(p):
            return p

    # maybe plain base.gif
    candidate = base + ".gif"
    p = os.path.join(mask_dir, candidate)
    if os.path.exists(p):
        return p

    # fallback fuzzy search
    for f in os.listdir(mask_dir):
        if base.split("_")[0] in f:
            return os.path.join(mask_dir, f)

    return None


class VesselSegDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=(256, 256)):
        print(f" ** Loading dataset from {images_dir}...")
        self.imgs = []
        self.masks = []

        files = sorted(os.listdir(images_dir))
        for img_name in files:
            img_path = os.path.join(images_dir, img_name)

            if not os.path.isfile(img_path):
                continue

            mask_path = match_mask(img_name, masks_dir)
            if mask_path is None:
                print(f"!! Warning: No mask found for {img_name}, skip.")
                continue

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                print(f"!! Warning: load failed {img_path} / {mask_path}")
                continue

            img = cv2.resize(img, img_size)
            mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)

            img = color_normalization(img)
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

            mask = torch.tensor(mask, dtype=torch.float32)
            mask = (mask > 127).float().unsqueeze(0)

            self.imgs.append(img)
            self.masks.append(mask)

        print(f" Loaded {len(self.imgs)} images.")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], self.masks[idx]
