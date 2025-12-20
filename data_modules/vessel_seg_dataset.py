import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class VesselSegDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=(256, 256), sources=None):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        if sources is None:
            sources = ["DRIVE", "HRF", "AdamHoover"]
        self.sources = sources
        self.samples = []
        if split == "train":
            self._load_train_samples()
        else:
            self._load_test_samples()

    def _load_train_samples(self):
        base = os.path.join(self.root_dir, "training")
        if "DRIVE" in self.sources:
            drive_root = os.path.join(base, "DRIVE")
            images_dir = os.path.join(drive_root, "images")
            targets_dir = os.path.join(drive_root, "targets")
            if os.path.isdir(images_dir) and os.path.isdir(targets_dir):
                for name in os.listdir(targets_dir):
                    if not name.endswith("_manual1.gif"):
                        continue
                    stem = name.split("_")[0]
                    img_name = stem + "_training.tif"
                    img_path = os.path.join(images_dir, img_name)
                    mask_path = os.path.join(targets_dir, name)
                    if os.path.exists(img_path) and os.path.exists(mask_path):
                        self.samples.append((img_path, mask_path))
        if "AdamHoover" in self.sources:
            hoover_root = os.path.join(base, "AdamHoover")
            images_dir = os.path.join(hoover_root, "images")
            targets_dir = os.path.join(hoover_root, "targets")
            if os.path.isdir(images_dir) and os.path.isdir(targets_dir):
                for name in os.listdir(images_dir):
                    if not name.endswith(".ppm"):
                        continue
                    stem, _ = os.path.splitext(name)
                    mask_name = stem + ".ah.ppm"
                    img_path = os.path.join(images_dir, name)
                    mask_path = os.path.join(targets_dir, mask_name)
                    if os.path.exists(mask_path):
                        self.samples.append((img_path, mask_path))
        if "HRF" in self.sources:
            hrf_root = os.path.join(base, "HRF")
            images_dir = os.path.join(hrf_root, "images")
            targets_dir = os.path.join(hrf_root, "targets")
            if os.path.isdir(images_dir) and os.path.isdir(targets_dir):
                for name in os.listdir(targets_dir):
                    if not name.endswith("_dr.tif"):
                        continue
                    img_name = name.replace(".tif", ".JPG")
                    img_path = os.path.join(images_dir, img_name)
                    mask_path = os.path.join(targets_dir, name)
                    if os.path.exists(img_path) and os.path.exists(mask_path):
                        self.samples.append((img_path, mask_path))

    def _load_test_samples(self):
        base = os.path.join(self.root_dir, "test")
        images_dir = os.path.join(base, "images")
        targets_dir = os.path.join(base, "targets")
        if os.path.isdir(images_dir) and os.path.isdir(targets_dir):
            for name in os.listdir(targets_dir):
                if not name.endswith("_manual1.gif"):
                    continue
                stem = name.split("_")[0]
                img_name = stem + "_test.tif"
                img_path = os.path.join(images_dir, img_name)
                mask_path = os.path.join(targets_dir, name)
                if os.path.exists(img_path) and os.path.exists(mask_path):
                    self.samples.append((img_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            raise RuntimeError(f"Failed to load {img_path} or {mask_path}")
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)
        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask

