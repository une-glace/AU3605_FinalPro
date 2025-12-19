import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from utils import color_normalization

class OD_FCT_Dataset(Dataset):
    def __init__(self, images_dir, target_csv_path, img_size=(256, 256)):
        print(f" ** Loading dataset from {images_dir} and {target_csv_path}...")
        self.data = []
        self.target = []
        self.img_size = img_size
        self.images_dir = images_dir
        targets_df = pd.read_csv(target_csv_path)
        targets_df = targets_df.dropna(subset=["ID", "OD_X", "OD_Y", "Fovea_X", "Fovea_Y"])
        self.targets_dict = {}
        for _, row in targets_df.iterrows():
            self.targets_dict[row["ID"]] = (row["OD_X"], row["OD_Y"], row["Fovea_X"], row["Fovea_Y"])
        print(f"Found {len(self.targets_dict)} entries in CSV.")
        count = 0
        for img_id in self.targets_dict.keys():
            img_path = os.path.join(images_dir, img_id + ".jpg")
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Warning: Failed to load image {img_path}")
                continue
            img_size_ori = img.shape
            y, x = img.shape[:2]
            diff = 0
            if x > y:
                diff = (x - y) // 2
                img = cv2.copyMakeBorder(
                    img, diff, diff, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )
            elif y > x:
                diff = (y - x) // 2
                img = cv2.copyMakeBorder(
                    img, 0, 0, diff, diff, cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )
            img = cv2.resize(img, self.img_size)
            img = color_normalization(img)
            img = torch.tensor(img, dtype=torch.float32)
            img = img.permute(2, 0, 1)
            self.data.append(img)
            od_x, od_y, fovea_x, fovea_y = self.targets_dict[img_id]
            max_dim = max(x, y)
            pad_x = 0
            pad_y = 0
            if x > y:
                pad_y = diff
            elif y > x:
                pad_x = diff
            target_tensor = torch.tensor(
                (
                    (od_x + pad_x) * 256 / max_dim,
                    (od_y + pad_y) * 256 / max_dim,
                    (fovea_x + pad_x) * 256 / max_dim,
                    (fovea_y + pad_y) * 256 / max_dim
                ),
                dtype=torch.float32
            )
            self.target.append(target_tensor)
            count += 1
        print(f"Successfully loaded {count} images.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

