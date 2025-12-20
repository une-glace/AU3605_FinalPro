import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import cv2
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.vessel_seg_net import VesselSegNet
from data_modules.vessel_seg_dataset import VesselSegDataset


def evaluate(model, data_loader, device):
    model.eval()
    dice_sum = 0.0
    eps = 1e-7
    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            intersection = (preds * masks).sum(dim=(1, 2, 3))
            union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
            dice = (2.0 * intersection + eps) / (union + eps)
            dice_sum += dice.mean().item()
    return dice_sum / len(data_loader)


def test(model_path, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f" ** Using device: {device}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    vessel_root = os.path.join(base_dir, "dataset", "Vessel")
    results_dir = os.path.join(base_dir, "results_vessel_seg")
    os.makedirs(results_dir, exist_ok=True)

    print("Loading Vessel Test Set...")
    test_dataset = VesselSegDataset(vessel_root, split="test", img_size=(256, 256))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"Loading vessel model from {model_path}...")
    model = VesselSegNet()

    try:
        state_dict = torch.load(model_path, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.to(device)
    model.eval()

    dice_scores = []

    print("Starting vessel segmentation evaluation...")
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(test_loader)):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            eps = 1e-7
            intersection = (preds * masks).sum()
            union = preds.sum() + masks.sum()
            dice = (2.0 * intersection + eps) / (union + eps)
            dice_scores.append(dice.item())

            if i < 20:
                img_np = images[0].cpu().permute(1, 2, 0).numpy()
                img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
                pred_mask = preds[0, 0].cpu().numpy().astype(np.uint8) * 255
                true_mask = masks[0, 0].cpu().numpy().astype(np.uint8) * 255

                pred_color = cv2.applyColorMap(pred_mask, cv2.COLORMAP_JET)
                true_color = cv2.applyColorMap(true_mask, cv2.COLORMAP_JET)

                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                overlay_pred = cv2.addWeighted(img_bgr, 0.7, pred_color, 0.3, 0)
                overlay_true = cv2.addWeighted(img_bgr, 0.7, true_color, 0.3, 0)

                concat = np.concatenate([img_bgr, overlay_true, overlay_pred], axis=1)
                save_path = os.path.join(results_dir, f"vessel_result_{i:03d}.jpg")
                cv2.imwrite(save_path, concat)

    mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
    print(f"\nVessel segmentation evaluation complete. Mean Dice: {mean_dice:.4f}")
    print(f"Visualization results saved to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="logs/vessel_seg_model_best.pth")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if not os.path.isabs(args.model_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        args.model_path = os.path.join(base_dir, args.model_path)

    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        print("Please train the vessel segmentation model first or provide the correct path.")
    else:
        test(args.model_path, args.device)

