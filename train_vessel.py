import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm

from models.vessel_seg_net import UNet
from data_modules.vessel_seg_dataset import VesselSegDataset

try:
    import wandb
    _use_wandb = True
except ImportError:
    wandb = None
    _use_wandb = False


def _get_state_dict(model):
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def dice_loss(pred, target, smooth=1.0):
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) /
                (pred.sum() + target.sum() + smooth))


def evaluate(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    return val_loss / len(val_loader)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--load-pretrained", action="store_true")
    parser.add_argument("--pretrained-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f" ** Using device: {device}")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    train_img = os.path.join(base_dir, "dataset", "Vessel", "training", "AdamHoover", "images")
    train_mask = os.path.join(base_dir, "dataset", "Vessel", "training", "AdamHoover", "targets")

    test_img = os.path.join(base_dir, "dataset", "Vessel", "test", "images")
    test_mask = os.path.join(base_dir, "dataset", "Vessel", "test", "targets")

    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    train_dataset = VesselSegDataset(train_img, train_mask)
    val_dataset   = VesselSegDataset(test_img, test_mask)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = UNet(num_classes=1)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f" ** Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model.to(device)

    criterion = dice_loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # wandb
    wandb_enabled = False
    if _use_wandb:
        try:
            wandb.init(project="vessel_seg")
            wandb_enabled = True
        except:
            wandb_enabled = False

    print(" ** Start training Vessel Segmentation model...")

    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = evaluate(model, val_loader, device, criterion)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if wandb_enabled:
            wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss})

        torch.save(_get_state_dict(model), os.path.join(log_dir, "vessel_seg_latest.pth"))

        if avg_val_loss < best_val:
            best_val = avg_val_loss
            torch.save(_get_state_dict(model), os.path.join(log_dir, "vessel_seg_best.pth"))
            print(f"  New best model saved! (Val Loss: {best_val:.4f})")

    if wandb_enabled:
        wandb.finish()

    print("Vessel segmentation training completed.")
