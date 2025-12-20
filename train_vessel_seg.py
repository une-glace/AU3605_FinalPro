import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.vessel_seg_net import VesselSegNet
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


def evaluate(model, data_loader, device, criterion):
    model.eval()
    val_loss = 0.0
    dice_sum = 0.0
    eps = 1e-7
    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            val_loss += loss.item()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            intersection = (preds * masks).sum(dim=(1, 2, 3))
            union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
            dice = (2.0 * intersection + eps) / (union + eps)
            dice_sum += dice.mean().item()
    return val_loss / len(data_loader), dice_sum / len(data_loader)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--lr-scheduler", type=str, default="step", choices=["none", "step"])
    parser.add_argument("--step-size", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--img-size", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f" ** Using device: {device}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    vessel_root = os.path.join(base_dir, "dataset", "Vessel")
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    print("Loading Vessel Training Set...")
    train_dataset = VesselSegDataset(vessel_root, split="train", img_size=(args.img_size, args.img_size))
    print("Loading Vessel Test/Validation Set...")
    val_dataset = VesselSegDataset(vessel_root, split="test", img_size=(args.img_size, args.img_size))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    model = VesselSegNet()
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f" ** Using {num_gpus} GPUs with DataParallel")
            model = nn.DataParallel(model)
        else:
            print(" ** Using single GPU")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)

    scheduler = None
    if args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    wandb_enabled = False
    if _use_wandb:
        try:
            wandb.init(
                project="retina_vessel_seg",
                config={
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "train_size": len(train_dataset),
                    "val_size": len(val_dataset),
                },
            )
            wandb_enabled = True
        except Exception as e:
            print(f" ** Wandb init failed (ignored): {e}")
            wandb_enabled = False

    print(" ** Start training vessel segmentation model...")

    best_val_loss = float("inf")
    best_model_path = os.path.join(log_dir, "vessel_seg_model_best.pth")
    latest_model_path = os.path.join(log_dir, "vessel_seg_model_latest.pth")

    if os.path.exists(best_model_path):
        print(f"Evaluating existing global best vessel model from {best_model_path}")
        best_model = VesselSegNet()
        best_model.to(device)
        state_dict = torch.load(best_model_path, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        best_model.load_state_dict(new_state_dict)
        best_val_loss, best_val_dice = evaluate(best_model, val_loader, device, criterion)
        print(f"Existing global best Val Loss: {best_val_loss:.4f}, Dice: {best_val_dice:.4f}")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_dice_sum = 0.0
        eps = 1e-7
        for images, masks in tqdm(
            train_loader,
            desc=f"Train Epoch {epoch + 1}/{num_epochs}",
            leave=False,
        ):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            intersection = (preds * masks).sum(dim=(1, 2, 3))
            union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
            dice = (2.0 * intersection + eps) / (union + eps)
            train_dice_sum += dice.mean().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_dice = train_dice_sum / len(train_loader)
        avg_val_loss, avg_val_dice = evaluate(model, val_loader, device, criterion)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_val_dice:.4f}"
        )

        if wandb_enabled:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "train_dice": avg_train_dice,
                    "val_loss": avg_val_loss,
                    "val_dice": avg_val_dice,
                }
            )

        torch.save(_get_state_dict(model), latest_model_path)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(_get_state_dict(model), best_model_path)
            print(f"  New best vessel model saved! (Val Loss: {best_val_loss:.4f}, Val Dice: {avg_val_dice:.4f})")

        if scheduler is not None:
            scheduler.step()

    if wandb_enabled:
        wandb.finish()

    print("Vessel segmentation training complete.")

