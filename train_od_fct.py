import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.od_fct_net import DiskMaculaNet
from data_modules.od_fct_dataset import OD_FCT_Dataset

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
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    return val_loss / len(data_loader)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--load-pretrained", action="store_true")
    parser.add_argument("--pretrained-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--lr-scheduler", type=str, default="step", choices=["none", "step"])
    parser.add_argument("--step-size", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.1)
    return parser.parse_args()


if __name__ == '__main__':
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f" ** Using device: {device}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_images_dir = os.path.join(base_dir, "dataset", "OD_FCT", "train", "images")
    train_csv_path = os.path.join(base_dir, "dataset", "OD_FCT", "train", "targets.csv")
    test_images_dir = os.path.join(base_dir, "dataset", "OD_FCT", "test", "images")
    test_csv_path = os.path.join(base_dir, "dataset", "OD_FCT", "test", "targets.csv")
    log_dir = os.path.join(base_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    print("Loading Training Set...")
    train_dataset = OD_FCT_Dataset(train_images_dir, train_csv_path)
    print("Loading Test/Validation Set...")
    val_dataset = OD_FCT_Dataset(test_images_dir, test_csv_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    model = DiskMaculaNet()
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f" ** Using {num_gpus} GPUs with DataParallel")
            model = nn.DataParallel(model)
        else:
            print(" ** Using single GPU")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)

    scheduler = None
    if args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )

    load_pretrained = args.load_pretrained
    if load_pretrained:
        pretrained_path = args.pretrained_path
        if pretrained_path is None:
            pretrained_path = os.path.join(log_dir, "od_fct_model_latest.pth")
        if os.path.exists(pretrained_path):
            print(f"Loading pretrained model from {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location=device)
            model.load_state_dict(state_dict)

    wandb_enabled = False
    if _use_wandb:
        try:
            # 尝试初始化 wandb（需要提前运行 wandb login 或配置好环境变量）
            wandb.init(
                project="retina_od_fct",
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

    print(" ** Start training OD/FCT model...")

    best_val_loss = float("inf")

    best_model_path = os.path.join(log_dir, "od_fct_model_best.pth")
    if os.path.exists(best_model_path):
        print(f"Evaluating existing global best model from {best_model_path}")
        best_model = DiskMaculaNet()
        best_model.to(device)
        state_dict = torch.load(best_model_path, map_location=device)
        best_model.load_state_dict(state_dict)
        best_val_loss = evaluate(best_model, val_loader, device, criterion)
        print(f"Existing global best Val Loss: {best_val_loss:.4f}")

    for epoch in range(num_epochs):
        train_loss = 0.0
        model.train()
        for images, labels in tqdm(
            train_loader,
            desc=f"Train Epoch {epoch+1}/{num_epochs}",
            leave=False,
        ):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        avg_val_loss = evaluate(model, val_loader, device, criterion)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if wandb_enabled:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                }
            )

        torch.save(_get_state_dict(model), os.path.join(log_dir, 'od_fct_model_latest.pth'))
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(_get_state_dict(model), os.path.join(log_dir, 'od_fct_model_best.pth'))
            print(f"  New best OD/FCT model saved! (Val Loss: {best_val_loss:.4f})")

        if scheduler is not None:
            scheduler.step()

    if wandb_enabled:
        wandb.finish()

    print("OD/FCT training complete.")
