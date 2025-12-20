import os
import sys
import argparse

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.od_fct_net import DiskMaculaNet
from utils import color_normalization


def preprocess_image(img, img_size):
    h, w = img.shape[:2]
    y = h
    x = w
    diff = 0
    pad_x = 0
    pad_y = 0
    if x > y:
        diff = (x - y) // 2
        img = cv2.copyMakeBorder(img, diff, diff, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        pad_y = diff
    elif y > x:
        diff = (y - x) // 2
        img = cv2.copyMakeBorder(img, 0, 0, diff, diff, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        pad_x = diff
    max_dim = max(x, y)
    img = cv2.resize(img, (img_size, img_size))
    img = color_normalization(img)
    return img, x, y, pad_x, pad_y, max_dim


def load_model(model_path, device):
    if not os.path.exists(model_path):
        print(f"Error: model not found at {model_path}")
        return None
    model = DiskMaculaNet()
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model


def predict_coords(model, img_256, device):
    img_tensor = torch.tensor(img_256, dtype=torch.float32)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
    coords = outputs[0].detach().cpu().numpy().astype(float)
    return coords[0], coords[1], coords[2], coords[3]


def compute_similarity_transform(od_x, od_y, fovea_x, fovea_y, target_od, target_f):
    center = np.array(
        [(od_x + fovea_x) / 2.0, (od_y + fovea_y) / 2.0],
        dtype=np.float32,
    )
    target_center = np.array(
        [(target_od[0] + target_f[0]) / 2.0, (target_od[1] + target_f[1]) / 2.0],
        dtype=np.float32,
    )

    shift = target_center - center

    m = np.array(
        [[1.0, 0.0, float(shift[0])], [0.0, 1.0, float(shift[1])]],
        dtype=np.float32,
    )
    return m


def ensure_demo_dirs(base_dir):
    images_dir = os.path.join(base_dir, "alignment_demo_input")
    output_dir = os.path.join(base_dir, "alignment_demo_output")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    return images_dir, output_dir


def process(args):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = args.images_dir
    output_dir = args.output_dir
    if images_dir is None or output_dir is None:
        images_dir, output_dir = ensure_demo_dirs(base_dir)

    device = torch.device("cpu")
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")

    model = load_model(args.model_path, device)
    if model is None:
        return

    ref_path = os.path.join(base_dir, "utils", "ref_img.jpg")
    ref_img = cv2.imread(ref_path, cv2.IMREAD_COLOR)
    if ref_img is None:
        print(f"Error: reference image not found at {ref_path}")
        return
    ref_256, _, _, _, _, _ = preprocess_image(ref_img, args.img_size)
    od_rx, od_ry, fv_rx, fv_ry = predict_coords(model, ref_256, device)
    target_od = np.array([od_rx, od_ry], dtype=np.float32)
    target_f = np.array([fv_rx, fv_ry], dtype=np.float32)
    ref_dx = target_f[0] - target_od[0]

    os.makedirs(output_dir, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    selected_original = None
    selected_aligned = None

    for name in sorted(os.listdir(images_dir)):
        base, ext = os.path.splitext(name)
        if ext.lower() not in exts:
            continue
        img_path = os.path.join(images_dir, name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue

        img_256, _, _, _, _, _ = preprocess_image(img, args.img_size)
        od_x, od_y, fovea_x, fovea_y = predict_coords(model, img_256, device)

        img_for_align = img_256
        dx = fovea_x - od_x
        if ref_dx * dx < 0:
            img_for_align = cv2.flip(img_256, 1)
            od_x = args.img_size - 1 - od_x
            fovea_x = args.img_size - 1 - fovea_x

        m = compute_similarity_transform(od_x, od_y, fovea_x, fovea_y, target_od, target_f)
        if m is None:
            continue

        aligned = cv2.warpAffine(
            img_for_align,
            m,
            (args.img_size, args.img_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        out_path = os.path.join(output_dir, name)
        cv2.imwrite(out_path, aligned)

        if selected_original is None:
            selected_original = img_256
            selected_aligned = aligned

    if selected_original is not None:
        ref_vis = cv2.resize(ref_256, (args.img_size, args.img_size))
        ref_vis = cv2.cvtColor(ref_vis, cv2.COLOR_BGR2RGB)
        orig_vis = cv2.cvtColor(selected_original, cv2.COLOR_BGR2RGB)
        aligned_vis = cv2.cvtColor(selected_aligned, cv2.COLOR_BGR2RGB)

        h, w, _ = ref_vis.shape
        corner = max(1, h // 16)
        corners = np.concatenate(
            [
                ref_vis[:corner, :corner].reshape(-1, 3),
                ref_vis[:corner, -corner:].reshape(-1, 3),
                ref_vis[-corner:, :corner].reshape(-1, 3),
                ref_vis[-corner:, -corner:].reshape(-1, 3),
            ],
            axis=0,
        )
        bg_color = corners.mean(axis=0).astype(ref_vis.dtype)

        def unify_background(img):
            mask = (
                (img[:, :, 0] == 0)
                & (img[:, :, 1] == 0)
                & (img[:, :, 2] == 0)
            )
            out = img.copy()
            out[mask] = bg_color
            return out

        ref_vis = unify_background(ref_vis)
        orig_vis = unify_background(orig_vis)
        aligned_vis = unify_background(aligned_vis)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        axes[0].imshow(ref_vis)
        axes[0].set_title("Reference")
        axes[0].axis("off")
        axes[1].imshow(orig_vis)
        axes[1].set_title("Original")
        axes[1].axis("off")
        axes[2].imshow(aligned_vis)
        axes[2].set_title("Aligned")
        axes[2].axis("off")
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-path", type=str, default="logs/od_fct_model_best.pth")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--img-size", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process(args)
