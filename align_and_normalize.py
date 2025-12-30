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
    src_od = np.array([od_x, od_y], dtype=np.float32)
    src_f = np.array([fovea_x, fovea_y], dtype=np.float32)
    tgt_od = np.array([target_od[0], target_od[1]], dtype=np.float32)
    tgt_f = np.array([target_f[0], target_f[1]], dtype=np.float32)

    src_vec = src_f - src_od
    tgt_vec = tgt_f - tgt_od

    src_len = float(np.linalg.norm(src_vec))
    tgt_len = float(np.linalg.norm(tgt_vec))
    if src_len < 1e-6 or tgt_len < 1e-6:
        return None

    scale = tgt_len / src_len

    angle_src = float(np.arctan2(src_vec[1], src_vec[0]))
    angle_tgt = float(np.arctan2(tgt_vec[1], tgt_vec[0]))
    angle = angle_tgt - angle_src

    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))
    r = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    rs = scale * r

    t = tgt_od - np.dot(rs, src_od)

    m = np.zeros((2, 3), dtype=np.float32)
    m[:, :2] = rs
    m[:, 2] = t
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
    else:
        if not os.path.isabs(images_dir):
            images_dir = os.path.join(base_dir, images_dir)
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(base_dir, output_dir)

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

    ref_vis = cv2.resize(ref_256, (args.img_size, args.img_size))
    ref_vis = cv2.cvtColor(ref_vis, cv2.COLOR_BGR2RGB)
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

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

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

        src_od_h = np.array([od_x, od_y, 1.0], dtype=np.float32)
        src_fv_h = np.array([fovea_x, fovea_y, 1.0], dtype=np.float32)
        aligned_od = np.dot(m, src_od_h)
        aligned_fv = np.dot(m, src_fv_h)
        print(
            f"{name}: "
            f"OD src=({od_x:.2f}, {od_y:.2f}), "
            f"Fovea src=({fovea_x:.2f}, {fovea_y:.2f}), "
            f"OD aligned=({aligned_od[0]:.2f}, {aligned_od[1]:.2f}), "
            f"Fovea aligned=({aligned_fv[0]:.2f}, {aligned_fv[1]:.2f}), "
            f"OD target=({target_od[0]:.2f}, {target_od[1]:.2f}), "
            f"Fovea target=({target_f[0]:.2f}, {target_f[1]:.2f})"
        )

        m_h = np.eye(3, dtype=np.float32)
        m_h[:2, :3] = m
        m_inv = np.linalg.inv(m_h)
        m_warp = m_inv[:2, :]

        aligned = cv2.warpAffine(
            img_for_align,
            m_warp,
            (args.img_size, args.img_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        out_path = os.path.join(output_dir, name)
        cv2.imwrite(out_path, aligned)

        orig_vis = cv2.cvtColor(img_256, cv2.COLOR_BGR2RGB)
        aligned_vis = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)

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
        triptych_base, _ = os.path.splitext(name)
        triptych_filename = f"alignment_demo_triptych_{triptych_base}.png"
        triptych_path = os.path.join(output_dir, triptych_filename)
        fig.savefig(triptych_path, dpi=300)
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=str, default="alignment_demo_input")
    parser.add_argument("--output-dir", type=str, default="alignment_demo_output")
    parser.add_argument("--model-path", type=str, default="logs/od_fct_model_best.pth")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--img-size", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process(args)
