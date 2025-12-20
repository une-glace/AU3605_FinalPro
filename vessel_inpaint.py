import sys
import os
import argparse
from typing import List, Tuple

import cv2
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.vessel_seg_net import VesselSegNet


def _list_images(input_dir: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".ppm"}
    paths: List[str] = []
    for name in os.listdir(input_dir):
        ext = os.path.splitext(name)[1].lower()
        if ext in exts:
            paths.append(os.path.join(input_dir, name))
    paths.sort()
    return paths


def _resolve_image_dir(input_dir: str) -> str:
    if os.path.isdir(input_dir):
        images_subdir = os.path.join(input_dir, "images")
        if os.path.isdir(images_subdir):
            if _list_images(images_subdir):
                return images_subdir
    return input_dir


def _pretty_path(path: str, base_dir: str) -> str:
    try:
        rel = os.path.relpath(path, base_dir)
        if not rel.startswith(".."):
            return rel
    except Exception:
        pass
    return path


def _load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    model = VesselSegNet()
    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _predict_mask(
    model: torch.nn.Module,
    img_bgr: np.ndarray,
    device: torch.device,
    img_size: int,
    threshold: float,
) -> np.ndarray:
    resized = cv2.resize(img_bgr, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
        mask = (probs > threshold).to(torch.uint8)[0, 0].cpu().numpy()
    h, w = img_bgr.shape[:2]
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def _add_title(img_bgr: np.ndarray, title: str) -> np.ndarray:
    out = img_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    x, y = 12, 32
    ((tw, th), baseline) = cv2.getTextSize(title, font, font_scale, thickness)
    cv2.rectangle(out, (x - 6, y - th - 10), (x + tw + 6, y + baseline + 6), (0, 0, 0), -1)
    cv2.putText(out, title, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def _overlay_mask_alpha(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    color_bgr: Tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    out = img_bgr.copy()
    m = mask > 0
    if not np.any(m):
        return out
    base = out[m].astype(np.float32)
    color = np.array(color_bgr, dtype=np.float32)
    blended = base * (1.0 - alpha) + color * alpha
    out[m] = blended.clip(0, 255).astype(np.uint8)
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        default=os.path.join("dataset", "Vessel", "test"),
    )
    parser.add_argument("--output-dir", type=str, default="results_vessel_inpaint")
    parser.add_argument("--model-path", type=str, default=os.path.join("logs", "vessel_seg_model_best.pth"))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--dilate", type=int, default=2)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--inpaint-radius", type=int, default=3)
    parser.add_argument("--inpaint-method", type=str, default="telea", choices=["telea", "ns"])
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    input_dir = args.input_dir
    if not os.path.isabs(input_dir):
        input_dir = os.path.join(base_dir, input_dir)
    input_dir = _resolve_image_dir(input_dir)
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(base_dir, output_dir)
    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(base_dir, model_path)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir(input_dir):
        raise RuntimeError(f"Input dir not found: {input_dir}")
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model not found: {model_path}")

    os.makedirs(output_dir, exist_ok=True)

    print(f" ** Using device: {device}")
    print(f"Input dir: {_pretty_path(input_dir, base_dir)}")
    print(f"Output dir: {_pretty_path(output_dir, base_dir)}")
    print(f"Model path: {_pretty_path(model_path, base_dir)}")

    model = _load_model(model_path, device)

    image_paths = _list_images(input_dir)
    if args.limit and args.limit > 0:
        image_paths = image_paths[: args.limit]

    if not image_paths:
        print("No images found.")
        return

    if args.inpaint_method == "telea":
        inpaint_flag = cv2.INPAINT_TELEA
    else:
        inpaint_flag = cv2.INPAINT_NS

    kernel = None
    if args.dilate and args.dilate > 0:
        k = 2 * args.dilate + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    for img_path in image_paths:
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"Skip unreadable image: {img_path}")
            continue

        mask_pred = _predict_mask(model, img_bgr, device, args.img_size, args.threshold)
        mask_inpaint = mask_pred
        if kernel is not None:
            mask_inpaint = cv2.dilate(mask_inpaint, kernel, iterations=1)

        inpaint_mask = (mask_inpaint > 0).astype(np.uint8) * 255
        inpainted = cv2.inpaint(img_bgr, inpaint_mask, float(args.inpaint_radius), inpaint_flag)

        mid = _overlay_mask_alpha(img_bgr, mask_pred, (0, 0, 255), args.overlay_alpha)

        left = _add_title(img_bgr, "Original")
        mid = _add_title(mid, "Vessels (Pred)")
        right = _add_title(inpainted, "Inpainted")

        triptych = np.concatenate([left, mid, right], axis=1)

        stem = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(output_dir, f"{stem}_vessel_inpaint.jpg")
        cv2.imwrite(save_path, triptych)

    print("Done.")


if __name__ == "__main__":
    main()

