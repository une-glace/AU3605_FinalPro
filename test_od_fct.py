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

from models.od_fct_net import DiskMaculaNet
from data_modules.od_fct_dataset import OD_FCT_Dataset

def test(model_path, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f" ** Using device: {device}")

    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_images_dir = os.path.join(base_dir, "dataset", "OD_FCT", "test", "images")
    test_csv_path = os.path.join(base_dir, "dataset", "OD_FCT", "test", "targets.csv")
    results_dir = os.path.join(base_dir, "results_od_fct")
    os.makedirs(results_dir, exist_ok=True)

    # Load Dataset
    print("Loading Test Set...")
    test_dataset = OD_FCT_Dataset(test_images_dir, test_csv_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load Model
    print(f"Loading model from {model_path}...")
    model = DiskMaculaNet()
    
    # Handle DataParallel state dict if necessary
    try:
        state_dict = torch.load(model_path, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.to(device)
    model.eval()

    # Metrics
    od_distances = []
    fovea_distances = []

    print("Starting evaluation...")
    with torch.no_grad():
        for i, (image, label) in enumerate(tqdm(test_loader)):
            image = image.to(device)
            label = label.to(device)

            output = model(image)
            
            # Calculate Euclidean distances
            # label and output are [1, 4] -> (od_x, od_y, fovea_x, fovea_y)
            pred_od = output[0, 0:2]
            true_od = label[0, 0:2]
            pred_fovea = output[0, 2:4]
            true_fovea = label[0, 2:4]

            od_dist = torch.norm(pred_od - true_od).item()
            fovea_dist = torch.norm(pred_fovea - true_fovea).item()

            od_distances.append(od_dist)
            fovea_distances.append(fovea_dist)

            # Visualization (save first 20 images)
            if i < 20:
                # Convert tensor back to image
                img_np = image[0].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
                img_bgr = img_np.copy()
                
                # Draw Ground Truth (Green)
                cv2.circle(img_bgr, (int(true_od[0]), int(true_od[1])), 5, (0, 255, 0), -1)
                cv2.circle(img_bgr, (int(true_fovea[0]), int(true_fovea[1])), 5, (0, 255, 0), -1)
                
                # Draw Prediction (Red)
                cv2.circle(img_bgr, (int(pred_od[0]), int(pred_od[1])), 4, (0, 0, 255), -1)
                cv2.circle(img_bgr, (int(pred_fovea[0]), int(pred_fovea[1])), 4, (0, 0, 255), -1)
                
                # Add text
                cv2.putText(img_bgr, f"OD Dist: {od_dist:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(img_bgr, f"Fovea Dist: {fovea_dist:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                save_path = os.path.join(results_dir, f"result_{i:03d}.jpg")
                cv2.imwrite(save_path, img_bgr)

    mean_od_dist = np.mean(od_distances)
    mean_fovea_dist = np.mean(fovea_distances)
    
    print("\nEvaluation Results:")
    print(f"Mean OD Euclidean Distance: {mean_od_dist:.4f}")
    print(f"Mean Fovea Euclidean Distance: {mean_fovea_dist:.4f}")
    print(f"Visualization results saved to {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='logs/od_fct_model_best.pth', help='Path to the trained model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    args = parser.parse_args()
    
    # Adjust path if relative
    if not os.path.isabs(args.model_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        args.model_path = os.path.join(base_dir, args.model_path)

    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        print("Please train the model first or provide the correct path.")
    else:
        test(args.model_path, args.device)
