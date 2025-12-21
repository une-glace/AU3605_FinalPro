import os
import sys
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from glob import glob
from sklearn.decomposition import PCA

# 添加项目根目录到路径，以便导入 models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.od_fct_net import DiskMaculaNet
from utils import color_normalization

# ==========================================
# 1. 辅助函数：模型加载与预处理 (复用自 align_and_normalize.py)
# ==========================================

def load_od_fct_model(model_path, device):
    if not os.path.exists(model_path):
        print(f"Error: OD/FCT model not found at {model_path}")
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

def preprocess_for_detection(img, img_size=256):
    """
    为了跑 OD/FCT 检测模型而做的预处理：补边+缩放+颜色归一化
    """
    h, w = img.shape[:2]
    y, x = h, w
    diff = 0
    pad_x, pad_y = 0, 0
    
    if x > y:
        diff = (x - y) // 2
        img = cv2.copyMakeBorder(img, diff, diff, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        pad_y = diff
    elif y > x:
        diff = (y - x) // 2
        img = cv2.copyMakeBorder(img, 0, 0, diff, diff, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        pad_x = diff
        
    img_resized = cv2.resize(img, (img_size, img_size))
    try:
        img_norm = color_normalization(img_resized)
    except Exception as e:
        print(f"Warning: Color normalization failed ({e}), utilizing raw image.")
        img_norm = img_resized
        
    return img_norm

def predict_landmarks(model, img_256, device):
    img_tensor = torch.tensor(img_256, dtype=torch.float32)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
    coords = outputs[0].detach().cpu().numpy().astype(float)
    return coords # [od_x, od_y, fovea_x, fovea_y]

def align_image(img, landmarks, target_size=(256, 256)):
    """
    根据预测的 landmarks 将图像对齐到标准位置
    Standard: OD在左(30% width), Fovea在右(70% width), 垂直居中
    """
    od_x, od_y, fovea_x, fovea_y = landmarks
    
    target_od = np.array([target_size[0] * 0.3, target_size[1] * 0.5])
    target_fovea = np.array([target_size[0] * 0.7, target_size[1] * 0.5])
    
    curr_center = np.array([(od_x + fovea_x)/2, (od_y + fovea_y)/2])
    target_center = np.array([(target_od[0] + target_fovea[0])/2, (target_od[1] + target_fovea[1])/2])
    
    shift = target_center - curr_center
    M_trans = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    
    aligned_img = cv2.warpAffine(img, M_trans, target_size, flags=cv2.INTER_LINEAR)
    
    return aligned_img

# ==========================================
# 2. 新增：ROI Mask 生成
# ==========================================

def get_eye_mask(img_gray, erosion_iter=10):
    """
    生成眼底区域的掩膜，并腐蚀边缘以去除边界伪影
    """
    # 1. 简单的阈值分割 (眼底通常比背景亮)
    # 使用 Otsu 自动阈值，或者固定阈值 (如 10/255)
    # 注意：输入 img_gray 应该是 0-255 的 uint8
    if img_gray.dtype != np.uint8:
        img_gray = (img_gray * 255).astype(np.uint8)
        
    # Use Otsu's thresholding to handle variable background levels due to normalization
    thresh_val, mask = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"  [Mask Gen] Otsu threshold: {thresh_val}")
    
    # 2. 腐蚀操作 (Erosion) - 收缩掩膜
    # Kernel size 决定收缩的平滑度
    # 减小 Kernel 大小，避免切掉太多有效区域
    kernel = np.ones((3, 3), np.uint8) 
    mask_eroded = cv2.erode(mask, kernel, iterations=erosion_iter)
    
    # 归一化为 0/1 float
    return mask_eroded.astype(np.float32) / 255.0

# ==========================================
# 3. PCA 核心流程
# ==========================================

def load_and_process_data(image_paths, od_model, device, img_size=(256, 256)):
    processed_data = []
    valid_paths = []
    
    print(f"Processing {len(image_paths)} images with Alignment...")
    
    for idx, path in enumerate(image_paths):
        if idx % 10 == 0:
            print(f"  Processed {idx}/{len(image_paths)}...")
            
        img = cv2.imread(path)
        if img is None:
            continue
            
        img_input = preprocess_for_detection(img, img_size[0])
        coords = predict_landmarks(od_model, img_input, device)
        img_aligned = align_image(img_input, coords, target_size=img_size)
        
        img_gray = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2GRAY)
        img_flat = img_gray.astype(np.float32) / 255.0
        
        processed_data.append(img_flat.flatten())
        valid_paths.append(path)
        
    return np.array(processed_data), valid_paths

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    normal_images_dir = os.path.join(base_dir, "Normal Retinal Images")
    test_images_dir = os.path.join(base_dir, "dataset", "OD_FCT", "test", "images")
    output_dir = os.path.join(base_dir, "results_anomaly_pca") # 更改输出目录
    model_path = os.path.join(base_dir, "logs", "od_fct_model_best.pth")
    
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    od_model = load_od_fct_model(model_path, device)
    if od_model is None:
        print("Failed to load OD/FCT model. Exiting.")
        return

    normal_paths = glob(os.path.join(normal_images_dir, "*.tif"))
    if not normal_paths:
        print("No normal images found.")
        return
        
    print("--- Step 1: Loading & Aligning Training Data ---")
    X_train, _ = load_and_process_data(normal_paths, od_model, device)
    
    print("\n--- Step 2: Training PCA ---")
    n_components = 0.95
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    print(f"PCA Trained. Components: {pca.n_components_}, Var: {sum(pca.explained_variance_ratio_):.4f}")
    
    # 5. 测试与可视化
    IMG_SIZE = (256, 256)
    
    def run_test(img_path, save_name):
        img = cv2.imread(img_path)
        if img is None: return
        
        # 1. 预处理 & 对齐
        img_input = preprocess_for_detection(img, IMG_SIZE[0])
        coords = predict_landmarks(od_model, img_input, device)
        img_aligned = align_image(img_input, coords, target_size=IMG_SIZE)
        
        img_gray = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2GRAY)
        img_flat = img_gray.astype(np.float32) / 255.0
        
        # 2. PCA 重构
        transformed = pca.transform(img_flat.flatten().reshape(1, -1))
        reconstructed = pca.inverse_transform(transformed).reshape(IMG_SIZE)
        
        # 3. 生成 Mask (收缩边缘)
        # 这里的 iteration 可以调节，越大边缘切掉越多
        mask = get_eye_mask(img_gray, erosion_iter=15)
        
        # 4. 计算差异 并 应用 Mask
        original = img_flat
        diff_raw = np.abs(original - reconstructed)
        diff_masked = diff_raw * mask  # 关键步骤：过滤边缘误差
        
        # 5. 可视化
        plt.figure(figsize=(20, 5))
        
        plt.subplot(1, 5, 1); plt.title("Aligned Input"); plt.imshow(original, cmap='gray'); plt.axis('off')
        plt.subplot(1, 5, 2); plt.title("PCA Reconstructed"); plt.imshow(reconstructed, cmap='gray'); plt.axis('off')
        plt.subplot(1, 5, 3); plt.title("ROI Mask"); plt.imshow(mask, cmap='gray'); plt.axis('off')
        
        # 显示 Mask 后的差异图
        plt.subplot(1, 5, 4); plt.title("Diff (Masked)"); plt.imshow(diff_masked, cmap='jet'); plt.colorbar(); plt.axis('off')
        
        # 阈值分割
        # 动态阈值：基于 Mask 区域内的统计值
        valid_pixels = diff_masked[mask > 0]
        if len(valid_pixels) > 0:
            threshold = valid_pixels.mean() + 3.0 * valid_pixels.std()
        else:
            threshold = 0.1
            
        anomaly_bin = diff_masked > threshold
        plt.subplot(1, 5, 5); plt.title("Thresholded Anomaly"); plt.imshow(anomaly_bin, cmap='gray'); plt.axis('off')
        
        save_path = os.path.join(output_dir, save_name)
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
        plt.close()

    print("\n--- Step 3: Testing on ALL images in test dir ---")
    
    # Run on one normal image for reference
    if normal_paths:
        run_test(normal_paths[0], "reference_normal.png")
    
    abnormal_paths = glob(os.path.join(test_images_dir, "*.jpg"))
    print(f"Found {len(abnormal_paths)} test images.")
    
    for i, path in enumerate(abnormal_paths):
        filename = os.path.basename(path)
        save_name = f"result_{filename}"
        print(f"[{i+1}/{len(abnormal_paths)}] Processing {filename}...")
        run_test(path, save_name)

if __name__ == "__main__":
    main()
