# AU3605_FinalPro 视网膜图像处理项目

本项目在 `AU3605_FinalPro` 目录下，主要用于完成课程大作业中视网膜图像的自动处理，包括：

- 视盘（Optic Disc, OD）与黄斑（Fovea, FCT）中心检测（第一个模型）  
- 血管分割（第二个模型，数据与脚本结构已预留）  
- 数据预处理工具（颜色归一化、批量解压、Adam Hoover 数据整理等）

下面只说明当前已经打通的第一阶段（OD/FCT 检测）和整体框架，方便在本机或实验室服务器上使用。

---

## 1. 目录结构

根目录：`AU3605_FinalPro`

- `train_od_fct.py`  
  OD/FCT 检测模型的训练入口脚本（支持 wandb、单卡、多卡 `DataParallel`、进度条显示）。

- `test_od_fct.py`  
  OD/FCT 模型的测试脚本，加载训练好的模型并在测试集上进行评估和可视化。

- `models/`  
  - `od_fct_net.py`  
    OD/FCT 检测网络结构 `DiskMaculaNet`，输入 `3×256×256`，输出 4 个坐标：  
    `[OD_x, OD_y, Fovea_x, Fovea_y]`。
  - `__init__.py`  
    方便从 `models` 导入模型类。

- `data_modules/`  
  - `od_fct_dataset.py`  
    OD/FCT 任务的数据集类 `OD_FCT_Dataset`：  
    - 读取 `dataset/OD_FCT/train` 与 `dataset/OD_FCT/test` 中的 IDRiD 图像和对应 CSV。  
    - 将图像补成正方形并缩放至 `256×256`。  
    - 使用 `utils.color_normalization` 做颜色归一化。  
    - 将原始坐标缩放映射到 `256×256` 坐标系。  
  - `__init__.py`

- `dataset/`  
  - `OD_FCT/`  
    第一个模型使用的 IDRiD 数据集（已整理好）：  
    - `train/images/IDRiD_XXX.jpg`：训练图像。  
    - `train/targets.csv`：训练标签（CSV 中列为 `ID, OD_X, OD_Y, Fovea_X, Fovea_Y`）。  
    - `test/images/IDRiD_XXX.jpg`：测试/验证图像。  
    - `test/targets.csv`：测试/验证标签。
  - `vessel_seg/`（预留）  
    为第二个模型（血管分割）预留的数据根目录，建议按来源分为三个子目录，例如：  
    - `normal/`：正常视网膜图像及对应血管标注。  
    - `adam_hoover/`：Adam Hoover 数据。  
    - `other/`：其他来源的数据。  
    后续可在此基础上实现 `vessel_seg` 的 Dataset 与训练脚本。

- `logs/`  
  存放训练过程中保存的模型权重（`od_fct_model_best.pth`, `od_fct_model_latest.pth`）。

- `results/`  
  运行测试脚本 `test_od_fct.py` 后生成的可视化结果（带有预测点和真实点的对比图）。

- `utils/`  
  - `__init__.py`：提供 `color_normalization` 函数，利用 `ref_img.jpg` 做颜色风格归一化。  
  - `ref_img.jpg`：颜色归一化参考图像。

- `tools/`  
  一些数据整理类工具脚本，均带有中文注释说明用途：  
  - `decompress_gz_dataset.py`  
    从指定根目录递归查找 `.gz` 文件并解压（适合 Adam Hoover 血管标注文件等）。  
  - `organize_hoover_ppm.py`  
    将 Adam Hoover 提供的 PPM 原图和血管标注整理到 `images/` 与 `labels/` 目录。

- `Normal Retinal Images/`  
  原始的正常视网膜图像（tif），可作为第二个模型或其他任务的数据来源。

---

## 2. 环境与依赖

建议使用 Conda 环境（示例）：

```bash
conda create -n dip python=3.10
conda activate dip
pip install torch torchvision torchaudio  # 按服务器 CUDA 版本选择对应 wheel
pip install opencv-python pandas numpy tqdm
pip install wandb  # 如需在线可视化
```

如果不安装 `wandb`，训练脚本会自动跳过 wandb 相关部分，不会报错。

---

## 3. 第一个模型：OD/FCT 中心检测训练与测试

### 3.1 数据要求

OD/FCT 模型使用的训练和测试数据已经整理在：

- 训练集：  
  - 图像：`dataset/OD_FCT/train/images/IDRiD_XXX.jpg`  
  - 标签：`dataset/OD_FCT/train/targets.csv`
- 测试/验证集：  
  - 图像：`dataset/OD_FCT/test/images/IDRiD_XXX.jpg`  
  - 标签：`dataset/OD_FCT/test/targets.csv`

`targets.csv` 文件格式（列名）：

- `ID`：不含后缀的图像名，例如 `IDRiD_001`。  
- `OD_X, OD_Y`：视盘中心坐标（原图坐标系）。  
- `Fovea_X, Fovea_Y`：黄斑中心坐标（原图坐标系）。

数据加载时会根据图像原始宽高和补边比例，将坐标映射到 `256×256` 的网络输入坐标系中。

### 3.2 在本机（CPU 或单 GPU）训练

进入 `AU3605_FinalPro` 目录：

```bash
cd AU3605_FinalPro
python train_od_fct.py
```

日志示例：

```text
 ** Using device: cpu
Loading Training Set...
 ** Loading dataset from ...\dataset\OD_FCT\train\images and ...\train\targets.csv...
Found 330 entries in CSV.
Successfully loaded 330 images.
Loading Test/Validation Set...
...
 ** Start training OD/FCT model...
Epoch 1/100, Train Loss: ..., Val Loss: ...
```

训练好的模型会保存在：

- 最新模型：`logs/od_fct_model_latest.pth`  
- 最优模型：`logs/od_fct_model_best.pth`

如需在命令行中指定超参数，可以使用：

Linux / macOS（bash/zsh）：

```bash
python train_od_fct.py \
  --epochs 200 \
  --batch-size 16 \
  --lr 5e-5 \
  --weight-decay 1e-4 \
  --num-workers 2 \
  --lr-scheduler step \
  --step-size 40 \
  --gamma 0.1
```

Windows PowerShell（建议）：

```powershell
python train_od_fct.py --epochs 200 --batch-size 16 --lr 5e-5 --weight-decay 1e-4 --num-workers 2 --lr-scheduler step --step-size 40 --gamma 0.1
```

常用参数说明：

- `--epochs`：训练轮数  
- `--batch-size`：批大小  
- `--lr`：学习率  
- `--weight-decay`：L2 正则化强度  
- `--num-workers`：DataLoader 使用的进程数  
- `--lr-scheduler`：学习率调度方式（`none` 或 `step`）  
- `--step-size`：StepLR 的步长（多少个 epoch 衰减一次）  
- `--gamma`：每次衰减的比例  

### 3.3 使用单张指定 GPU 训练

在 Linux 或服务器上（bash/zsh）：

```bash
CUDA_VISIBLE_DEVICES=4 python train_od_fct.py
```

在 PowerShell（例如 Windows 服务器）：

```powershell
$env:CUDA_VISIBLE_DEVICES="4"
python .\train_od_fct.py
```

这两种方式都会让脚本只“看到”一张 GPU（逻辑上的 `cuda:0` 实际是物理 GPU 4），相当于单卡训练。

### 3.4 单机多卡训练（DataParallel）

当前脚本内已支持 `nn.DataParallel`：当可见 GPU 数量大于 1 时，会自动使用多卡并行。

示例（Linux）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_od_fct.py
```

日志中会看到类似提示：

```text
 ** Using device: cuda
 ** Using 4 GPUs with DataParallel
```

此时会在一个进程内使用多张 GPU 做数据并行训练，模型保存时会自动去掉 `DataParallel` 外壳，直接保存为普通 `state_dict`。

### 3.5 wandb 可视化

如果希望在线监控训练曲线，请确保已安装 `wandb` 并在本地登录。

1. 安装并登录 wandb：

```bash
pip install wandb
wandb login
# 按提示输入你的 API Key (从 https://wandb.ai/authorize 获取)
```

2. 运行训练脚本：

```bash
python train_od_fct.py
```

脚本检测到 `wandb` 库且已登录后，会自动初始化项目 `project="retina_od_fct"` 并记录训练过程。

如果未安装 `wandb` 或初始化失败，脚本将自动跳过 wandb 相关功能，仅在本地打印日志。

### 3.6 模型测试与评估

训练完成后，可以使用 `test_od_fct.py` 脚本加载最佳模型并在测试集上进行评估。

**基本用法：**

```bash
python test_od_fct.py
```

默认行为：
- 加载 `logs/od_fct_model_best.pth`。
- 在测试集上计算 OD 和 Fovea 的平均欧氏距离误差。
- 将前 20 张测试图片的可视化结果保存到 `results/` 目录。
  - **绿色点**：真实坐标 (Ground Truth)
  - **红色点**：预测坐标 (Prediction)

**高级用法：**

指定模型路径或设备：

```bash
# 指定加载最新模型
python test_od_fct.py --model_path logs/od_fct_model_latest.pth

# 强制使用 CPU
python test_od_fct.py --device cpu
```

---

## 4. 第二个模型：血管分割（预留说明）

第二个模型（血管分割）尚未实现完整训练脚本，已预留如下结构，便于未来扩展：

- 数据建议结构：`dataset/vessel_seg/`
  - `normal/`：来自 `Normal Retinal Images/` 或其他正常眼底照片及其血管标注。  
  - `adam_hoover/`：Adam Hoover 数据（通过 `tools/organize_hoover_ppm.py` 整理得到的 `images/` 与 `labels/`）。  
  - `other/`：其他来源的数据（例如 DRIVE/HRF 等）。

推荐后续在此基础上添加：

- `data_modules/vessel_seg_dataset.py`：血管分割任务的 Dataset（图像 + mask）。  
- `models/vessel_seg_net.py`：如 U-Net 等分割网络结构。  
- `train_vessel_seg.py`：与 `train_od_fct.py` 类似的训练入口，带 wandb 与多卡支持。

---

## 5. 数据处理工具脚本

工具脚本统一放在 `tools/` 目录下，均可在命令行直接调用。  

### 5.1 批量解压 `.gz` 文件

脚本：`tools/decompress_gz_dataset.py`

用途：  
- 从指定根目录开始，递归查找所有以 `.gz` 结尾的文件，并在同一目录下解压为同名但不带 `.gz` 的文件。  
- 如果解压目标文件已存在且 `overwrite=False`，则跳过。

示例（从 `AU3605_FinalPro` 根目录执行）：

```bash
python tools/decompress_gz_dataset.py
```

### 5.2 整理 Adam Hoover PPM 图像和标注

脚本：`tools/organize_hoover_ppm.py`

用途：  
- 针对 Adam Hoover 提供的血管数据，将：  
  - 原始 PPM 图像整理到 `images/` 目录；  
  - 对应手工标注的 PPM 图像整理到 `labels/` 目录。  

脚本内部假设的原始目录结构与 Adam Hoover 发布的数据一致，执行后会在指定根目录下生成 `images/` 和 `labels/`，便于后续统一构建分割任务的数据集。

### 5.3 OD/Fovea 空域对齐 + 颜色归一化 Demo

脚本：`align_and_normalize.py`（位于 `AU3605_FinalPro` 根目录）  

用途：  
- 使用已训练好的 OD/Fovea 检测模型，对单张或多张图像做：  
  - 视盘/黄斑方向的一致化（必要时自动左右翻转）；  
  - 视盘–黄斑中点的空域对齐；  
  - 颜色归一化到参考图 `utils/ref_img.jpg`。  
- 并通过 `matplotlib` 给出“参考图 – 原始图 – 处理后图像”的三图对比。

默认目录约定（由脚本自动创建）：  
- `alignment_demo_input/`：放入待处理的原始眼底图像；  
- `alignment_demo_output/`：脚本运行后，处理好的图像会保存到这里，文件名与原图一致。

基本用法（在 `AU3605_FinalPro` 目录下）：

```bash
python align_and_normalize.py
```

效果：  
- 自动加载 `logs/od_fct_model_best.pth` 作为检测模型；  
- 对 `alignment_demo_input/` 中的所有支持格式图像（`.jpg/.png/...`）进行处理；  
- 结果图像写入 `alignment_demo_output/`，并弹出一个三图对比窗口便于观察预处理效果。

---

## 6. 典型使用流程总结

1. 准备环境（Conda + PyTorch + OpenCV + pandas + tqdm）。  
2. 确认 `dataset/OD_FCT` 下的 `train` 和 `test` 图像与 `targets.csv` 完整无误。  
3. 在 `AU3605_FinalPro` 根目录运行：
   - 本机 CPU / 单卡：`python train_od_fct.py`  
   - 指定 GPU：`CUDA_VISIBLE_DEVICES=4 python train_od_fct.py`  
   - 单机多卡：`CUDA_VISIBLE_DEVICES=0,1,2,3 python train_od_fct.py`  
4. 如需在线监控训练曲线，安装 wandb 并设置 `WANDB_API_KEY` 后再次运行训练脚本。  
5. 训练完成后，在 `logs/` 中获取最佳模型权重 `od_fct_model_best.pth`。
6. 运行 `python test_od_fct.py` 评估模型性能并查看 `results/` 下的可视化效果。

第二个模型（血管分割）在整理好 `dataset/vessel_seg` 结构后，可以按照第一模型的模式增添相应的 Dataset、模型和训练脚本。
