# AU3605_FinalPro 视网膜图像处理项目

本项目主要覆盖以下 5 个任务（对应课程大作业的核心流程）：

1. 视盘中心检测（OD Center）
2. 黄斑中心检测（Fovea Center）
3. 颜色归一化、空域对齐（相对容易）
4. 血管分割（Vessel Segmentation）
5. 血管区域光滑填充（去血管 + Inpainting）

下文按上述任务给出目录结构与脚本用法，默认在 `AU3605_FinalPro` 目录下运行命令。

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
  - `Vessel/`  
    第二个模型（血管分割）使用的数据根目录（已整理）：  
    - `training/`：训练集（按数据来源分子目录）。  
    - `test/`：测试/验证集（统一放在 `images/` 与 `targets/` 下）。

- `logs/`  
  存放训练过程中保存的模型权重（`od_fct_model_best.pth`, `od_fct_model_latest.pth`）。

- `results_od_fct/`  
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
- `--device`：运行设备（`auto/cpu/cuda`，默认 `auto`，自动优先用 GPU）  
- `--lr-scheduler`：学习率调度方式（`none` 或 `step`）  
- `--step-size`：StepLR 的步长（多少个 epoch 衰减一次）  
- `--gamma`：每次衰减的比例  
- `--load-pretrained`：是否加载预训练权重（flag，默认不启用）  
- `--pretrained-path`：预训练权重路径（配合 `--load-pretrained` 使用，默认 `None`）

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
- 将前 20 张测试图片的可视化结果保存到 `results_od_fct/` 目录。
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

参数说明：

- `--model_path`：加载的 OD/FCT 模型权重路径（默认 `logs/od_fct_model_best.pth`）  
- `--device`：运行设备（`cuda` 或 `cpu`，默认 `cuda`；若本机无 CUDA 会自动回退到 `cpu`）

---

## 4. 第二个模型：血管分割

第二个模型实现了典型的 U-Net 结构，用于眼底血管分割任务，训练与评估脚本均已就绪。

数据集目录约定为 `dataset/Vessel/`，内部结构如下：

- `dataset/Vessel/training/`
  - `DRIVE/images/*.tif` 与 `DRIVE/targets/*_manual1.gif`
  - `AdamHoover/images/*.ppm` 与 `AdamHoover/targets/*.ah.ppm`
  - `HRF/images/*_dr.JPG` 与 `HRF/targets/*_dr.tif`
- `dataset/Vessel/test/`
  - `images/*_test.tif`
  - `targets/*_manual1.gif`

`data_modules/vessel_seg_dataset.py` 会自动将 `training` 下的三个子目录当作一个统一的大训练集来处理，无需手动区分来源数据。

### 4.1 模型结构

- 模型文件：`models/vessel_seg_net.py`  
- 网络结构：`VesselSegNet`，为 3 层下采样的 U-Net：
  - 编码端：`DoubleConv` + `MaxPool2d` 堆叠；
  - 解码端：`ConvTranspose2d` 上采样 + skip connection + `DoubleConv`；
  - 输出层：`1×1` 卷积输出单通道血管概率图。

损失函数采用 `BCEWithLogitsLoss`，训练和验证阶段同时计算 Dice 系数作为性能指标：  
- `loss`（BCE）在训练时被最小化，用于优化逐像素的分类概率；  
- `Dice` 作为评估指标，取值范围约为 `[0, 1]`，越接近 1 表示预测血管区域与真实标注的重叠越好。

### 4.2 训练脚本 `train_vessel_seg.py`

在 `DIPfp` 目录下运行血管分割训练脚本：

```bash
python train_vessel_seg.py --device cpu --epochs 40 --batch-size 2 --img-size 256 --lr 1e-4 --weight-decay 1e-4 --num-workers 0 --lr-scheduler step --step-size 20 --gamma 0.1
```

上述是一套适合在普通 Windows CPU 上训练的推荐参数：

- 使用 CPU 进行训练（`--device cpu`）；  
- 训练 40 个 epoch；  
- batch size 设为 2，兼顾速度与内存占用；  
- 输入统一缩放到 `256×256`；  
- 初始学习率 `1e-4`，权重衰减 `1e-4`；  
- 使用 `StepLR` 学习率调度器，在第 20 个 epoch 将学习率衰减为原来的 `0.1`。

如需快速小测试，可适当减小 `--epochs`（例如 5 或 10）来验证流程。

参数说明：

- `--epochs`：训练轮数（默认 `80`）  
- `--batch-size`：批大小（默认 `4`）  
- `--img-size`：输入图像缩放到的边长（默认 `256`，实际输入为 `img-size×img-size`）  
- `--lr`：学习率（默认 `1e-4`）  
- `--weight-decay`：权重衰减（默认 `1e-4`）  
- `--num-workers`：DataLoader 使用的进程数（默认 `2`）  
- `--device`：运行设备（`auto/cpu/cuda`，默认 `auto`，自动优先用 GPU）  
- `--lr-scheduler`：学习率调度方式（`none` 或 `step`，默认 `step`）  
- `--step-size`：StepLR 的步长（默认 `30`）  
- `--gamma`：每次衰减的比例（默认 `0.1`）

训练过程中会自动：

- 读取 `dataset/Vessel/training` 作为训练集；  
- 读取 `dataset/Vessel/test` 作为验证集；  
- 在每个 epoch 结束后，保存：
  - 最新模型到 `logs/vessel_seg_model_latest.pth`；  
  - 验证集损失最优的模型到 `logs/vessel_seg_model_best.pth`。

如果环境中安装并登录了 `wandb`，脚本会自动将训练曲线同步到 `retina_vessel_seg` 项目；若未配置，脚本会自动忽略 wandb，仅在本地打印日志。

### 4.3 测试脚本 `test_vessel_seg.py`

训练完成后，可以使用 `test_vessel_seg.py` 对血管分割模型在测试集上的性能进行评估，并生成可视化结果。

基本用法：

```bash
python test_vessel_seg.py --device cpu
```

默认行为：

- 加载 `logs/vessel_seg_model_best.pth`；  
- 在 `dataset/Vessel/test` 上计算平均 Dice 系数；  
- 将前 20 张测试样本的可视化结果保存到 `results_vessel_seg/` 目录：
  - 左：原始眼底图像；  
  - 中：原图叠加真实血管标注；  
  - 右：原图叠加模型预测血管图。

同样可以通过参数指定模型路径或设备，例如：

```bash
python test_vessel_seg.py --model_path logs/vessel_seg_model_latest.pth --device cpu
```

参数说明：

- `--model_path`：加载的血管分割模型权重路径（默认 `logs/vessel_seg_model_best.pth`）  
- `--device`：运行设备（`cuda` 或 `cpu`，默认 `cuda`；若本机无 CUDA 会自动回退到 `cpu`）

### 4.4 血管区域光滑填充（去血管 + Inpainting）

在血管分割得到血管区域后，可以进一步对血管区域做光滑填充，从而得到“去血管”的结果图。

脚本：`vessel_inpaint.py`

基本用法：

```bash
python vessel_inpaint.py --device cpu
```

默认行为：

- 输入目录：`dataset/Vessel/test/images`（或传入 `dataset/Vessel/test` 时自动找到 `images/` 子目录）
- 加载模型：`logs/vessel_seg_model_best.pth`
- 输出目录：`results_vessel_inpaint/`（如不存在会自动创建）
- 每张图输出一张三联图（`*_vessel_inpaint.jpg`）：
  - 左：原图
  - 中：原图叠加预测血管（细红线透明叠加）
  - 右：对血管区域做 inpaint 后的结果

常用参数：

```bash
python vessel_inpaint.py --input-dir dataset/Vessel/test --output-dir results_vessel_inpaint --model-path logs/vessel_seg_model_best.pth --threshold 0.5 --dilate 2 --inpaint-radius 3 --limit 0
```

参数说明：

- `--input-dir`：输入目录（默认 `dataset/Vessel/test`；若存在 `images/` 子目录会自动读取 `images/`）  
- `--output-dir`：输出目录（默认 `results_vessel_inpaint`，不存在会自动创建）  
- `--model-path`：血管分割模型权重（默认 `logs/vessel_seg_model_best.pth`）  
- `--device`：运行设备（`auto/cpu/cuda`，默认 `auto`，自动优先用 GPU）  
- `--img-size`：推理时缩放到的边长（默认 `256`）  
- `--threshold`：血管二值化阈值（默认 `0.5`，越大预测血管越“保守”）  
- `--dilate`：对用于修补的血管 mask 做膨胀的半径（默认 `2`；只影响右图的 inpaint 区域，不影响中间红线显示）  
- `--overlay-alpha`：中间图红色血管叠加透明度（默认 `0.45`）  
- `--inpaint-radius`：OpenCV inpaint 半径（默认 `3`；越大修补越“平滑/模糊”）  
- `--inpaint-method`：inpaint 方法（`telea` 或 `ns`，默认 `telea`）  
- `--limit`：最多处理多少张图片（默认 `0` 表示不限制）

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

脚本内参数说明（无命令行参数，直接在代码里配置）：

- `decompress_gz_root(root_dir, overwrite=False)`  
  - `root_dir`：要递归查找并解压 `.gz` 的根目录  
  - `overwrite`：若目标文件已存在，是否覆盖（默认 `False`，不覆盖并跳过）

脚本默认会将 `root_dir` 设为 `tools/` 的上一级目录（即整个 `DIPfp`），如需只处理某个数据目录，可在脚本末尾修改 `target_dir`。

### 5.2 整理 Adam Hoover PPM 图像和标注

脚本：`tools/organize_hoover_ppm.py`

用途：  
- 针对 Adam Hoover 提供的血管数据，将：  
  - 原始 PPM 图像整理到 `images/` 目录；  
  - 对应手工标注的 PPM 图像整理到 `labels/` 目录。  

脚本内部假设的原始目录结构与 Adam Hoover 发布的数据一致，执行后会在指定根目录下生成 `images/` 和 `labels/`，便于后续统一构建分割任务的数据集。

脚本内参数说明（无命令行参数，直接在代码里配置）：

- `organize_hoover(base_dir)`  
  - `base_dir`：Adam Hoover 原始数据根目录  

脚本默认将 `base_dir` 指向 `tools/` 的上一级目录下的 `大作业二-血管标注`，如需更换位置可修改脚本末尾的 `hoover_root`。

### 5.3 颜色归一化、空域对齐（任务 3）

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

参数说明：

- `--images-dir`：输入图像目录（默认 `None`；为 `None` 时使用 `alignment_demo_input/`）  
- `--output-dir`：输出图像目录（默认 `None`；为 `None` 时使用 `alignment_demo_output/`）  
- `--model-path`：OD/FCT 检测模型权重（默认 `logs/od_fct_model_best.pth`）  
- `--device`：运行设备（`cpu` 或 `cuda`，默认 `cpu`；`cuda` 需要本机可用 CUDA）  
- `--img-size`：对齐与归一化处理的目标边长（默认 `256`）

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
6. 运行 `python test_od_fct.py` 评估模型性能并查看 `results_od_fct/` 下的可视化效果。
7. 整理 `dataset/Vessel` 目录结构，并在 `DIPfp` 目录运行：
   - `python train_vessel_seg.py --device cpu --epochs 40 --batch-size 2 --img-size 256 --lr 1e-4 --weight-decay 1e-4 --num-workers 0 --lr-scheduler step --step-size 20 --gamma 0.1`
   - `python test_vessel_seg.py --device cpu`
   - `python vessel_inpaint.py --device cpu`
