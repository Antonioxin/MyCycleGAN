# MyCycleGAN（horse2zebra）复现指南（训练 / 测试）

本仓库是一个经过 **GPU 吞吐优化** 的 CycleGAN 实现，用于 **horse2zebra** 数据集训练与推理。
支持在 NVIDIA GPU 上使用 **AMP(fp16)**、**TF32**、**channels_last**、可选 **torch.compile** 来提升训练/推理速度。

> 重要：本仓库默认 **不上传训练集、训练结果、权重文件**（已通过 `.gitignore` 排除）。

---

## 0. 项目结构说明

训练完成后常见目录结构如下：

```
MyCycleGAN/
  data/horse2zebra/         # 数据集（不提交到 GitHub）
    trainA/ trainB/ testA/ testB/
  models/
  datasets/
  utils/
  train.py
  test.py
  environment.yml
  checkpoints_horse2zebra/  # 权重（不提交到 GitHub）
  results_*/                # 输出图像（不提交到 GitHub）
```

> ⚠️ `data/`、`checkpoints_*`、`results_*`、`logs/` 等均不会被提交到 GitHub。

---

## 1. 环境配置（推荐 Conda）

### 方式 A（推荐）：Conda 创建环境

```bash
conda env create -f environment.yml
conda activate cyclegan-env
python -V
```

建议使用 **Python 3.10**。
本项目代码也兼容较低版本 Python（我们已移除一些高版本专属语法）。

---

### NVIDIA GPU 用户（推荐执行：安装 CUDA 版 PyTorch）

如果你的机器有 NVIDIA GPU，建议安装 CUDA wheel（更通用、更不容易 conda 解依赖失败）：

```bash
pip uninstall -y torch torchvision
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.3.1 torchvision==0.18.1
```

快速确认 CUDA 是否可用：

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available()); print('torch:', torch.__version__)"
```

---

### CPU 用户（无 GPU）

无需额外操作。训练会很慢，但推理可跑通。

---

## 2. 数据集准备（horse2zebra）

### 必须满足的目录结构

请把数据集放到：

```
./data/horse2zebra/
  trainA/
  trainB/
  testA/
  testB/
```

* A 域：horse（马）
* B 域：zebra（斑马）

快速检查：

```bash
ls -la ./data/horse2zebra
ls -la ./data/horse2zebra/trainA | head
ls -la ./data/horse2zebra/trainB | head
```

---

## 3. 训练（Train）

### 推荐 GPU 训练命令（稳定 + 快）

> 注意：请使用 `fp16`（不要用 `bf16`），因为某些算子（如 ReflectionPad2d）在部分 torch/cuda 组合下不支持 bf16。

```bash
python train.py \
  --data-root ./data/horse2zebra \
  --epochs 200 \
  --batch-size 4 \
  --lr 0.0002 \
  --lambda-cycle 10.0 \
  --lambda-idt 0.5 \
  --pool-size 50 \
  --load-size 286 \
  --crop-size 256 \
  --device cuda \
  --tf32 \
  --amp --amp-dtype fp16 \
  --channels-last \
  --fused-adam \
  --compile --compile-mode max-autotune \
  --num-workers 8 \
  --prefetch-factor 4 \
  --pin-memory \
  --drop-last \
  --print-freq 200 \
  --save-freq 10 \
  --checkpoints-dir ./checkpoints_horse2zebra \
  --save-optim
```

训练过程中会生成 checkpoint：

```
./checkpoints_horse2zebra/epoch_10.pth
./checkpoints_horse2zebra/epoch_20.pth
...
./checkpoints_horse2zebra/epoch_200.pth
```

---

### CPU 训练命令（慢，但能跑通）

```bash
python train.py \
  --data-root ./data/horse2zebra \
  --epochs 10 \
  --batch-size 1 \
  --device cpu \
  --no-amp --no-tf32 --no-compile \
  --num-workers 2 \
  --no-pin-memory \
  --checkpoints-dir ./checkpoints_cpu
```

---

## 4. 测试 / 推理（Test / Inference）

### A→B（horse → zebra），输出对比图（推荐）

该模式会保存 **[输入 | 输出]** 拼接图（左 real，右 fake），非常适合肉眼对比与写报告：

```bash
python test.py \
  --data-root ./data/horse2zebra \
  --checkpoint ./checkpoints_horse2zebra/epoch_200.pth \
  --save-dir ./results_epoch200_AtoB \
  --direction AtoB \
  --phase test \
  --batch-size 16 \
  --num-workers 8 --prefetch-factor 4 \
  --amp --amp-dtype fp16 \
  --channels-last \
  --save-mode pair \
  --max-images 200
```

输出在：

```
results_epoch200_AtoB/
  AtoB/
    000000_AtoB_pair.png
    000001_AtoB_pair.png
    ...
```

---

### 双向导出（A↔B）

```bash
python test.py \
  --data-root ./data/horse2zebra \
  --checkpoint ./checkpoints_horse2zebra/epoch_200.pth \
  --save-dir ./results_epoch200_both \
  --direction both \
  --phase test \
  --batch-size 16 \
  --num-workers 8 --prefetch-factor 4 \
  --amp --amp-dtype fp16 \
  --channels-last \
  --save-mode pair \
  --max-images 200
```

会得到：

* `results_epoch200_both/AtoB/`（horse→zebra）
* `results_epoch200_both/BtoA/`（zebra→horse）

---

## 5. 断点续训（Resume）

示例：从 `epoch_100.pth` 继续训练（建议换一个 checkpoints 目录避免覆盖）：

```bash
python train.py \
  --data-root ./data/horse2zebra \
  --epochs 200 \
  --batch-size 4 \
  --lr 0.0002 \
  --lambda-cycle 10.0 \
  --lambda-idt 0.5 \
  --pool-size 50 \
  --load-size 286 \
  --crop-size 256 \
  --device cuda \
  --tf32 \
  --amp --amp-dtype fp16 \
  --channels-last \
  --fused-adam \
  --compile --compile-mode max-autotune \
  --num-workers 8 \
  --prefetch-factor 4 \
  --pin-memory \
  --drop-last \
  --print-freq 200 \
  --save-freq 10 \
  --checkpoints-dir ./checkpoints_horse2zebra_resume \
  --save-optim \
  --resume ./checkpoints_horse2zebra/epoch_100.pth \
  --resume-optim
```

---

## 6. 性能建议（GPU）

* 如果 `nvidia-smi` 看到 **GPU-Util 接近 100%**：说明 GPU 已被喂饱（目标达成）。
* 如果 GPU 利用率偏低，可尝试：

  * 增大 `--num-workers`（比如 8/12）
  * 增大 `--prefetch-factor`（比如 4）
  * 显存足够时适当增大 `--batch-size`（注意 GAN 稳定性需观察输出效果）
* `--compile` 前几步会有 warm-up，后续通常更快。
* 强烈建议 `--amp-dtype fp16`（兼容性与速度更稳）。

---

## 7. 常见报错与解决

### 7.1 数据集路径错误：`Directory not found: .../trainA`

说明你的 `--data-root` 指向的目录下没有 `trainA/trainB/testA/testB`。
请确保结构为：

```
data/horse2zebra/trainA
data/horse2zebra/trainB
data/horse2zebra/testA
data/horse2zebra/testB
```

---

### 7.2 BF16 报错：`reflection_pad2d ... not implemented for 'BFloat16'`

请改用 fp16：

* 训练：`--amp-dtype fp16`
* 测试：`--amp-dtype fp16`

---

### 7.3 GPU 利用率低/卡数据

* 提升 `--num-workers` / `--prefetch-factor`
* CUDA 下开启 `--pin-memory`
* 确保数据在本地盘/高速盘（远程盘会拖慢）

---

## 8. 复现实验建议（可选）

如需更强复现性（牺牲少量速度），可设随机种子：

```bash
python train.py ... --seed 42
```

---

## 说明 / 致谢

* CycleGAN 经典结构：ResNet Generator + PatchGAN Discriminator + LSGAN Loss。
* 数据集：horse2zebra（CycleGAN 常用公开数据集）。
