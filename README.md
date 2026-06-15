# ERRNet: Single Image Reflection Removal — DIP26 Course Project

基于 ERRNet (CVPR 2019) 的单图像反射去除改进策略研究。复现 baseline 并系统尝试 5 个改进方向，在 6 个公开测试集 + 5 张自拍照片上评估。

> 原论文: [Single Image Reflection Removal Exploiting Misaligned Training Data and Network Enhancements](https://arxiv.org/abs/1904.00637)
>
> Fork from: https://github.com/innerway-xq/ERRNet

## 环境

- PyTorch 2.7.0, torchvision 0.22.0
- opencv-python-headless, scikit-image, h5py, tensorboardX, visdom
- 硬件: PPU-ZW810E (98GB), 需 `source /usr/local/PPU_SDK/envsetup.sh ppu`

## 数据准备

```bash
# 原始数据软链接
ln -s /mnt/dip26-raw-data/errnet_dip26/ERRNet/checkpoints checkpoints
ln -s /mnt/dip26-raw-data/errnet_dip26/ERRNet/datasets/raw_data datasets/raw_data

# 生成训练/测试数据
python datasets/prepare_test_data.py
python datasets/prepare_train_data.py
```

## 训练

### Baseline（60 epoch, ~7–8h）

```bash
python train_errnet.py --name errnet_baseline --hyper --gpu_ids 0 --nThreads 0
```

### 微调（从 baseline checkpoint）

```bash
# Edge-Aware Loss
python train_errnet_finetune.py --name errnet_edge_only --hyper --gpu_ids 0 \
    --icnn_path checkpoints/errnet_baseline/errnet_060_00463920.pt \
    --lambda_edge 0.05 --nThreads 0

# CBAM Attention
python train_errnet_finetune.py --name errnet_cbam --hyper --gpu_ids 0 \
    --inet errnet_cbam \
    --icnn_path checkpoints/errnet_baseline/errnet_060_00463920.pt

# Exclusion Loss
python train_errnet_finetune.py --name errnet_excl --hyper --gpu_ids 0 \
    --icnn_path checkpoints/errnet_baseline/errnet_060_00463920.pt \
    --lambda_exclusion 0.01 --nThreads 0

# SSIM Loss
python train_errnet_finetune.py --name errnet_ssim --hyper --gpu_ids 0 \
    --icnn_path checkpoints/errnet_baseline/errnet_060_00463920.pt \
    --lambda_ssim 0.1 --nThreads 0

# Predict R̂
python train_errnet_finetune.py --name errnet_predict_r --hyper --gpu_ids 0 \
    --icnn_path checkpoints/errnet_baseline/errnet_060_00463920.pt \
    --predict_reflection --lambda_exclusion 0.01 --lambda_ssim 0.1 --lambda_r_pixel 0.5
```

## 评估

```bash
python test_errnet.py --name <exp_name> --hyper --gpu_ids 0 --which_epoch latest
```

## 模型权重

- **百度网盘**: https://pan.baidu.com/s/1n67vzXjwMCZPL3W40yZ-Fw?pwd=dipz 提取码: dipz
- 包含: Baseline (`errnet_060_00463920.pt`) + Edge-Aware (`errnet_latest.pt`)

## 主要结果

### Baseline 复现

| 数据集 | PSNR (Ours) | PSNR (Pretrained) |
|--------|:---:|:---:|
| CEILNet Table2 | **28.10** | 27.88 |
| real20 | **23.63** | 23.55 |
| postcard | 21.51 | 22.07 |
| objects | 24.40 | 24.85 |
| wild | **25.41** | 25.18 |
| sir2_withgt | 23.53 | 23.88 |

### 改进方向汇总

| 方向 | 类别 | 结果 |
|------|------|------|
| Exclusion Loss | Loss | ▼ 全面下降（数学退化） |
| SSIM Loss | Loss | ▼ 全面下降 |
| Edge-Aware Loss | Loss | 3 升 3 降，无一致性 |
| CBAM Attention | 架构 | 仅 wild 提升 |
| Predict R̂ | 语义 | ▼▼ 灾难性退化 |

**核心发现**: 所有改进方向均未在 6 个测试集上取得一致提升。VGG Perceptual Loss 梯度主导（~90%）、ERRNet 单输出架构的根本约束、以及微调训练不足是三条共性失败根因。

### 自拍照片测试

5 张现实玻璃反射场景照片，无配对 GT，定性分析。可视化结果见 `results/mypic/visualizations/`。

## 文件结构

```
├── models/
│   ├── arch/__init__.py          # 网络工厂（含 errnet_cbam）
│   ├── arch/default.py           # DRNet, SE, CBAM, EdgeMap
│   ├── errnet_model.py           # 主模型（baseline + 全部改进）
│   └── losses.py                 # Exclusion, SSIM, EdgeAware Loss
├── options/errnet/train_options.py  # 训练参数
├── train_errnet.py               # 从头训练脚本
├── train_errnet_finetune.py      # 微调脚本（从 checkpoint）
├── test_errnet.py                # 评估脚本
├── visualize_mypic.py            # 自拍照片可视化
├── mypic/                        # 5 张自拍原始照片
└── results/mypic/
    ├── mypic_baseline/           # Baseline 模型输出
    ├── mypic_edge/               # Edge-Aware 模型输出
    └── visualizations/           # 并排对比图 + 残差图
```

## 参考

- [1] Li & Brown, "Single Image Layer Separation Using Relative Smoothness", CVPR 2014.
- [2] Fan et al., "A Generic Deep Architecture for Single Image Reflection Removal and Smoothing", ICCV 2017.
- [3] Zhang et al., "Single Image Reflection Separation with Perceptual Losses", CVPR 2018.
- [4] Wei et al., "Single Image Reflection Removal Exploiting Misaligned Training Data and Network Enhancements", CVPR 2019.
- [5] Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
- [6] Wan et al., "Benchmarking Single-Image Reflection Removal Algorithms", ICCV 2017.
