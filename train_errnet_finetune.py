"""Fine-tune ERRNet with improved losses from a pretrained checkpoint.

Usage:
    # SSIM only
    python train_errnet_finetune.py --name errnet_ssim --hyper --gpu_ids 0 --nThreads 0 \
        --icnn_path checkpoints/errnet_baseline/errnet_060_00463920.pt \
        --lambda_exclusion 0 --lambda_ssim 0.1

    # Exclusion only
    python train_errnet_finetune.py --name errnet_excl --hyper --gpu_ids 0 --nThreads 0 \
        --icnn_path checkpoints/errnet_baseline/errnet_060_00463920.pt \
        --lambda_exclusion 0.01 --lambda_ssim 0

    # Combined
    python train_errnet_finetune.py --name errnet_combined --hyper --gpu_ids 0 --nThreads 0 \
        --icnn_path checkpoints/errnet_baseline/errnet_060_00463920.pt \
        --lambda_exclusion 0.01 --lambda_ssim 0.1

    # Residual-based reflection prediction (predict R̂, derive T̂ = I - R̂)
    python train_errnet_finetune.py --name errnet_predict_r --hyper --gpu_ids 0 --nThreads 0 \
        --icnn_path checkpoints/errnet_baseline/errnet_060_00463920.pt \
        --predict_reflection \
        --lambda_exclusion 0.01 --lambda_ssim 0.1 --lambda_r_pixel 0.5
"""
from os.path import join
from options.errnet.train_options import TrainOptions
from engine import Engine
from data.image_folder import read_fns
import torch.backends.cudnn as cudnn
import data.reflect_dataset as datasets
import util.util as util
import data

opt = TrainOptions().parse()

cudnn.benchmark = True
opt.display_freq = 10

if opt.debug:
    opt.display_id = 1
    opt.display_freq = 20
    opt.print_freq = 20
    opt.nEpochs = 40
    opt.max_dataset_size = 100
    opt.no_log = False
    opt.nThreads = 0
    opt.decay_iter = 0
    opt.serial_batches = True
    opt.no_flip = True

# ---- data ----
datadir = './datasets/processed_data'
datadir_syn = join(datadir, 'VOCdevkit/VOC2012/PNGImages')
datadir_real = join(datadir, 'real_train')

train_dataset = datasets.CEILDataset(
    datadir_syn, read_fns('VOC2012_224_train_png.txt'), size=opt.max_dataset_size, enable_transforms=True,
    low_sigma=opt.low_sigma, high_sigma=opt.high_sigma,
    low_gamma=opt.low_gamma, high_gamma=opt.high_gamma)

train_dataset_real = datasets.CEILTestDataset(datadir_real, enable_transforms=True)
train_dataset_fusion = datasets.FusionDataset([train_dataset, train_dataset_real], [0.7, 0.3])

train_dataloader_fusion = datasets.DataLoader(
    train_dataset_fusion, batch_size=opt.batchSize, shuffle=not opt.serial_batches,
    num_workers=opt.nThreads, pin_memory=True)

eval_dataset_ceilnet = datasets.CEILTestDataset(join(datadir, 'testdata_CEILNET_table2'))
eval_dataset_real = datasets.CEILTestDataset(join(datadir, 'real20'), size=20, max_long_edge=512)

eval_dataloader_ceilnet = datasets.DataLoader(
    eval_dataset_ceilnet, batch_size=1, shuffle=False, num_workers=opt.nThreads, pin_memory=True)
eval_dataloader_real = datasets.DataLoader(
    eval_dataset_real, batch_size=1, shuffle=False, num_workers=opt.nThreads, pin_memory=True)

# ---- finetuning strategy (set before Engine init) ----
# No GAN for fine-tuning — avoid creating unused netD/optimizer_D
opt.lambda_gan = 0

# ---- engine ----
engine = Engine(opt)


def set_learning_rate(lr):
    for optimizer in engine.model.optimizers:
        print('[i] set learning rate to {}'.format(lr))
        util.set_opt_param(optimizer, 'lr', lr)


# Reset epoch counter (was 60 from full training checkpoint)
engine.epoch = 0
engine.iterations = 0

total_epochs = opt.nEpochs if opt.nEpochs != 60 else (30 if opt.predict_reflection else 15)
set_learning_rate(5e-5)

print(f'[i] Fine-tuning for {total_epochs} epochs with lr={5e-5}')
print(f'    lambda_exclusion={opt.lambda_exclusion}, lambda_ssim={opt.lambda_ssim}, '
      f'predict_reflection={opt.predict_reflection}, lambda_r_pixel={opt.lambda_r_pixel}')

# Eval before training
engine.eval(eval_dataloader_ceilnet, dataset_name='testdata_table2')
engine.eval(eval_dataloader_real, dataset_name='testdata_real20')

while engine.epoch < total_epochs:
    engine.train(train_dataloader_fusion)

    if engine.epoch % 2 == 0 or engine.epoch == 1:
        engine.eval(eval_dataloader_ceilnet, dataset_name='testdata_table2')
        engine.eval(eval_dataloader_real, dataset_name='testdata_real20')

# Final eval after training completes
engine.eval(eval_dataloader_ceilnet, dataset_name='testdata_table2')
engine.eval(eval_dataloader_real, dataset_name='testdata_real20')
