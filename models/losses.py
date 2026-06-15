import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
import numpy as np
from torch import autograd
import torchvision.models as models
import util.util as util
from models.vgg import Vgg19
from torch.autograd import Function
from models.CX import CX_loss

###############################################################################
# Functions
###############################################################################
def compute_gradient(img):
    gradx=img[...,1:,:]-img[...,:-1,:]
    grady=img[...,1:]-img[...,:-1]
    return gradx,grady


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.loss = nn.L1Loss()
    
    def forward(self, predict, target):
        predict_gradx, predict_grady = compute_gradient(predict)
        target_gradx, target_grady = compute_gradient(target) 
        
        return self.loss(predict_gradx, target_gradx) + self.loss(predict_grady, target_grady)


class ExclusionLoss(nn.Module):
    """Gradient exclusion loss: minimize correlation between transmission
    and reflection gradients to enforce edge separation between layers.

    T_hat: predicted transmission (from forward)
    R_hat: predicted reflection (= input - T_hat)
    """
    def forward(self, trans, refl):
        # x-direction gradients: [B, C, H, W-1]
        grad_tx = trans[..., 1:, :] - trans[..., :-1, :]
        grad_rx = refl[..., 1:, :] - refl[..., :-1, :]
        # y-direction gradients: [B, C, H-1, W]
        grad_ty = trans[..., 1:] - trans[..., :-1]
        grad_ry = refl[..., 1:] - refl[..., :-1]

        # crop to common spatial size [B, C, H-1, W-1]
        grad_tx = grad_tx[..., :-1]
        grad_rx = grad_rx[..., :-1]
        grad_ty = grad_ty[..., :-1, :]
        grad_ry = grad_ry[..., :-1, :]

        # per-pixel gradient norm
        norm_t = torch.sqrt(grad_tx ** 2 + grad_ty ** 2 + 1e-6)
        norm_r = torch.sqrt(grad_rx ** 2 + grad_ry ** 2 + 1e-6)

        # normalized gradient correlation (cosine similarity of gradient directions)
        loss_x = torch.abs(grad_tx * grad_rx) / (norm_t * norm_r + 1e-6)
        loss_y = torch.abs(grad_ty * grad_ry) / (norm_t * norm_r + 1e-6)

        return (loss_x + loss_y).mean()


class SSIMLoss(nn.Module):
    """1 - SSIM loss. Simplified implementation using average pooling as
    a Gaussian window approximation (window_size=11)."""
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.pool = nn.AvgPool2d(window_size, stride=1)

    def forward(self, predict, target):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = self.pool(predict)
        mu_y = self.pool(target)

        sigma_x = self.pool(predict ** 2) - mu_x ** 2
        sigma_y = self.pool(target ** 2) - mu_y ** 2
        sigma_xy = self.pool(predict * target) - mu_x * mu_y

        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))

        return (1 - ssim).mean()


class EdgeAwareLoss(nn.Module):
    """Edge-aware L1 loss with gradient term — penalizes errors in edge
    regions more heavily to improve fine structure preservation.

    Uses a pre-computed edge map (from the EdgeMap module) as spatial
    weights for both:
      - pixel-level L1:  mean((1 + α · edge) ⊙ |pred - target|)
      - gradient L1:     mean((1 + α · edge) ⊙ (|∇x_err| + |∇y_err|))

    Args:
        alpha: edge weight multiplier (higher = more edge emphasis)
        grad_weight: relative weight of gradient term vs pixel term
    """
    def __init__(self, alpha=2.0, grad_weight=0.5):
        super().__init__()
        self.alpha = alpha
        self.grad_weight = grad_weight

    def forward(self, predict, target, edge_map):
        # edge_map: [B, 1, H, W], values in [0, ~2] depending on image intensity
        weight = 1.0 + self.alpha * edge_map

        # --- pixel L1 with edge weighting ---
        pixel_diff = (predict - target).abs()          # [B, C, H, W]
        loss_pixel = (pixel_diff * weight).mean()

        # --- gradient L1 with edge weighting ---
        pred_grad_x = predict[..., 1:, :] - predict[..., :-1, :]
        pred_grad_y = predict[..., 1:] - predict[..., :-1]
        targ_grad_x = target[..., 1:, :] - target[..., :-1, :]
        targ_grad_y = target[..., 1:] - target[..., :-1]

        # crop weight to match gradient size
        # grad_x: gradient along H dim, shape [B, C, H-1, W]
        # grad_y: gradient along W dim, shape [B, C, H, W-1]
        w_grad_x = weight[..., :-1, :]                 # [B, 1, H-1, W]
        w_grad_y = weight[..., :-1]                    # [B, 1, H, W-1]

        loss_grad_x = ((pred_grad_x - targ_grad_x).abs() * w_grad_x).mean()
        loss_grad_y = ((pred_grad_y - targ_grad_y).abs() * w_grad_y).mean()
        loss_grad = loss_grad_x + loss_grad_y

        return loss_pixel + self.grad_weight * loss_grad


class MultipleLoss(nn.Module):
    def __init__(self, losses, weight=None):
        super(MultipleLoss, self).__init__()
        self.losses = nn.ModuleList(losses)
        self.weight = weight or [1/len(self.losses)] * len(self.losses)
    
    def forward(self, predict, target):
        total_loss = 0
        for weight, loss in zip(self.weight, self.losses):
            total_loss += loss(predict, target) * weight
        return total_loss


class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        """norm (bool): normalize/denormalize the stats"""
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False


class VGGLoss(nn.Module):
    def __init__(self, vgg=None, weights=None, indices=None, normalize=True):
        super(VGGLoss, self).__init__()        
        if vgg is None:
            self.vgg = Vgg19()
        else:
            self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.weights = weights or [1.0/2.6, 1.0/4.8, 1.0/3.7, 1.0/5.6, 10/1.5]
        self.indices = indices or [2, 7, 12, 21, 30]
        device = next(self.vgg.parameters()).device
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).to(device)
        else:
            self.normalize = None

    def forward(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        
        return loss


class CXLoss(VGGLoss):
    # Contextual Loss from
    # https://arxiv.org/abs/1803.02077
    def __init__(self, vgg=None, weights=None, indices=None, criterions=None):        
        super(CXLoss, self).__init__(vgg, weights, indices)
        self.criterions = criterions or [CX_loss] * (len(weights))
    
    def forward(self, x, y):
        x = self.normalize(x)
        y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterions[i](x_vgg[i], y_vgg[i].detach())
        
        loss = loss[0] if loss.dim() == 1 else loss
        return loss


class ContentLoss():
    def initialize(self, loss):
        self.criterion = loss

    def get_loss(self, fakeIm, realIm):
        return self.criterion(fakeIm, realIm)


class GANLoss(nn.Module):
    def __init__(self, use_l1=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_l1:
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.BCEWithLogitsLoss() # absorb sigmoid into BCELoss

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = real_tensor
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = fake_tensor
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input, list):
            loss = 0
            for input_i in input:
                target_tensor = self.get_target_tensor(input_i, target_is_real)
                loss += self.loss(input_i, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input, target_is_real)
            return self.loss(input, target_tensor)


class DiscLoss():
    def name(self):
        return 'SGAN'

    def initialize(self, opt, tensor):
        self.criterionGAN = GANLoss(use_l1=False, tensor=tensor)

    def get_g_loss(self, net, realA, fakeB, realB):
        # First, G(A) should fake the discriminator
        pred_fake = net.forward(fakeB)
        return self.criterionGAN(pred_fake, 1)

    def get_loss(self, net, realA=None, fakeB=None, realB=None):
        pred_fake = None
        pred_real = None
        loss_D_fake = 0
        loss_D_real = 0
        # Fake
        # stop backprop to the generator by detaching fake_B
        # Generated Image Disc Output should be close to zero

        if fakeB is not None:
            pred_fake = net.forward(fakeB.detach())
            loss_D_fake = self.criterionGAN(pred_fake, 0)

        # Real
        if realB is not None:
            pred_real = net.forward(realB)
            loss_D_real = self.criterionGAN(pred_real, 1)

        # Combined loss
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        return loss_D, pred_fake, pred_real


class DiscLossR(DiscLoss):
    # RSGAN from 
    # https://arxiv.org/abs/1807.00734        
    def name(self):
        return 'RSGAN'

    def initialize(self, opt, tensor):
        DiscLoss.initialize(self, opt, tensor)
        self.criterionGAN = GANLoss(use_l1=False, tensor=tensor)

    def get_g_loss(self, net, realA, fakeB, realB, pred_real=None):
        if pred_real is None:
            pred_real = net.forward(realB)
        pred_fake = net.forward(fakeB)
        return self.criterionGAN(pred_fake - pred_real, 1)

    def get_loss(self, net, realA, fakeB, realB):
        pred_real = net.forward(realB)
        pred_fake = net.forward(fakeB.detach())

        loss_D = self.criterionGAN(pred_real - pred_fake, 1) # BCE_stable loss
        return loss_D, pred_fake, pred_real


class DiscLossRa(DiscLoss):
    # RaSGAN from 
    # https://arxiv.org/abs/1807.00734    
    def name(self):
        return 'RaSGAN'

    def initialize(self, opt, tensor):
        DiscLoss.initialize(self, opt, tensor)
        self.criterionGAN = GANLoss(use_l1=False, tensor=tensor)

    def get_g_loss(self, net, realA, fakeB, realB, pred_real=None):
        if pred_real is None:
            pred_real = net.forward(realB)
        pred_fake = net.forward(fakeB)

        loss_G = self.criterionGAN(pred_real - torch.mean(pred_fake, dim=0, keepdim=True), 0)
        loss_G += self.criterionGAN(pred_fake - torch.mean(pred_real, dim=0, keepdim=True), 1)
        return loss_G * 0.5

    def get_loss(self, net, realA, fakeB, realB):
        pred_real = net.forward(realB)
        pred_fake = net.forward(fakeB.detach())
        
        loss_D = self.criterionGAN(pred_real - torch.mean(pred_fake, dim=0, keepdim=True), 1)
        loss_D += self.criterionGAN(pred_fake - torch.mean(pred_real, dim=0, keepdim=True), 0)
        return loss_D * 0.5, pred_fake, pred_real


def init_loss(opt, tensor):
    disc_loss = None
    content_loss = None

    loss_dic = {}

    pixel_loss = ContentLoss()
    pixel_loss.initialize(MultipleLoss([nn.MSELoss(), GradientLoss()], [0.2,0.4]))

    loss_dic['t_pixel'] = pixel_loss
    loss_dic['r_pixel'] = pixel_loss

    # --- improved losses ---
    if getattr(opt, 'lambda_exclusion', 0) > 0:
        exclusion_loss = ContentLoss()
        exclusion_loss.initialize(ExclusionLoss())
        loss_dic['exclusion'] = exclusion_loss

    if getattr(opt, 'lambda_ssim', 0) > 0:
        ssim_loss = ContentLoss()
        ssim_loss.initialize(SSIMLoss())
        loss_dic['ssim'] = ssim_loss
    # -----------------------

    if opt.lambda_gan > 0:
        if opt.gan_type == 'sgan' or opt.gan_type == 'gan':
            disc_loss = DiscLoss()
        elif opt.gan_type == 'rsgan':
            disc_loss = DiscLossR()
        elif opt.gan_type == 'rasgan':
            disc_loss = DiscLossRa()
        else:
            raise ValueError("GAN [%s] not recognized." % opt.gan_type)

        disc_loss.initialize(opt, tensor)
        loss_dic['gan'] = disc_loss

    return loss_dic
