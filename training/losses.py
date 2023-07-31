

import torch
from torch import autograd
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
import configparser

from training.data_preprocessing import TRAIN_MEAN_COLOR, TRAIN_STD_COLOR, INVERSE_NORMALIZE

mse_loss = torch.nn.MSELoss()

# Load config value
config = configparser.ConfigParser()
config.read('configs/training.cfg')
W1 = config.getfloat('Training', 'w1_rec_loss')

# _____________________________________________________________________________

class VGG16PerceptualLoss(torch.nn.Module):
    '''
        Perceptual Loss using a pretrained VGG-16 as loss network.
        Specifically, a feature reconstruction loss is calculated 
        following the work of https://arxiv.org/abs/1603.08155.

        Code for the Loss was obtained and modified from
        https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
    '''

    def __init__(self):
        super().__init__()

        # Use mean and std of train dataset (manga dataset or fine-tune dataset)
        self.register_buffer("mean", TRAIN_MEAN_COLOR.view(1, 3, 1, 1))
        self.register_buffer("std", TRAIN_STD_COLOR.view(1, 3, 1, 1))
        # Weights of pretrained VGG-16
        vgg_pretrained_features = vgg16(weights=VGG16_Weights.DEFAULT).features.eval()
        self.block_relu1_2 = vgg_pretrained_features[:4].eval()
        self.block_relu2_2 = vgg_pretrained_features[4:9].eval()
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    # _________________________________________________________________________
    def forward(self, target, input, level_weight):
        # Standardize target and input
        target = (target-self.mean) / self.std
        input = (input-self.mean) / self.std
        # Results of layer relu2_2
        feat_target = self.block_relu2_2(self.block_relu1_2(target))
        feat_input = self.block_relu2_2(self.block_relu1_2(input))
        perceptual_loss = level_weight * mse_loss(feat_input, feat_target)
        return perceptual_loss

# _____________________________________________________________________________
class MultiScaleReconstructionLoss(torch.nn.Module):
    '''
        Reconstruction loss that ensures the functionality of style extraction 
        and style propagation.
        Depends on the perceptual loss and the pixel-wise mean square error.
    '''

    def __init__(self):
        super().__init__()

        self.vgg16_perceptual_loss = VGG16PerceptualLoss()
    
    # _________________________________________________________________________
    def pixel_wise_mse(self, target, input, level_weight):
        return level_weight * mse_loss(INVERSE_NORMALIZE(input),
                                       INVERSE_NORMALIZE(target))

    # _________________________________________________________________________
    def forward(self, batch, img_l1, img_l2, img_l3, img_l4):

        # Multi-scale perceptual loss (mean of all levels)
        perceptual_loss = 1/4 *(
            self.vgg16_perceptual_loss(batch.l1_color_sketch, img_l1, 1) +
            self.vgg16_perceptual_loss(batch.l2_color_sketch, img_l2, 2) +
            self.vgg16_perceptual_loss(batch.l3_color_sketch, img_l3, 3) +
            self.vgg16_perceptual_loss(batch.color_sketch, img_l4, 10)
        )
        # Pixel-wise mean square error (mean of all levels)
        pixel_loss = W1/4 *(
            self.pixel_wise_mse(batch.l1_color_sketch, img_l1, 1) +
            self.pixel_wise_mse(batch.l2_color_sketch, img_l2, 2) +
            self.pixel_wise_mse(batch.l3_color_sketch, img_l3, 3) +
            self.pixel_wise_mse(batch.color_sketch, img_l4, 10)
        )
        return perceptual_loss + pixel_loss

# _____________________________________________________________________________
def loss_hinge_dis(dis_value, type):
    '''
        # GAN Hinge Loss # (https://arxiv.org/abs/1802.05957)

        Code obtained and modified from
        https://github.com/pfnet-research/sngan_projection/blob/master/updater.py
    '''

    if type == 'real':
        #  min(0, -1 + dis_real) => F.relu(1. - dis_real)
        loss = torch.mean(F.relu(1. - dis_value, inplace=True))
    elif type == 'fake':
        #  min(0, -1 - dis_fake) => F.relu(1. + dis_fake)
        loss = torch.mean(F.relu(1. + dis_value, inplace=True))
    return loss

# _____________________________________________________________________________
def compute_grad_reg(d_out, x_in):
    '''
        # Gradient Penalty Regularization # (https://arxiv.org/abs/1801.04406)
        (for multi-scale adversarial loss)

        Code obtained and modified from
        https://github.com/LMescheder/GAN_stability/blob/master/gan_training/train.py
    '''

    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg
