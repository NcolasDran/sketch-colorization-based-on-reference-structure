
'''
    # Colorization network with multi-scale outputs #

    Based on the U-Net structure:
        > Downscaling Sketch Encoder (5 downsampling blocks)
        > Upscaling Style-Content Fusion Decoder (5 upsampling blocks)

    Code obtained and modified from
    https://github.com/milesial/Pytorch-UNet

    Other references:
    https://github.com/mateuszbuda/brain-segmentation-pytorch
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py (class UnetGenerator)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
import configparser

# Load config value
config = configparser.ConfigParser()
config.read('configs/training.cfg')
FINE_TUNE = config.getboolean('Training', 'fine_tune')


# Small value to avoid zero division in AdaIN
EPS = 1e-5

# ____________________________________________________________________________
class EncoderBlock(nn.Module):
    '''
        Encoder architecture:
        Conv2d -> InstanceNorm2d -> ReLU
    '''

    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2,
                 norm=True):
        super().__init__()

        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                           stride=stride, padding=1, bias=False))
        if norm:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        layers.append(nn.ReLU(inplace=True))

        self.single_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.single_conv(x)


# ____________________________________________________________________________
def get_mean_std_style(color_style):
    '''
        Calculate values needed for AdaIN
        (mean and standard deviation of the style)
    '''

    if len(color_style.shape) == 4:
        mu_style = torch.mean(color_style, dim=[2, 3])
        mu_style = mu_style.unsqueeze(-1).unsqueeze(-1)
        sigma_style = torch.std(color_style, dim=[2, 3])
        sigma_style = sigma_style.unsqueeze(-1).unsqueeze(-1) + EPS
    else:
        mu_style = torch.mean(color_style, dim=1)
        mu_style = mu_style.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        sigma_style = torch.std(color_style, dim=1)
        sigma_style = sigma_style.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + EPS

    return mu_style, sigma_style

# ____________________________________________________________________________
def adaptive_instance_normalization(content, mu_style, sigma_style):
                                    
    '''
        Adaptive Instance Normalization (AdaIN)

        Code obtained and modified from:
        https://github.com/media-comp/2022-AdaIN-pytorch/
    '''

    mu_content = torch.mean(content, dim=[2, 3])
    mu_content = mu_content.unsqueeze(-1).unsqueeze(-1)

    sigma_content = torch.std(content, dim=[2, 3])
    sigma_content = sigma_content.unsqueeze(-1).unsqueeze(-1) + EPS

    return ((content - mu_content) / sigma_content) * sigma_style  + mu_style


# ____________________________________________________________________________
class DecoderBlock(nn.Module):
    '''
        Decoder architecture:
        [Conv2d -> AdaIN -> ReLU -> Conv2d -> AdaIN] => [ConvTranspose2d -> ReLU]
    '''

    # ________________________________________________________________________
    def __init__(self, in_channels, out_channels, extra_output=True):
        super().__init__()

        self.extra_output = extra_output

        self.conv_1 = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                stride=1, padding=1, bias=False)

        # Deconvolution | bias=True (no normalization)
        self.tr_conv = nn.ConvTranspose2d(in_channels*2, out_channels,
                                          kernel_size=4, stride=2, padding=1)
        
        # Conv2d to output extra image of lower resolution
        if self.extra_output:
            self.conv_out = nn.Conv2d(out_channels, 3, kernel_size=1, bias=False) 
            for param in self.conv_out.parameters():
                param.requires_grad = False
            self.tanh = nn.Tanh()
        
        self.fine_tune = FINE_TUNE

    # ________________________________________________________________________
    def forward(self, x, enc_x, multi_scale, mu_style, sigma_style):
        x = self.conv_1(x)
        x = adaptive_instance_normalization(x, mu_style, sigma_style) # AdaIN
        x = self.relu(x)
        x = self.conv_2(x)
        x = adaptive_instance_normalization(x, mu_style, sigma_style) # AdaIN

        # Pad x if necessary
        diff_height = enc_x.size()[2] - x.size()[2]
        diff_width = enc_x.size()[3] - x.size()[3]
        if diff_height > 0 or diff_width > 0:
            x = F.pad(x, [diff_width//2, diff_width-diff_width//2,
                      diff_height//2, diff_height-diff_height//2])

        # Concatenated Skip Connection with corresponding encoder output
        x = torch.cat([enc_x, x], dim=1)
        
        x = self.tr_conv(x)
        x = self.relu(x)

        if self.extra_output and multi_scale:
            return x, self.tanh(self.conv_out(x))
        else:
            return x, None

# ____________________________________________________________________________
class OutputDecoder(nn.Module):
    '''
        Final decoder output that has the same resolution as the input sketch.
        [Conv2d -> Tanh]
    '''
    # ________________________________________________________________________
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # bias=True (no normalization)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    # ________________________________________________________________________
    def forward(self, x, enc_x):
        # Pad x if necessary
        diff_height = enc_x.size()[2] - x.size()[2]
        diff_width = enc_x.size()[3] - x.size()[3]
        if diff_height > 0 or diff_width > 0:
            x = F.pad(x, [diff_width//2, diff_width-diff_width//2,
                      diff_height//2, diff_height-diff_height//2])

        # Concatenate with corresponding encoder output
        x = torch.cat([enc_x, x], dim=1)
        x = self.conv(x)
        x = self.tanh(x)
        return x


# ____________________________________________________________________________
class ColorizationNet(nn.Module):
    '''
        Colorization network, based on the U-Net structure
    '''
    # ________________________________________________________________________
    def __init__(self, input_channels=3, output_channels=3, middle_channels=64):
        super(ColorizationNet, self).__init__()

        # Maintain image resolution in first Encoder Block
        self.enc_1 = EncoderBlock(input_channels, middle_channels, 
                                  kernel_size=3, stride=1)                   # (3, 64)
        self.enc_2 = EncoderBlock(middle_channels, middle_channels*2)        # (64, 128)
        self.enc_3 = EncoderBlock(middle_channels*2, middle_channels*4)      # (128, 256)
        self.enc_4 = EncoderBlock(middle_channels*4, middle_channels*8)      # (256, 512)
        self.enc_5 = EncoderBlock(middle_channels*8, middle_channels*16)     # (512, 1024)

        self.dec_1 = DecoderBlock(middle_channels*16, middle_channels*8)     # (1024, 512)
        self.dec_2 = DecoderBlock(middle_channels*8, middle_channels*4)      # (512, 256)
        self.dec_3 = DecoderBlock(middle_channels*4, middle_channels*2)      # (256, 128)
        self.dec_4 = DecoderBlock(middle_channels*2, middle_channels, False) # (128, 64)
        # In OutputDecoder, 'in_channels' is 2*middle_channels because of the concatenation
        self.out_dec = OutputDecoder(2*middle_channels, output_channels)     # (128, 3)

        self.fine_tune = FINE_TUNE

    # ________________________________________________________________________
    def forward(self, sketch, color_style, multi_scale=True):
        #____________________________________________________
        # ENCODER # 
        enc_x1 = self.enc_1(sketch)
        enc_x2 = self.enc_2(enc_x1)
        enc_x3 = self.enc_3(enc_x2)
        enc_x4 = self.enc_4(enc_x3)
        enc_x5 = self.enc_5(enc_x4)

        # Get the values needed for AdaIN
        if self.fine_tune:
            style_1, style_2, style_3, style_4 = color_style
            mu_style1, sigma_style1 = get_mean_std_style(style_1)
            mu_style2, sigma_style2 = get_mean_std_style(style_2)
            mu_style3, sigma_style3 = get_mean_std_style(style_3)
            mu_style4, sigma_style4 = get_mean_std_style(style_4)
        else:
            mu_style, sigma_style = get_mean_std_style(color_style)
            mu_style1 = mu_style2 = mu_style3 = mu_style4 = mu_style
            sigma_style1 = sigma_style2 = sigma_style3 = sigma_style4 = sigma_style

        #____________________________________________________
        # DECODER #
        dec_x1, dec_img1 = self.dec_1(enc_x5, enc_x5, multi_scale, mu_style1, sigma_style1)
        dec_x2, dec_img2 = self.dec_2(dec_x1, enc_x4, multi_scale, mu_style2, sigma_style2)
        dec_x3, dec_img3 = self.dec_3(dec_x2, enc_x3, multi_scale, mu_style3, sigma_style3)
        dec_x4, _ = self.dec_4(dec_x3, enc_x2, multi_scale, mu_style4, sigma_style4)
        out_img = self.out_dec(dec_x4, enc_x1)

        if multi_scale:
            # 32x32, 64x64, 128x128, 256x256 (train resolutions)
            return dec_img1, dec_img2, dec_img3, out_img
        else:
            # Only output last img
            return out_img
