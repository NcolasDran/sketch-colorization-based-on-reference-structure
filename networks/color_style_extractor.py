
'''
        # Color style extractor # 
        
        4-block downscaling residual network
'''

import torch.nn as nn
import torch
import configparser

# Load config value
config = configparser.ConfigParser()
config.read('configs/training.cfg')
FINE_TUNE = config.getboolean('Training', 'fine_tune')

# ____________________________________________________________________________
# 
class InstanceNorm2dClass(nn.Module):
    '''
        Custom InstanceNorm2d with affine = True, used in ResNetBlock
    '''
    def __init__(self, num_features):
        super(InstanceNorm2dClass, self).__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=True)

    def forward(self, x):
        return self.norm(x)

# ____________________________________________________________________________
class ResNetBlock(nn.Module):
    '''
        # ResNet Block #

        Code obtained and modified from:
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    '''

    # ________________________________________________________________________
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,
                 relu=nn.ReLU, padding_type='reflect', use_bias=False, 
                 use_dropout=False, norm_layer=InstanceNorm2dClass):

        super().__init__()
        
        self.conv_block = self.build_conv_block(in_channels, out_channels, 
                                                kernel_size, padding_type, 
                                                use_bias, stride, use_dropout, 
                                                norm_layer, relu)

        downsample_block = [nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                      stride=stride, bias=use_bias)]
        if norm_layer is not None:
                downsample_block.append(norm_layer(out_channels))
        self.downsample = nn.Sequential(*downsample_block)
        self.relu = relu(inplace=True)

    # ________________________________________________________________________
    def build_conv_block(self, in_channels, out_channels, kernel_size, 
                         padding_type, use_bias, stride, use_dropout, 
                         norm_layer, relu):
        p = 0
        if padding_type == 'reflect':
            pad_type = nn.ReflectionPad2d
        elif padding_type == 'replicate':
            pad_type = nn.ReplicationPad2d
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block = []
        conv_block.append(pad_type(1))
        conv_block.append(nn.Conv2d(in_channels, out_channels, 
                                 kernel_size=kernel_size,  stride=stride, 
                                 padding=p, bias=use_bias))
        if norm_layer is not None:
            conv_block.append(norm_layer(out_channels))
        conv_block.append(relu(inplace=True))
        if use_dropout:
            conv_block.append(nn.Dropout(0.5))
        conv_block.append(pad_type(1))
        conv_block.append(nn.Conv2d(out_channels, out_channels, 
                                    kernel_size=kernel_size, padding=p, 
                                    bias=use_bias))
        if norm_layer is not None:
            conv_block.append(norm_layer(out_channels))
        return nn.Sequential(*conv_block)

    # ________________________________________________________________________
    def forward(self, x):
        identity = self.downsample(x)
        out = identity + self.conv_block(x)  # add skip connections
        out = self.relu(out)
        return out


# ____________________________________________________________________________
class ColorStyleExtractor(nn.Module):

    '''
    Implementation inspired by the ResNet model from Pytorch:
    https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    '''

    def __init__(self, in_channels=3, middle_channels=64, out_features=256):
        super().__init__()
        
        # No InstanceNorm to maintain color of the input
        self.block1 = ResNetBlock(in_channels, middle_channels, norm_layer=None)          # (3, 64)
        self.block2 = ResNetBlock(middle_channels, middle_channels*2, norm_layer=None)    # (64, 128)
        self.block3 = ResNetBlock(middle_channels*2, middle_channels*4, norm_layer=None)  # (128, 256)
        self.block4 = ResNetBlock(middle_channels*4, middle_channels*8, norm_layer=None)  # (256, 512)
    
        self.glob_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(middle_channels*8, out_features)

        self.fine_tune = FINE_TUNE
    
    def add_block5(self, middle_channels=64):
        # Add extra block when fine-tuning
        self.block5 = ResNetBlock(middle_channels*8, middle_channels*16, norm_layer=None)  # (512, 1024)

    def forward(self, color_sketch):
        x_b1 = self.block1(color_sketch)
        x_b2 = self.block2(x_b1)
        x_b3 = self.block3(x_b2)
        x_b4 = self.block4(x_b3)

        # Different forward when fine-tuning
        if self.fine_tune:
            x_b5 = self.block5(x_b4)
            return [x_b5, x_b4, x_b3, x_b2]
        else:
            x_gap = self.glob_avg_pool(x_b4)
            color_style_code = torch.flatten(x_gap, 1)
            color_style_code = self.fc(color_style_code)
            return color_style_code
