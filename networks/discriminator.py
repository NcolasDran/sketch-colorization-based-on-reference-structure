
'''
    # Multi-scale discriminator #

    Patch-based discriminator, with 3 downscaling residual blocks
    
    Code obtained and modified from class NLayerDiscriminator, in:
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
'''

import torch.nn as nn

from .color_style_extractor import ResNetBlock

# ____________________________________________________________________________
class LeakyReLUClass(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.func = nn.LeakyReLU(negative_slope=0.2, inplace=inplace)
        
    def forward(self, x):
        return self.func(x)


# ____________________________________________________________________________
class MultiScalePatchDiscriminator(nn.Module):

    def __init__(self, in_channels=3, middle_channels=64):
        '''
            # Custom PatchGAN Discriminator #

            Receptive field of the discriminator of 59x59 patches:

                Layer 7: |kernel=3, stride=1| -> 3 receptive field
                         |-- ResNet Block --|
                Layer 6: |kernel=3, stride=1| -> 5 receptive field
                Layer 5: |kernel=3, stride=2| -> 11 receptive field
                         |-- ResNet Block --|
                Layer 4: |kernel=3, stride=1| -> 13 receptive field
                Layer 3: |kernel=3, stride=2| -> 27 receptive field
                         |-- ResNet Block --|
                Layer 2: |kernel=3, stride=1| -> 29 receptive field
                Layer 1: |kernel=3, stride=2| -> 59 receptive field

            For more info:
            https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/39
        '''
        super().__init__()
        
        # No use of InstanceNorm in the first block because:
        #   'If you use an instancenorm in the first layer, the color of the 
        #    input image will be normalized and get ignored' 
        #   (https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/782)
        self.block1 = ResNetBlock(in_channels, middle_channels,  
                                  relu=LeakyReLUClass, norm_layer=None) # (3, 64)
        self.block2 = ResNetBlock(middle_channels, middle_channels*2,
                                  relu=LeakyReLUClass)                  # (64, 128)
        self.block3 = ResNetBlock(middle_channels*2, middle_channels*4,
                                  relu=LeakyReLUClass)                  # (128, 256)

        self.out = nn.Conv2d(middle_channels*4, 1, kernel_size=3,
                             stride=1, padding=1)                       # (256, 1)


    def forward(self, colored_sketch):
        x = self.block1(colored_sketch)
        x = self.block2(x)
        x = self.block3(x)
        x = self.out(x)
        return x
