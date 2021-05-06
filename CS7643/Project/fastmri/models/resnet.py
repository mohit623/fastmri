"""
Below model is inspired and adapted from below blog post:
https://medium.com/@khandkers/improving-undersampled-mri-with-deep-learning-3f0839e6ba4c
https://github.com/SamiKhandker/ImprovingMRIQuality

"""

import torch
import sys


class Conv2dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=torch.nn.ReflectionPad2d):
        super().__init__()
        k_1_padding = kernel_size // 2
        k_2_padding = k_1_padding - 1 if kernel_size % 2 == 0 else k_1_padding
        self.net = torch.nn.Sequential(
            padding_layer((k_1_padding,k_2_padding,k_1_padding,k_2_padding)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )
    def forward(self, x):
        return self.net(x)


class ResNet5Block(torch.nn.Module):
    def __init__(self, num_filters=32, filter_size=3, T=4, num_filters_start=2, num_filters_end=2, batch_norm=False):
        super(ResNet5Block, self).__init__()
        num_filters_start = num_filters_end = 320
        if batch_norm:
            self.model = torch.nn.Sequential(
                Conv2dSame(num_filters_start,num_filters,filter_size),
                torch.nn.BatchNorm2d(num_filters),
                torch.nn.ReLU(),
                Conv2dSame(num_filters,num_filters,filter_size),
                torch.nn.BatchNorm2d(num_filters),
                torch.nn.ReLU(),
                Conv2dSame(num_filters,num_filters,filter_size),
                torch.nn.BatchNorm2d(num_filters),
                torch.nn.ReLU(),
                Conv2dSame(num_filters,num_filters,filter_size),
                torch.nn.BatchNorm2d(num_filters),
                torch.nn.ReLU(),
                Conv2dSame(num_filters,num_filters,filter_size),
                torch.nn.BatchNorm2d(num_filters),
                torch.nn.ReLU(),
                Conv2dSame(num_filters,num_filters_end,filter_size)
            )
        else:
            self.model = torch.nn.Sequential(
                Conv2dSame(num_filters_start,num_filters,filter_size),
                torch.nn.ReLU(),
                Conv2dSame(num_filters,num_filters,filter_size),
                torch.nn.ReLU(),
                Conv2dSame(num_filters,num_filters,filter_size),
                torch.nn.ReLU(),
                Conv2dSame(num_filters,num_filters,filter_size),
                torch.nn.ReLU(),
                Conv2dSame(num_filters,num_filters,filter_size),
                torch.nn.ReLU(),
                Conv2dSame(num_filters,num_filters_end,filter_size)
            )
        self.T = T
        
    def forward(self,x,device='cpu'):
        return x + self.step(x, device=device)
    
    def step(self, x_im, device='cpu'):
        x_im = x_im.permute(0, 3, 1, 2)
        y = self.model(x_im)
        return y.permute(0, 2, 3, 1)


class ResNetBlock(torch.nn.Module):
    def __init__(self, in_channels=2, latent_channels=64, out_channels=64, kernel_size=3, bias=False, batch_norm=True, final_relu=True, dropout=0):
        super(ResNetBlock, self).__init__()

        self.val_batch_norm = batch_norm
        self.val_final_relu = final_relu

        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.out_channels = out_channels
        self.val_kernel_size = kernel_size
        self.bias_value = bias
        if dropout > 0:
            self.val_drop_out = torch.nn.Dropout(dropout)
        else:
            self.val_drop_out = None

        if self.in_channels == self.out_channels:
            self.val_conv_0 = None
        else:
            self.val_conv_0 = self._conv_zero(self.in_channels, self.out_channels)
        self.val_conv_1 = self._conv(self.in_channels, self.latent_channels)
        self.val_conv_2 = self._conv(self.latent_channels, self.out_channels)

        if self.val_batch_norm:
            self.bnorm_1 = self._bn(self.in_channels)
            self.bnorm_2 = self._bn(self.latent_channels)

        self.relu = self._relu()

    def forward(self, x):
        if self.val_conv_0:
            residual_val = self.val_conv_0(x)
        else:
            residual_val = x

        out = x

        if self.val_batch_norm:
            out = self.bnorm_1(out)

        out = self.relu(out)

        out = self.val_conv_1(out)

        if self.val_drop_out is not None:
            out = self.val_drop_out(out)

        if self.val_batch_norm:
            out = self.bnorm_2(out)

        if self.val_final_relu:
            out = self.relu(out)

        out = self.val_conv_2(out)

        if self.val_drop_out is not None:
            out = self.val_drop_out(out)

        out += residual_val

        return out

    def _conv(self, in_channels, out_channels):
        return Conv2dSame(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=self.val_kernel_size,
                          bias=self.bias_value)

    def _conv_zero(self, in_channels, out_channels):
        return Conv2dSame(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=1,
                          bias=self.bias_value)

    def _bn(self, channels):
        return torch.nn.BatchNorm2d(channels)

    def _relu(self):
        #return torch.nn.ReLU(inplace=True)
        return torch.nn.ReLU()
