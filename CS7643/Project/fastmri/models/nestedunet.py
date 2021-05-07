import torch
from torch import nn

"""
Below model is inspired and adapted from:

https://arxiv.org/abs/1807.10165
https://github.com/4uiiurz1/pytorch-nested-unet

"""

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv_1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(middle_channels)
        self.conv_2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.relu(out)
        out = self.conv_2(out)
        out = self.bn_2(out)
        out = self.relu(out)

        return out

class NestedUnet(nn.Module):
    def __init__(self, input_channels, output_channels, is_deep_supervision=False, **kwargs):
        super().__init__()

        number_of_filters = [32, 64, 128, 256, 512]

        self.is_deep = is_deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.val_conv_0_0 = VGGBlock(input_channels, number_of_filters[0], number_of_filters[0])
        self.val_conv_1_0 = VGGBlock(number_of_filters[0], number_of_filters[1], number_of_filters[1])
        self.val_conv_2_0 = VGGBlock(number_of_filters[1], number_of_filters[2], number_of_filters[2])
        self.val_conv_3_0 = VGGBlock(number_of_filters[2], number_of_filters[3], number_of_filters[3])
        self.val_conv_4_0 = VGGBlock(number_of_filters[3], number_of_filters[4], number_of_filters[4])

        self.val_conv_0_1 = VGGBlock(number_of_filters[0] + number_of_filters[1], number_of_filters[0], number_of_filters[0])
        self.val_conv_1_1 = VGGBlock(number_of_filters[1] + number_of_filters[2], number_of_filters[1], number_of_filters[1])
        self.val_conv_2_1 = VGGBlock(number_of_filters[2] + number_of_filters[3], number_of_filters[2], number_of_filters[2])
        self.val_conv3_1 = VGGBlock(number_of_filters[3] + number_of_filters[4], number_of_filters[3], number_of_filters[3])

        self.val_conv_0_2 = VGGBlock(number_of_filters[0] * 2 + number_of_filters[1], number_of_filters[0], number_of_filters[0])
        self.val_conv_1_2 = VGGBlock(number_of_filters[1] * 2 + number_of_filters[2], number_of_filters[1], number_of_filters[1])
        self.val_conv_2_2 = VGGBlock(number_of_filters[2] * 2 + number_of_filters[3], number_of_filters[2], number_of_filters[2])

        self.val_conv_0_3 = VGGBlock(number_of_filters[0] * 3 + number_of_filters[1], number_of_filters[0], number_of_filters[0])
        self.val_conv_1_3 = VGGBlock(number_of_filters[1] * 3 + number_of_filters[2], number_of_filters[1], number_of_filters[1])

        self.val_conv_0_4 = VGGBlock(number_of_filters[0] * 4 + number_of_filters[1], number_of_filters[0], number_of_filters[0])

        if self.is_deep:
            self.final_layer_1 = nn.Conv2d(number_of_filters[0], output_channels, kernel_size=1)
            self.final_layer_2 = nn.Conv2d(number_of_filters[0], output_channels, kernel_size=1)
            self.final_layer_3 = nn.Conv2d(number_of_filters[0], output_channels, kernel_size=1)
            self.final_layer_4 = nn.Conv2d(number_of_filters[0], output_channels, kernel_size=1)
        else:
            self.final_layer = nn.Conv2d(number_of_filters[0], output_channels, kernel_size=1)

    def forward(self, input):
        x_0_0 = self.val_conv_0_0(input)
        x_1_0 = self.val_conv_1_0(self.pool(x_0_0))
        x_0_1 = self.val_conv_0_1(torch.cat([x_0_0, self.up(x_1_0)], 1))

        x_2_0 = self.val_conv_2_0(self.pool(x_1_0))
        x_1_1 = self.val_conv_1_1(torch.cat([x_1_0, self.up(x_2_0)], 1))
        x_0_2 = self.val_conv_0_2(torch.cat([x_0_0, x_0_1, self.up(x_1_1)], 1))

        x_3_0 = self.val_conv_3_0(self.pool(x_2_0))
        x_2_1 = self.val_conv_2_1(torch.cat([x_2_0, self.up(x_3_0)], 1))
        x_1_2 = self.val_conv_1_2(torch.cat([x_1_0, x_1_1, self.up(x_2_1)], 1))
        x_0_3 = self.val_conv_0_3(torch.cat([x_0_0, x_0_1, x_0_2, self.up(x_1_2)], 1))

        x_4_0 = self.val_conv_4_0(self.pool(x_3_0))
        x_3_1 = self.val_conv3_1(torch.cat([x_3_0, self.up(x_4_0)], 1))
        x_2_2 = self.val_conv_2_2(torch.cat([x_2_0, x_2_1, self.up(x_3_1)], 1))
        x_1_3 = self.val_conv_1_3(torch.cat([x_1_0, x_1_1, x_1_2, self.up(x_2_2)], 1))
        x_0_4 = self.val_conv_0_4(torch.cat([x_0_0, x_0_1, x_0_2, x_0_3, self.up(x_1_3)], 1))

        if self.is_deep:
            output = self.final_layer_1(x_0_1)
            output += self.final_layer_2(x_0_2)
            output += self.final_layer_3(x_0_3)
            output += self.final_layer_4(x_0_4)
            output /= 4
            return output

        else:
            output = self.final_layer(x_0_4)
            return output
