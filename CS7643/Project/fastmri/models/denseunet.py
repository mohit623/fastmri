import torch
from torch import nn

"""
Below model is inspired and adapted from:
https://github.com/veritas9872/fastMRI-kspace/blob/master/models/dense_unet.py
"""

class ChannelAttention(nn.Module):
    def __init__(self, num_chans, reduction=16):
        super().__init__()
        self.global_average_pooling = nn.AdaptiveAvgPool2d(1)
        self.layer = nn.Sequential(
            nn.Linear(in_features=num_chans, out_features=num_chans // reduction),
            nn.ReLU(),
            nn.Linear(in_features=num_chans // reduction, out_features=num_chans)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, tensor):
        batch, channels, _, _ = tensor.shape
        global_average_pooling = self.global_average_pooling(tensor).view(batch, channels)
        features = self.layer(global_average_pooling)
        att = self.sigmoid(features).view(batch, channels, 1, 1)
        return tensor * att


class AdapterConv(nn.Module):
    def __init__(self, in_chans, out_chans, stride=1, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(num_chans=out_chans, reduction=reduction)
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=stride, padding=1),
            nn.ReLU()
        )

    def forward(self, tensor):
        return self.layers(self.channel_attention(tensor))


class ResBlock(nn.Module):
    def __init__(self, num_chans, res_scale=1., reduction=16):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=num_chans, out_channels=num_chans, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=num_chans, out_channels=num_chans, kernel_size=3, padding=1),
        )
        self.channel_attention = ChannelAttention(num_chans=num_chans, reduction=reduction)
        self.res_scale = res_scale

    def forward(self, tensor):  
        return tensor + self.res_scale * self.channel_attention(self.layer(tensor))


class ShuffleUp(nn.Module):
    def __init__(self, in_chans, out_chans, scale_factor=2, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(num_chans=in_chans, reduction=reduction)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=out_chans * scale_factor ** 2, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor=scale_factor),
            nn.ReLU()
        )

    def forward(self, tensor):
        return self.layer(self.channel_attention(tensor))


class ConcatConv(nn.Module):
    

    def __init__(self, in_chans, growth_rate, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(num_chans=in_chans, reduction=reduction)
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=growth_rate, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, tensor):
        return torch.cat([tensor, self.conv_layer(self.channel_attention(tensor))], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_chans, out_chans, growth_rate, num_layers, reduction=16):
        super().__init__()
        layers = list()
        for idx in range(num_layers):
            channels = in_chans + idx * growth_rate
            conv = ConcatConv(in_chans=channels, growth_rate=growth_rate)
            layers.append(conv)

        self.layers = nn.Sequential(*layers)
        channels = in_chans + num_layers * growth_rate
        self.channel_attention = ChannelAttention(num_chans=channels, reduction=reduction)
        self.feature_fusion = nn.Conv2d(in_channels=channels, out_channels=out_chans, kernel_size=1)

    def forward(self, tensor):
        return self.feature_fusion(self.channel_attention(self.layers(tensor)))


class DenseUNet(nn.Module):
    def __init__(self, in_chans, out_chans, chans, num_pool_layers,
                 num_depth_blocks, growth_rate, num_layers, res_scale=0.1, reduction=16):

        super().__init__()
        self.num_pool_layers = num_pool_layers

        if isinstance(num_layers, int):
            num_layers = [num_layers] * num_pool_layers
        elif isinstance(num_layers, (list, tuple)):
            assert len(num_layers) == num_pool_layers
        else:
            raise RuntimeError('Invalid type for num_layers.')

        # Remove ReLU from the first layer
        conv_block = nn.Conv2d(in_channels=in_chans, out_channels=chans, kernel_size=3, padding=1)
        dense_block = DenseBlock(in_chans=chans, out_chans=chans, growth_rate=growth_rate,
                                 num_layers=num_layers[0], reduction=reduction)
        self.down_layers = nn.ModuleList([conv_block])
        self.down_dense_blocks_layer = nn.ModuleList([dense_block])

        all_channels = chans
        for idx in range(num_pool_layers - 1):
            conv_block = AdapterConv(in_chans=all_channels, out_chans=all_channels, stride=2, reduction=reduction)
            dense_block = DenseBlock(in_chans=all_channels, out_chans=all_channels * 2, growth_rate=growth_rate,
                                     num_layers=num_layers[idx + 1], reduction=reduction)
            self.down_layers += [conv_block]
            self.down_dense_blocks_layer += [dense_block]
            all_channels *= 2

        self.mid_conv_block = AdapterConv(in_chans=all_channels, out_chans=all_channels,
                                          stride=2, reduction=reduction)
        mid_res_blocks = list()
        for _ in range(num_depth_blocks):
            mid_res_blocks.append(ResBlock(num_chans=all_channels, res_scale=res_scale,
                                           reduction=reduction))
        self.mid_res_blocks = nn.Sequential(*mid_res_blocks)

        self.up_scale_layers_list = nn.ModuleList()
        self.up_dense_blocks_list = nn.ModuleList()

        for idx in range(num_pool_layers - 1):
            shuffle_up = ShuffleUp(in_chans=all_channels, out_chans=all_channels, scale_factor=2)
            dense_block = DenseBlock(in_chans=all_channels * 2, out_chans=all_channels // 2, growth_rate=growth_rate,
                                     num_layers=num_layers[-idx - 1], reduction=reduction)
            self.up_scale_layers_list += [shuffle_up]
            self.up_dense_blocks_list += [dense_block]
            all_channels //= 2
        else:  # Last block of up-sampling.
            shuffle_up = ShuffleUp(in_chans=all_channels, out_chans=all_channels, scale_factor=2)
            dense_block = DenseBlock(in_chans=all_channels * 2, out_chans=all_channels, growth_rate=growth_rate,
                                     num_layers=num_layers[-num_pool_layers], reduction=reduction)
            self.up_scale_layers_list += [shuffle_up]
            self.up_dense_blocks_list += [dense_block]
            assert chans == all_channels, 'Channel indexing error!'

        self.final_layers = nn.Conv2d(in_channels=all_channels, out_channels=out_chans, kernel_size=1)

        assert len(self.down_layers) == len(self.down_dense_blocks_layer) == len(self.up_scale_layers_list) \
               == len(self.up_dense_blocks_list) == self.num_pool_layers, 'Layer number error!'

    def forward(self, tensor):
        stack = list()
        output = tensor

        # Down-Sampling
        for idx in range(self.num_pool_layers):
            output = self.down_layers[idx](output)
            output = self.down_dense_blocks_layer[idx](output)
            stack.append(output)

        # Middle blocks
        output = self.mid_conv_block(output)
        output = output + self.mid_res_blocks(output)

        # Up-Sampling.
        for idx in range(self.num_pool_layers):
            output = self.up_scale_layers_list[idx](output)
            output = torch.cat([output, stack.pop()], dim=1)
            output = self.up_dense_blocks_list[idx](output)

        return self.final_layers(output)
