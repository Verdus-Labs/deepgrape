import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_conv(x)


class Bridge(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bridge, self).__init__()
        self.bridge = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet(nn.Module):
    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None, upsampling_method="conv_transpose"):
        super(UpBlockForUNetWithResNet, self).__init__()
        
        if up_conv_in_channels is None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels is None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
            self.up_conv = None
        elif upsampling_method == "bilinear":
            self.upsample = None
            self.up_conv = nn.Conv2d(up_conv_in_channels, up_conv_out_channels, kernel_size=1)
        else:
            raise ValueError("Unsupported upsampling method")

        self.conv_block = ConvBlock(in_channels, out_channels)
        self.upsampling_method = upsampling_method

    def forward(self, up_x, down_x):
        if self.upsampling_method == "conv_transpose":
            if self.upsample is not None:
                up_x = self.upsample(up_x)
        elif self.upsampling_method == "bilinear":
            up_x = F.interpolate(up_x, size=(down_x.size(2), down_x.size(3)), mode="bilinear", align_corners=True)
            if self.up_conv is not None:
                up_x = self.up_conv(up_x)

        x = torch.cat([down_x, up_x], dim=1)
        return self.conv_block(x)


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class BaseUNet(nn.Module):
    def __init__(self):
        super(BaseUNet, self).__init__()
        
    def _create_decoder_blocks(self, up_channels_list, skip_channels_list):
        up_blocks = []
        for i, (up_ch, skip_ch) in enumerate(zip(up_channels_list, skip_channels_list)):
            if i == 0:
                up_blocks.append(UpBlockForUNetWithResNet(up_ch + skip_ch, up_ch, up_ch, up_ch))
            else:
                up_blocks.append(UpBlockForUNetWithResNet(up_ch + skip_ch, up_ch, up_channels_list[i-1], up_ch))
        return nn.ModuleList(up_blocks)
    
    def _apply_decoder(self, x, skip_connections, up_blocks):
        for i, (skip_x, up_block) in enumerate(zip(reversed(skip_connections), up_blocks)):
            x = up_block(x, skip_x)
        return x
