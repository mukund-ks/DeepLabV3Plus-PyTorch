import torch
import torch.nn as nn
from typing import Any


class SEModule(nn.Module):
    def __init__(self, channels: int, ratio: int = 8) -> None:
        super(SEModule, self).__init__()

        # Average Pooling for Squeeze
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Excitation Operation
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // ratio, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Any) -> Any:
        # Squeeze & Excite Forward Pass
        b, c, _, _ = x.size()

        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y


class ASPPModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilations: list[int]) -> None:
        super(ASPPModule, self).__init__()

        # Atrous Convolution 1
        self.at_conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            dilation=dilations[0],
            padding="same",
            bias=False,
        )

        # Atrous Convolution 2
        self.at_conv2 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            dilation=dilations[1],
            padding="same",
            bias=False,
        )

        # Atrous Convolution 3
        self.at_conv3 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            dilation=dilations[2],
            padding="same",
            bias=False,
        )

        self.batch_norm = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=0.5)

        # Upsampling by Bilinear Interpolation
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=16)

        # Global Average Pooling
        self.avgpool = nn.AvgPool2d(kernel_size=(16, 16))

        # 1x1 Convolution
        self.conv1x1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, padding="same", bias=False
        )

        # Final 1x1 Convolution
        self.final_conv = nn.Conv2d(
            out_channels * 5, out_channels, kernel_size=1, padding="same", bias=False
        )

    def forward(self, x: Any) -> Any:
        # ASPP Forward Pass

        # 1x1 Convolution
        x1 = self.conv1x1(x)
        x1 = self.batch_norm(x1)
        x1 = self.dropout(x1)
        x1 = self.relu(x1)

        # Atrous Convolution - Rate: 6
        x2 = self.at_conv1(x)
        x2 = self.batch_norm(x2)
        x2 = self.relu(x2)

        # Atrous Convolution - Rate: 12
        x3 = self.at_conv2(x)
        x3 = self.batch_norm(x3)
        x3 = self.relu(x3)

        # Atrous Convolution - Rate: 18
        x4 = self.at_conv3(x)
        x4 = self.batch_norm(x4)
        x4 = self.relu(x4)

        # Global Average Pooling and 1x1 Convolution for global context
        avg_pool = self.avgpool(x)
        avg_pool = self.conv1x1(avg_pool)
        avg_pool = self.batch_norm(avg_pool)
        avg_pool = self.relu(avg_pool)
        avg_pool = self.upsample(avg_pool)

        # Concatenating Dilated Convolutions and Global Average Pooling
        combined_output = torch.cat((*[x1, x2, x3, x4], avg_pool), dim=1)

        # Final 1x1 Convolution for ASPP Output
        aspp_output = self.final_conv(combined_output)
        aspp_output = self.batch_norm(aspp_output)
        aspp_output = self.dropout(aspp_output)
        aspp_output = self.relu(aspp_output)

        return aspp_output


class DecoderModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(DecoderModule, self).__init__()

        # Squeeze and Excite Module
        self.squeeze_excite = SEModule(channels=304)

        self.squeeze_excite2 = SEModule(channels=out_channels)

        # Upsampling by Bilinear Interpolation
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=4)

        # 1x1 Convolution
        self.conv_low = nn.Conv2d(in_channels, 48, kernel_size=1, padding="same", bias=False)

        self.batch_norm = nn.BatchNorm2d(48)

        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=0.5)

        # 3x3 Convolution
        self.final_conv1 = nn.Conv2d(
            in_channels=304, out_channels=256, kernel_size=3, padding="same", bias=False
        )

        # 3x3 Convolution
        self.final_conv2 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding="same", bias=False
        )

    def forward(self, x_high: Any, x_low: Any) -> Any:
        # Decoder Forward Pass

        # Upsampling High-Level Features
        x_high = self.upsample(x_high)
        # x_high = self.dropout(x_high)

        # 1x1 Convolution on Low-Level Features
        x_low = self.conv_low(x_low)
        x_low = self.batch_norm(x_low)
        x_low = self.dropout(x_low)
        x_low = self.relu(x_low)

        # Concatenating High-Level and Low-Level Features
        x = torch.cat((x_high, x_low), dim=1)
        x = self.squeeze_excite(x)
        x = self.dropout(x)

        # 3x3 Convolution on Concatenated Feature Map
        x = self.final_conv1(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.squeeze_excite2(x)

        # 3x3 Convolution on Concatenated Feature Map
        x = self.final_conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.squeeze_excite2(x)

        return x