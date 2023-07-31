import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary

# TODO: Add L2, Batch Normalization and Dropout

class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, dilations):
        super(ASPPModule, self).__init__()

        # Branches with dilated convolutions
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=d, dilation=d)
                for d in dilations
            ]
        )

        # 1x1 Convolution for the global context
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Final 1x1 Convolution to combine the branches and global context
        self.final_conv = nn.Conv2d(
            out_channels * (len(dilations) + 1), out_channels, kernel_size=1
        )

    def forward(self, x):
        branches_outputs = [branch(x) for branch in self.branches]

        # Global Average Pooling and 1x1 Convolution for global context
        global_avg_pool_output = F.adaptive_avg_pool2d(x, output_size=1)
        global_avg_pool_output = self.conv1x1(global_avg_pool_output)
        global_avg_pool_output = F.interpolate(
            global_avg_pool_output, size=x.size()[2:], mode="bilinear", align_corners=True
        )

        # Concatenating Dilated Convolutions and Global Average Pooling
        combined_output = torch.cat([*branches_outputs, global_avg_pool_output], dim=1)

        # Final Convolution for ASPP Output
        aspp_output = self.final_conv(combined_output)

        return aspp_output


class DecoderModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderModule, self).__init__()

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=4)

        self.conv_low = nn.Conv2d(in_channels, 48, kernel_size=1)

        self.final_conv1 = nn.Conv2d(in_channels=304, out_channels=256, kernel_size=3, padding=1)
        
        self.final_conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x_high, x_low):
        x_high = self.upsample(x_high)

        x_low = self.conv_low(x_low)

        x = torch.cat((x_high, x_low), dim=1)

        x = self.final_conv1(x)
        
        x = self.final_conv2(x)

        return x


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Plus, self).__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        in_channels = 1024
        out_channels = 256

        dilations = [1, 6, 12, 18]

        self.aspp = ASPPModule(in_channels, out_channels, dilations)

        # Decoder Module
        self.decoder = DecoderModule(out_channels, out_channels)
        
        # Upsampling with Bilinear Interpolation
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=4)

        # Final convolution layer for binary segmentation
        self.final_conv = nn.Conv2d(out_channels, num_classes, kernel_size=1)

    def forward(self, x):
        low_level_features = self.backbone[:-3](x)

        x = self.backbone[:-1](x)

        # ASPP forward pass
        x = self.aspp(x)

        # Decoder forward pass
        x = self.decoder(x, low_level_features)

        x = self.upsample(x)
        
        x = self.final_conv(x)

        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepLabV3Plus(num_classes=1)  # For binary segmentation, num_classes=1
    model.to(device)

    # Random input tensor for testing
    batch_size = 1
    input_channels = 3
    height, width = 256, 256
    random_input = torch.randn(batch_size, input_channels, height, width).to(device)

    # Forward pass
    output = model(random_input)
    print("Output shape:", output.shape)
    # summary(model, input_size=(3, 256, 256))
