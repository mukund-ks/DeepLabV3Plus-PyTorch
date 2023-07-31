import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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

        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x):
        # Upsample the feature maps to increase the spatial resolution by a factor of 4
        x = self.upsample(x)

        return x


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Plus, self).__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # ASPP Module
        in_channels = 2048  
        out_channels = 256
        dilations = [1, 6, 12, 18]

        self.aspp = ASPPModule(in_channels, out_channels, dilations)

        # Decoder Module
        self.decoder = DecoderModule(out_channels, out_channels // 4)

        # Final convolution layer for binary segmentation
        self.final_conv = nn.Conv2d(out_channels // 4, num_classes, kernel_size=1)

    def forward(self, x):
        # Backbone (ResNet50) forward pass
        x = self.backbone(x)

        # ASPP forward pass
        x = self.aspp(x)

        # Decoder forward pass
        x = self.decoder(x)

        x = self.final_conv(x)

        # Resizing the output to match the input image size (256x256)
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=True)

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
