import torch
import torch.nn as nn
import torchvision.models as models
from modules import ASPPModule, DecoderModule, SEModule
from torchsummary import summary
from typing import Any


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes: int = 1) -> None:
        super(DeepLabV3Plus, self).__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        in_channels = 1024
        out_channels = 256

        # Dilation Rates
        dilations = [6, 12, 18, 24]
        
        # ASPP Module
        self.aspp = ASPPModule(in_channels, out_channels, dilations)

        # Decoder Module
        self.decoder = DecoderModule(out_channels, out_channels)

        # Upsampling with Bilinear Interpolation
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=4)

        # Dropout
        self.dropout = nn.Dropout(p=0.5)

        # Final 1x1 Convolution
        self.final_conv = nn.Conv2d(out_channels, num_classes, kernel_size=1)

        # Sigmoid Activation for Binary-Seg
        self.sigmoid = nn.Sigmoid()

        self.tanh = nn.Tanh()

    def forward(self, x: Any) -> Any:
        # DeepLabV3+ Forward Pass

        # Getting Low-Level Features
        x_low = self.backbone[:-3](x)

        # Getting Image Features from Backbone
        x = self.backbone[:-1](x)

        # ASPP forward pass - High-Level Features
        x = self.aspp(x)

        # Upsampling High-Level Features
        x = self.upsample(x)
        x = self.dropout(x)

        # Decoder forward pass - Concatenating Features
        x = self.decoder(x, x_low)

        # Upsampling Concatenated Features from Decoder
        x = self.upsample(x)

        # Final 1x1 Convolution for Binary-Segmentation
        x = self.final_conv(x)
        # x = self.sigmoid(x)
        x = self.tanh(x)
        normalized_x = (x + 1) * 0.5

        return normalized_x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepLabV3Plus(num_classes=1)  # For binary segmentation, num_classes=1
    model.to(device)

    # Random input tensor for testing
    batch_size = 2
    input_channels = 3
    height, width = 256, 256
    random_input = torch.randn(batch_size, input_channels, height, width).to(device)

    # Forward pass
    output = model(random_input)
    print("Output shape:", output.shape)
    summary(model, input_size=(3, 256, 256))
