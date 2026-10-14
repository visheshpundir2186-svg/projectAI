import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Attention Gate (Oktay et al.) to focus on salient regional features
    (e.g., dry soil, vegetation patches, surface water bodies).
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNetClassifier(nn.Module):
    """
    Attention U-Net modified with a Classification Head for Groundwater
    Depletion & Stress Risk Evaluation.
    """
    def __init__(self, in_channels=3, num_classes=3, base_filters=32):
        super().__init__()
        # Encoder stages
        self.c1 = ConvBlock(in_channels, base_filters)          # 32
        self.pool1 = nn.MaxPool2d(2)

        self.c2 = ConvBlock(base_filters, base_filters * 2)      # 64
        self.pool2 = nn.MaxPool2d(2)

        self.c3 = ConvBlock(base_filters * 2, base_filters * 4)  # 128
        self.pool3 = nn.MaxPool2d(2)

        # Deepest Bottleneck (High-level semantic features)
        self.bottleneck = ConvBlock(base_filters * 4, base_filters * 8) # 256

        # Attention gate highlighting salient patterns in encoder feature x3
        self.att_gate = AttentionGate(F_g=base_filters * 8, F_l=base_filters * 4, F_int=base_filters * 4)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Classification Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_filters * 8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(self.pool1(x1))
        x3 = self.c3(self.pool2(x2))
        
        # Bottleneck
        b = self.bottleneck(self.pool3(x3))

        # Global average pool over the deep bottleneck features
        pooled = self.gap(b)
        logits = self.classifier(pooled)
        return logits
