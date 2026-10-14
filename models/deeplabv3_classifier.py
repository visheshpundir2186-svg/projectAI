import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

class DeepLabV3PlusClassifier(nn.Module):
    """
    DeepLabV3+ architecture with Atrous Spatial Pyramid Pooling (ASPP)
    modified with a Classification Head.
    Captures multi-scale contextual features (water bodies, dry soil, crops)
    across diverse dilation rates.
    """
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        try:
            weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
            base_model = deeplabv3_resnet50(weights=weights)
        except Exception:
            # Fallback if offline
            base_model = deeplabv3_resnet50(weights=None)
        
        # ResNet-50 feature extractor
        self.backbone = base_model.backbone
        
        # ASPP module (Atrous Spatial Pyramid Pooling)
        # base_model.classifier[0] is ASPP module outputting 256 channels
        self.aspp = base_model.classifier[0]

        # Classification Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)['out']  # (B, 2048, H/8, W/8)
        aspp_out = self.aspp(features)       # (B, 256, H/8, W/8)
        pooled = self.gap(aspp_out)          # (B, 256, 1, 1)
        logits = self.classifier(pooled)     # (B, num_classes)
        return logits
