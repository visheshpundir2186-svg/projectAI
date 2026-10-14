import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights

class SwinTransformerClassifier(nn.Module):
    """
    Swin Transformer (Shifted Windows Transformer - Swin-T)
    Hierarchical Vision Transformer with linear computational complexity.
    Particularly effective for remote sensing / satellite spatial representation.
    """
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        try:
            weights = Swin_T_Weights.DEFAULT if pretrained else None
            self.model = swin_t(weights=weights)
        except Exception:
            self.model = swin_t(weights=None)

        # Replace classification head with custom head for groundwater evaluation
        in_features = self.model.head.in_features
        self.model.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)
