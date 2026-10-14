import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50Classifier(nn.Module):
    """
    ResNet-50 (Residual Network) Baseline Model
    Standard residual convolutional architecture with 50 layers.
    Serves as the benchmark control to evaluate against
    Vision Transformers (Swin, SegFormer) and Attention CNNs (Attention U-Net, DeepLabV3+).
    """
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        try:
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            self.model = resnet50(weights=weights)
        except Exception:
            self.model = resnet50(weights=None)

        in_features = self.model.fc.in_features
        # Replace fully connected layer with custom classification head
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)
