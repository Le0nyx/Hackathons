import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class PlantClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Carico EfficientNet-B0 pretrained e cambio ultimo layer
        self.backbone = efficientnet_b0(pretrained=False)
        self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
