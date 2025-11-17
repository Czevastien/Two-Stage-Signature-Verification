import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class TripletNetwork(nn.Module):
    def __init__(self, embedding_dim=128, pretrained=True, in_channels=1):
        super().__init__()

        # ResNet-18 backbone
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )

        if in_channels == 1:
            backbone.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        self.fc = nn.Sequential(
            nn.Linear(backbone.fc.in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, embedding_dim),
        )

    def forward_once(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, anchor, positive, negative):
        return (
            self.forward_once(anchor),
            self.forward_once(positive),
            self.forward_once(negative),
        )
