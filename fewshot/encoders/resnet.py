import torch
import torchvision  # type: ignore

from .base import BaseEncoder


class ResNet(BaseEncoder):
    def __init__(self, architecture: str = "resnet18", pretrained: bool = True):
        super().__init__()

        if architecture == "resnet18":
            if pretrained:
                self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.model = torchvision.models.resnet18()
        elif architecture == "resnet50":
            if pretrained:
                self.model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                self.model = torchvision.models.resnet50()

    @property
    def embedding_size(self):
        return self.model.fc.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        return x
