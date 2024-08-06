import torch

from .base import BaseEncoder


class SimpleCNN(BaseEncoder):
    def __init__(self, num_classes: int = 10, classify: bool = True):
        super(SimpleCNN, self).__init__()

        self._num_classes = num_classes
        self._classify = classify

        self._output_dim = 512

        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(2)

        self.conv_1 = torch.nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.conv_2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv_3 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv_4 = torch.nn.Conv2d(128, self._output_dim, kernel_size=3, stride=1, padding=1)

        self.linear = torch.nn.Linear(self._output_dim, self._num_classes)

    @property
    def output_size(self) -> int:
        return self._output_dim

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def classify(self) -> bool:
        return self._classify

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv_2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv_3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv_4(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = x.mean(dim=(2, 3)).flatten(start_dim=1)

        if self._classify:
            x = self.linear(x)

        return x
