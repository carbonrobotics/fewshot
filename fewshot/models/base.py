import torch

from ..dataset_types import Episode


class BaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, episode: Episode) -> torch.Tensor:
        raise NotImplementedError()
