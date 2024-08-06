import pathlib
from typing import List

import torch


class DatapointRole:
    SUPPORT = "SUPPORT"
    QUERY = "QUERY"


class Datapoint:
    def __init__(self, image: torch.Tensor, filepath: pathlib.Path, label: str, role: DatapointRole):
        self._image = image
        self._filepath = filepath
        self._label = label
        self._role = role

    @property
    def image(self):
        return self._image

    @property
    def filepath(self):
        return self._filepath

    @property
    def label(self):
        return self._label

    @property
    def role(self):
        return self._role


class Episode:
    def __init__(self, datapoints: List[Datapoint]):
        self._datapoints = datapoints
        self._class_names = list(set([datapoint.label for datapoint in datapoints]))
        self._roles = [datapoint.role for datapoint in datapoints]

        self._x = torch.stack([datapoint.image for datapoint in datapoints])
        self._y = torch.tensor(
            [
                self._class_names.index(datapoint.label)
                for datapoint in self._datapoints
                if datapoint.role == DatapointRole.QUERY
            ]
        )

    @property
    def datapoints(self):
        return self._datapoints

    @property
    def class_names(self):
        return self._class_names

    @property
    def roles(self):
        return self._roles

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def to(self, gpu_id: int):
        self._x = self._x.to(gpu_id)
        self._y = self._y.to(gpu_id)
        return self
