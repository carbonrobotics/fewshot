import pathlib
from typing import Any, Callable, List, Tuple

import torch
from PIL import Image

from .dataset_types import Datapoint, DatapointRole
from .utilities import (default_evaluation_transform_fn, default_load_fn,
                        default_training_transform_fn)


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: List[str],
        labels: List[str],
        load_fn: Callable[[Any], Image.Image] = default_load_fn,
        transform_fn: Callable[[Any], torch.Tensor] = default_evaluation_transform_fn,
    ) -> None:
        self._data = data
        self._labels = labels
        self._load_fn = load_fn
        self._transform_fn = transform_fn

        self._class_names = list(set(self._labels))
        self._name2indices = {
            name: [i for i, label in enumerate(self._labels) if label == name] for name in self._class_names
        }

    @property
    def class_names(self):
        return self._class_names

    @property
    def name2indices(self):
        return self._name2indices

    def __getitem__(self, item: Tuple[int, DatapointRole]) -> Datapoint:
        index, role = item

        filepath = self._data[index]
        label = self._labels[index]

        pil_image = self._load_fn(filepath)
        tensor_image = self._transform_fn(pil_image)

        datapoint = Datapoint(tensor_image, pathlib.Path(filepath), label, role)

        return datapoint
