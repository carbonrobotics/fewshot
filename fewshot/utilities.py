import random
from typing import List

import torch
import torchvision  # type: ignore
from PIL import Image

from .dataset_types import Datapoint, Episode

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def default_load_fn(filepath: str) -> Image.Image:
    with Image.open(filepath) as image:
        image = image.convert("RGB")

    return image


def default_training_transform_fn(image: Image.Image) -> torch.Tensor:
    if random.random() > 0.5:
        image = torchvision.transforms.functional.hflip(image)

    if random.random() > 0.5:
        image = torchvision.transforms.functional.vflip(image)

    tensor_image = torchvision.transforms.functional.to_tensor(image)
    tensor_image = torchvision.transforms.functional.resize(tensor_image, (224, 224))
    tensor_image = torchvision.transforms.functional.normalize(tensor_image, IMAGENET_MEAN, IMAGENET_STD)

    return tensor_image


def default_evaluation_transform_fn(image: Image.Image) -> torch.Tensor:
    tensor_image = torchvision.transforms.functional.to_tensor(image)
    tensor_image = torchvision.transforms.functional.resize(tensor_image, (224, 224))
    tensor_image = torchvision.transforms.functional.normalize(tensor_image, IMAGENET_MEAN, IMAGENET_STD)

    return tensor_image


def collate_fn(datapoints: List[Datapoint]) -> Episode:
    return Episode(datapoints)
