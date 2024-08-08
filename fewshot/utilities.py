import random
from typing import List

import torch
import torchvision  # type: ignore
from PIL import Image
from torch.distributed.launcher.api import LaunchConfig

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


def get_elastic_launcher_config() -> LaunchConfig:
    launch_config = LaunchConfig(
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=8,
        run_id="none",
        role="default",
        rdzv_endpoint="127.0.0.1:29500",
        rdzv_backend="static",
        rdzv_configs={"rank": 0, "timeout": 900},
        rdzv_timeout=-1,
        max_restarts=0,
        monitor_interval=5,
        start_method="spawn",
        metrics_cfg={},
        local_addr=None,
    )
    return launch_config
