import logging

from . import encoders, models
from .dataset_types import Episode
from .datasets import Dataset
from .evaluate import evaluate
from .train import train

logging.basicConfig()
logging.getLogger("fewshot").setLevel(logging.INFO)

__all__ = [
    "Dataset",
    "Episode",
    "encoders",
    "models",
    "train",
    "evaluate",
]
