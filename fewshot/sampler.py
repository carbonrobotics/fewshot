from typing import Iterator, List, Tuple

import numpy
import torch

from .dataset_types import DatapointRole
from .datasets import Dataset


class Sampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset: Dataset,
        num_classes: int,
        num_support_samples: int,
        num_query_samples: int,
        num_episodes: int,
        seed=1000,
    ):
        self._dataset = dataset
        self._num_classes = num_classes
        self._num_support_samples = num_support_samples
        self._num_query_samples = num_query_samples
        self._num_episodes = num_episodes

        self._rng = numpy.random.RandomState(seed)

    def __iter__(self) -> Iterator[List[Tuple[int, str]]]:
        for _ in range(self._num_episodes):
            class_names = self._rng.choice(self._dataset.class_names, self._num_classes, replace=False)

            support_roles = [DatapointRole.SUPPORT] * self._num_support_samples
            query_roles = [DatapointRole.QUERY] * self._num_query_samples

            samples_list = []
            roles_list = []
            for class_name in class_names:
                samples = [
                    int(sample)
                    for sample in self._rng.choice(
                        self._dataset.name2indices[class_name],
                        self._num_support_samples + self._num_query_samples,
                        replace=False,
                    )
                ]

                samples_list.extend(samples)
                roles_list.extend(support_roles + query_roles)

            episode = [(a, b) for a, b in zip(samples_list, roles_list)]

            yield episode
