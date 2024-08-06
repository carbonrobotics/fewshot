import torch

from ..dataset_types import DatapointRole, Episode
from ..encoders import BaseEncoder
from .base import BaseModel


class ProtoNets(BaseModel):
    def __init__(self, encoder: BaseEncoder):
        super().__init__()

        self._encoder = encoder

    def forward(self, episode: Episode) -> torch.Tensor:
        embeddings = self._encoder(episode.x)

        support_set_list = []
        for class_name in episode.class_names:
            support_indices = []
            for index, datapoint in enumerate(episode.datapoints):
                if datapoint.label == class_name and datapoint.role == DatapointRole.SUPPORT:
                    support_indices.append(index)

            support_emebddings = embeddings[support_indices]
            support_centroid = torch.mean(support_emebddings, dim=0)

            support_set_list.append(support_centroid)

        support_set = torch.stack(support_set_list, dim=0)

        query_indices = []
        for index, datapoint in enumerate(episode.datapoints):
            if datapoint.role == DatapointRole.QUERY:
                query_indices.append(index)

        query_set = embeddings[query_indices]

        logits = -torch.cdist(query_set, support_set)

        return logits
