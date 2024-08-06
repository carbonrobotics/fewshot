import logging
import os

import torch
from torch.nn.parallel import DistributedDataParallel

from .datasets import Dataset
from .models import BaseModel
from .sampler import Sampler
from .utilities import collate_fn


def evaluate(
    model_id: str,
    data_dir: str,
    model: BaseModel,
    dataset: Dataset,
    num_episodes: int = 250,
    destroy_process_group: bool = True,
):
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    rank = int(os.environ["LOCAL_RANK"])

    if rank == 0:
        LOG = logging.getLogger("fewshot")
        LOG.info("Starting evaluation...")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=Sampler(
            dataset,
            num_classes=5,
            num_support_samples=5,
            num_query_samples=8,
            num_episodes=num_episodes,
        ),
        collate_fn=collate_fn,
        num_workers=4,
    )

    model.to(rank)
    distributed_model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    loss_fn = torch.nn.CrossEntropyLoss()

    distributed_model.eval()
    losses = []
    accuracies = []
    for episode_index, episode in enumerate(dataloader):
        episode = episode.to(rank)

        with torch.no_grad():
            logits = distributed_model(episode)
            loss = loss_fn(logits, episode.y)
            accuracy = (logits.argmax(dim=1) == episode.y).float().mean()

        losses.append(loss.item())
        accuracies.append(accuracy.item())

        if rank == 0:
            LOG.info(
                f"Rank {rank}: Evaluation Episode {episode_index + 1} / {num_episodes} | Loss: {loss.item()} | Accuracy: {accuracy.item()}",
            )

    loss = sum(losses) / len(losses)
    accuracy = sum(accuracies) / len(accuracies)

    if rank == 0:
        LOG.info(f"Evaluation sweep average loss: {loss}")
        LOG.info(f"Evaluation sweep average accuracy: {accuracy}")

    if destroy_process_group:
        torch.distributed.destroy_process_group()

    if rank == 0:
        LOG.info("Evaluation complete.")

    return loss, accuracy
