import dataclasses
import logging
import os

import torch
import wandb
from torch.distributed.launcher.api import elastic_launch
from torch.nn.parallel import DistributedDataParallel

from .datasets import Dataset
from .models import BaseModel
from .sampler import Sampler
from .utilities import collate_fn, get_elastic_launcher_config


@dataclasses.dataclass
class EvaluationConfig:
    model_id: str
    data_dir: str
    num_episodes: int = 250
    num_classes: int = 5
    num_support_samples: int = 5
    num_query_samples: int = 8


def evaluate(model: BaseModel, dataset: Dataset, *args, **kwargs) -> None:
    LOG = logging.getLogger("fewshot")
    LOG.info("Starting evaluation...")

    launch_config = get_elastic_launcher_config()
    launcher = elastic_launch(launch_config, evaluate_subprocess)
    evaluation_config = EvaluationConfig(*args, **kwargs)
    launcher(model, dataset, evaluation_config)


def evaluate_subprocess(
    model: BaseModel,
    dataset: Dataset,
    config: EvaluationConfig,
):
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    rank = int(os.environ["LOCAL_RANK"])

    if rank == 0:
        experiment_dir = os.path.join(config.data_dir, config.model_id)
        LOG = logging.getLogger("fewshot")
        LOG.info("Starting evaluation...")

        run = wandb.init(
            project="fewshot-testing",
            id=config.model_id,
            dir=experiment_dir,
            config=dataclasses.asdict(config),
            resume="must",
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=Sampler(
            dataset,
            num_classes=config.num_classes,
            num_support_samples=config.num_support_samples,
            num_query_samples=config.num_query_samples,
            num_episodes=config.num_episodes,
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
                f"Rank {rank}: Evaluation Episode {episode_index + 1} / {config.num_episodes} | Loss: {loss.item()} | Accuracy: {accuracy.item()}",
            )

    loss = sum(losses) / len(losses)
    accuracy = sum(accuracies) / len(accuracies)

    if rank == 0:
        LOG.info(f"Evaluation sweep average loss: {loss}")
        LOG.info(f"Evaluation sweep average accuracy: {accuracy}")
        run.log(
            {
                "test-loss": loss,
                "test-accuracy": accuracy,
            }
        )

        run.finish()
        LOG.info("Evaluation complete.")

    return loss, accuracy
