import dataclasses
import logging
import os

import torch
import wandb
from torch.distributed.launcher.api import LaunchConfig, elastic_launch
from torch.nn.parallel import DistributedDataParallel

from .datasets import Dataset
from .models import BaseModel
from .sampler import Sampler
from .utilities import collate_fn


@dataclasses.dataclass
class TrainingConfig:
    model_id: str
    data_dir: str
    num_epochs: int = 1000
    num_training_episodes: int = 250
    num_validation_episodes: int = 250
    learning_rate: float = 1e-5
    num_classes: int = 5
    num_support_samples: int = 5
    num_query_samples: int = 8
    destroy_process_group: bool = False


def train(model: BaseModel, training_dataset: Dataset, validation_dataset: Dataset, *args, **kwargs) -> None:
    LOG = logging.getLogger("fewshot")
    LOG.info("Starting training...")
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
    launcher = elastic_launch(launch_config, train_subprocess)
    training_config = TrainingConfig(*args, **kwargs)
    launcher(model, training_dataset, validation_dataset, training_config)


def train_subprocess(
    model: BaseModel, training_dataset: Dataset, validation_dataset: Dataset, config: TrainingConfig
) -> None:
    LOG = logging.getLogger("fewshot")

    torch.distributed.init_process_group(backend="nccl")
    rank = int(os.environ["LOCAL_RANK"])

    if rank == 0:
        experiment_dir = os.path.join(config.data_dir, config.model_id)
        os.makedirs(experiment_dir, exist_ok=True)

        run = wandb.init(
            project="fewshot-testing",
            id=config.model_id,
            dir=experiment_dir,
            config=dataclasses.asdict(config),
        )

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_sampler=Sampler(
            validation_dataset,
            num_classes=config.num_classes,
            num_support_samples=config.num_support_samples,
            num_query_samples=config.num_query_samples,
            num_episodes=config.num_validation_episodes,
        ),
        collate_fn=collate_fn,
        num_workers=4,
    )

    model.to(rank)
    distributed_model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(distributed_model.parameters(), lr=config.learning_rate)
    global_step = 0

    distributed_model.eval()
    losses = []
    accuracies = []
    for episode_index, episode in enumerate(validation_dataloader):
        episode = episode.to(rank)

        with torch.no_grad():
            logits = distributed_model(episode)
            loss = loss_fn(logits, episode.y)
            accuracy = (logits.argmax(dim=1) == episode.y).float().mean()

        losses.append(loss.item())
        accuracies.append(accuracy.item())

        if rank == 0:
            step_log(
                -1,
                config.num_epochs,
                episode_index,
                config.num_validation_episodes,
                loss.item(),
                accuracy.item(),
                training=False,
            )

    if rank == 0:
        validation_end_log(sum(losses) / len(losses), sum(accuracies) / len(accuracies))
        run.log(
            {
                "validation-loss": sum(losses) / len(losses),
                "validation-accuracy": sum(accuracies) / len(accuracies),
                "epoch": 0,
            },
            step=global_step,
        )

    training_dataloader = torch.utils.data.DataLoader(
        training_dataset,
        batch_sampler=Sampler(
            training_dataset,
            num_classes=config.num_classes,
            num_support_samples=config.num_support_samples,
            num_query_samples=config.num_query_samples,
            num_episodes=config.num_training_episodes,
        ),
        collate_fn=collate_fn,
        num_workers=4,
    )

    for epoch_index in range(config.num_epochs):
        distributed_model.train()
        for episode_index, episode in enumerate(training_dataloader):
            episode = episode.to(rank)

            optimizer.zero_grad()

            logits = distributed_model(episode)
            loss = loss_fn(logits, episode.y)
            accuracy = (logits.argmax(dim=1) == episode.y).float().mean()

            loss.backward()
            optimizer.step()
            global_step += 1

            if rank == 0:
                step_log(
                    epoch_index,
                    config.num_epochs,
                    episode_index,
                    config.num_training_episodes,
                    loss.item(),
                    accuracy.item(),
                )
                run.log(
                    {
                        "training-loss": loss.item(),
                        "training-accuracy": accuracy.item(),
                        "epoch": epoch_index + 1,
                    },
                    step=global_step,
                )

        if rank == 0:
            save_model(experiment_dir, distributed_model)

        distributed_model.eval()
        losses = []
        accuracies = []
        for episode_index, episode in enumerate(validation_dataloader):
            episode = episode.to(rank)

            with torch.no_grad():
                logits = distributed_model(episode)
                loss = loss_fn(logits, episode.y)
                accuracy = (logits.argmax(dim=1) == episode.y).float().mean()

            losses.append(loss.item())
            accuracies.append(accuracy.item())

            if rank == 0:
                step_log(
                    epoch_index,
                    config.num_epochs,
                    episode_index,
                    config.num_validation_episodes,
                    loss.item(),
                    accuracy.item(),
                    training=False,
                )

        if rank == 0:
            validation_end_log(sum(losses) / len(losses), sum(accuracies) / len(accuracies))
            run.log(
                {
                    "validation-loss": sum(losses) / len(losses),
                    "validation-accuracy": sum(accuracies) / len(accuracies),
                    "epoch": epoch_index + 1,
                },
                step=global_step,
            )

    if rank == 0:
        run.finish()
        LOG.info("Training complete.")


def validation_end_log(loss: float, accuracy: float) -> None:
    LOG = logging.getLogger("fewshot")
    LOG.info(f"Validation sweep average loss: {loss}")
    LOG.info(f"Validation sweep average accuracy: {accuracy}")


def step_log(
    epoch_index: int,
    num_epochs: int,
    episode_index: int,
    num_episodes: int,
    loss: float,
    accuracy: float,
    training: bool = True,
) -> None:
    message = f"Epoch {epoch_index + 1} / {num_epochs} | "

    if training:
        message += f"Training Episode {episode_index + 1: >3} / {num_episodes} | "
    else:
        message += f"Validation Episode {episode_index + 1: >3} / {num_episodes} | "

    message += f"Loss: {loss:2.2f} | Accuracy: {accuracy:.3f}"

    LOG = logging.getLogger("fewshot")
    LOG.info(message)


def save_model(experiment_dir: str, model: torch.nn.Module) -> None:
    weights_dir = os.path.join(experiment_dir, "weights")
    filepath = os.path.join(weights_dir, "latest.pt")

    LOG = logging.getLogger("fewshot")
    LOG.info(f"Saving model to {filepath}")

    os.makedirs(weights_dir, exist_ok=True)
    torch.save(model.state_dict(), filepath)
