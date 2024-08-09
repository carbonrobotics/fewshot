import os
import uuid
from typing import Dict, List, Tuple

import numpy
import torchvision  # type: ignore

import fewshot
import fewshot.utilities


def get_datasets(data_dir: str, seed: int = 100) -> Tuple[fewshot.Dataset, fewshot.Dataset, fewshot.Dataset]:
    torch_dataset = {
        "train": torchvision.datasets.DTD(root=data_dir, split="train", download=True),
        "val": torchvision.datasets.DTD(root=data_dir, split="val", download=True),
        "test": torchvision.datasets.DTD(root=data_dir, split="test", download=True),
    }

    os.makedirs(os.path.join(data_dir, "local_data"), exist_ok=True)

    dataset: Dict[str, List[str]] = {}
    for split, data in torch_dataset.items():
        for index, (image, target) in enumerate(data):
            key = f"{split}-{index}"

            filepath = os.path.join(data_dir, f"local_data/{key}.jpg")

            if target not in dataset.keys():
                dataset[target] = [filepath]
            else:
                dataset[target].append(filepath)

            image.save(filepath)

    class_names = list(dataset.keys())
    random = numpy.random.RandomState(seed)
    random.shuffle(class_names)

    train_dataset = fewshot.Dataset(
        [datapoint for class_name in class_names[:60] for datapoint in dataset[class_name]],
        [class_name for class_name in class_names[:60] for _ in dataset[class_name]],
    )

    val_dataset = fewshot.Dataset(
        [datapoint for class_name in class_names[60:80] for datapoint in dataset[class_name]],
        [class_name for class_name in class_names[60:80] for _ in dataset[class_name]],
    )
    test_dataset = fewshot.Dataset(
        [datapoint for class_name in class_names[80:] for datapoint in dataset[class_name]],
        [class_name for class_name in class_names[80:] for _ in dataset[class_name]],
    )

    return train_dataset, val_dataset, test_dataset


def get_model():
    encoder = fewshot.encoders.ResNet(architecture="resnet50", pretrained=False)
    model = fewshot.models.ProtoNets(encoder)
    return model


if __name__ == "__main__":
    data_dir = "/data/fewshot/cifar100-example/"

    train_data, val_data, test_data = get_datasets(data_dir)

    model = get_model()
    model_id = str(uuid.uuid4())

    fewshot.train(
        model_id=model_id,
        data_dir=data_dir,
        model=model,
        training_dataset=train_data,
        validation_dataset=val_data,
        num_epochs=10,
    )

    fewshot.evaluate(
        model_id=model_id,
        data_dir=data_dir,
        model=model,
        dataset=test_data,
    )
