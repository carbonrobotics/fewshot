import datetime
import os
import uuid

import numpy

import fewshot
import fewshot.utilities


def get_datasets(seed: int = 100) -> tuple[fewshot.Dataset, fewshot.Dataset, fewshot.Dataset]:
    imagenet_dir = "/imagenet/imagenet/"

    class_names = []
    data: dict[str, list[str]] = {}
    for class_name in os.listdir(imagenet_dir):
        class_dir = os.path.join(imagenet_dir, class_name)

        data[class_name] = []
        class_names.append(class_name)

        for filename in os.listdir(class_dir):
            filepath = os.path.join(class_dir, filename)
            data[class_name].append(filepath)

    random = numpy.random.RandomState(seed)
    random.shuffle(class_names)

    train = class_names[:600]
    val = class_names[600:800]
    test = class_names[800:]

    filepaths = []
    labels = []
    for class_name in train:
        filepaths.extend(data[class_name])
        labels.extend([class_name] * len(data[class_name]))

    train_dataset = fewshot.Dataset(filepaths, labels)

    filepaths = []
    labels = []
    for class_name in val:
        filepaths.extend(data[class_name])
        labels.extend([class_name] * len(data[class_name]))

    val_dataset = fewshot.Dataset(filepaths, labels)

    filepaths = []
    labels = []
    for class_name in test:
        filepaths.extend(data[class_name])
        labels.extend([class_name] * len(data[class_name]))

    test_dataset = fewshot.Dataset(filepaths, labels)

    return train_dataset, val_dataset, test_dataset


def get_model():
    encoder = fewshot.encoders.ResNet(architecture="resnet50", pretrained=False)
    model = fewshot.models.ProtoNets(encoder)
    return model


def get_model_id():
    date = str(datetime.date.today()).replace("-", "")
    identifier = str(uuid.uuid4())[:8]
    return f"fst-{date}-{identifier}"


if __name__ == "__main__":
    train_data, val_data, test_data = get_datasets()

    model = get_model()
    model_id = get_model_id()
    data_dir = "/data/carbon/models/"

    fewshot.train(
        model_id=model_id,
        data_dir=data_dir,
        model=model,
        training_dataset=train_data,
        validation_dataset=val_data,
        num_training_episodes=2000,
    )

    fewshot.evaluate(
        model_id=model_id,
        data_dir=data_dir,
        model=model,
        dataset=test_data,
    )
