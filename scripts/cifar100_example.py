import datetime
import hashlib
import pathlib
import pickle
import tarfile
import uuid

import numpy
import pandas
import requests
from PIL import Image

import fewshot
import fewshot.utilities

DATA_DIR = "/space/zach/data"


def unpickle(file):
    import pickle

    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def download_cifar100(data_dir: str) -> None:
    md5sum = "eb9058c3a382ffc7106e4002c42a8d85"
    download_link = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filepath = pathlib.Path(f"{data_dir}/cifar_100/cifar-100-python.tar.gz")

    filepath.parent.mkdir(parents=True, exist_ok=True)

    if not filepath.exists():
        response = requests.get(download_link)
        assert md5sum == hashlib.md5(response.content).hexdigest()
        with open(filepath, "wb") as f:
            f.write(response.content)

    with tarfile.open(filepath) as f:
        f.extractall(filepath.parent, filter=lambda member, _: member)

    with open(filepath.parent / "metadata.csv", "w") as f:
        f.write("filepath,superclass_name,class_name\n")

    with open(filepath.parent / f"cifar-100-python/meta", "rb") as f:
        metadata = pickle.load(f, encoding="bytes")

    index2class_name = metadata[b"fine_label_names"]
    index2superclass_name = metadata[b"coarse_label_names"]

    for split in ["train", "test"]:
        with open(filepath.parent / f"cifar-100-python/{split}", "rb") as f:
            dataset = pickle.load(f, encoding="bytes")

        data = dataset[b"data"]
        filenames = dataset[b"filenames"]
        class_names = dataset[b"fine_labels"]
        superclass_names = dataset[b"coarse_labels"]

        with open(filepath.parent / "metadata.csv", "+a") as f:
            for index in range(data.shape[0]):
                image_data = data[index]
                filename = filenames[index].decode("ascii")
                class_name = index2class_name[class_names[index]].decode("ascii")
                superclass_name = index2superclass_name[superclass_names[index]].decode("ascii")

                image_filepath = filepath.parent / f"images/{filename}"
                image_filepath.parent.mkdir(parents=True, exist_ok=True)

                f.write(f"images/{filename},{superclass_name},{class_name}\n")

                image_data = numpy.transpose(numpy.reshape(image_data, (3, 32, 32)), (1, 2, 0))

                image = Image.fromarray(numpy.uint8(image_data))
                image.save(image_filepath, format="png")


def get_datasets(data_dir: str, seed: int = 100) -> tuple[fewshot.Dataset, fewshot.Dataset, fewshot.Dataset]:
    data_dir_obj = pathlib.Path(data_dir)
    metadata_filepath = data_dir_obj / "metadata.csv"
    metadata = pandas.read_csv(metadata_filepath, header=0)
    class_names = metadata["class_name"].unique()

    random = numpy.random.RandomState(seed)

    random.shuffle(class_names)

    train = class_names[:60]
    val = class_names[60:80]
    test = class_names[80:]

    filepaths = []
    class_names_list = []
    with open(data_dir_obj / "train.csv", "+w") as f:
        f.write("filepath,class_name\n")
        for _, row in metadata.iterrows():
            filepath = row["filepath"]
            class_name = row["class_name"]

            if class_name in train:
                f.write(f"{filepath},{class_name}\n")
                filepaths.append(data_dir_obj / filepath)
                class_names_list.append(class_name)

    train_dataset = fewshot.Dataset(
        filepaths, class_names_list, transform_fn=fewshot.utilities.default_training_transform_fn
    )

    filepaths = []
    class_names_list = []
    with open(data_dir_obj / "val.csv", "+w") as f:
        f.write("filepath,class_name\n")
        for _, row in metadata.iterrows():
            filepath = row["filepath"]
            class_name = row["class_name"]

            if class_name in val:
                f.write(f"{filepath},{class_name}\n")
                filepaths.append(data_dir_obj / filepath)
                class_names_list.append(class_name)

    val_dataset = fewshot.Dataset(filepaths, class_names_list)

    filepaths = []
    class_names_list = []
    with open(data_dir_obj / "test.csv", "+w") as f:
        f.write("filepath,class_name\n")
        for _, row in metadata.iterrows():
            filepath = row["filepath"]
            class_name = row["class_name"]

            if class_name in test:
                f.write(f"{filepath},{class_name}\n")
                filepaths.append(data_dir_obj / filepath)
                class_names_list.append(class_name)

    test_dataset = fewshot.Dataset(filepaths, class_names_list)

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
    if False:
        download_cifar100(DATA_DIR)

    train_data, val_data, test_data = get_datasets(DATA_DIR + "/cifar_100")

    model = get_model()
    model_id = get_model_id()

    data_dir = "/data/carbon/models/"

    fewshot.train(
        model_id=model_id,
        data_dir=data_dir,
        model=model,
        training_dataset=train_data,
        validation_dataset=val_data,
        num_epochs=2,
        num_training_episodes=10,
        num_validation_episodes=10,
    )

    fewshot.evaluate(
        model_id=model_id,
        data_dir=data_dir,
        model=model,
        dataset=test_data,
        num_episodes=10,
    )
