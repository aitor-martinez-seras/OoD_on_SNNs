from pathlib import Path

import torch
import torchvision
from torchvision.transforms import Lambda


def load_MNIST_Letters(batch_size, datasets_path: Path, test_only=False):
    # The rotation and horizontal flip are for getting the images in a similar
    # way to the MNIST dataset i.e. vertical and readable from left to right
    transform = torchvision.transforms.Compose(
        [
            lambda img: torchvision.transforms.functional.rotate(img, -90),
            lambda img: torchvision.transforms.functional.hflip(img),
            torchvision.transforms.ToTensor(),
        ]
    )
    test_data_letters = torchvision.datasets.EMNIST(
        root=datasets_path,
        split="letters",
        train=False,
        download=True,
        transform=transform,
        target_transform=Lambda(
            lambda y: y - 1
        ),
    )
    # Eliminate the first class, that is non-existant for our case
    test_data_letters.classes = test_data_letters.classes[1:]
    test_loader_letters = torch.utils.data.DataLoader(
        test_data_letters,
        batch_size=batch_size,
        shuffle=False
    )
    # To obtain 10.000 test samples, pass trought the function
    # test_loader_letters = parse_size_of_dataloader(test_loader_letters, batch_size)
    if test_only is False:
        train_data_letters = torchvision.datasets.EMNIST(
            root=datasets_path,
            split="letters",
            train=True,
            download=True,
            transform=transform,
            target_transform=Lambda(
                lambda y: y - 1
            ),
        )
        train_data_letters.classes = train_data_letters.classes[1:]
        train_loader_letters = torch.utils.data.DataLoader(
            train_data_letters,
            batch_size=batch_size,
            shuffle=True
        )

        return train_data_letters, train_loader_letters, test_loader_letters
    else:
        return test_loader_letters
