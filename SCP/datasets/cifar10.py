from pathlib import Path

import torch
import torchvision


def load_CIFAR10_BW(batch_size, datasets_path: Path, *args, **kwargs):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.Resize((28, 28))
        ]
    )

    test_data_CIFAR10 = torchvision.datasets.CIFAR10(
        root=datasets_path,
        train=False,
        download=True,
        transform=transform,
    )
    test_loader_CIFAR10 = torch.utils.data.DataLoader(
        test_data_CIFAR10,
        batch_size=batch_size,
        shuffle=False
    )
    return test_loader_CIFAR10


def load_CIFAR10(batch_size, datasets_path: Path, test_only=False, *args, **kwargs):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor()
        ]
    )
    test_data_CIFAR10 = torchvision.datasets.CIFAR10(
        root=datasets_path,
        train=False,
        download=True,
        transform=transform,
    )
    test_loader_CIFAR10 = torch.utils.data.DataLoader(
        test_data_CIFAR10,
        batch_size=batch_size,
        shuffle=False
    )

    if test_only is False:
        train_data_CIFAR10 = torchvision.datasets.CIFAR10(
            root=datasets_path,
            train=True,
            download=True,
            transform=transform,
        )

        train_loader_CIFAR10 = torch.utils.data.DataLoader(
            train_data_CIFAR10,
            batch_size=batch_size,
            shuffle=True
        )
        return train_data_CIFAR10, train_loader_CIFAR10, test_loader_CIFAR10

    else:
        return test_loader_CIFAR10
