import random
from pathlib import Path

import torch
import torchvision

from constants import DATASETS_PATH


def load_MNIST(batch_size, datasets_path: Path, test_only=False):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )
    test_data_MNIST = torchvision.datasets.MNIST(
        root=datasets_path,
        train=False,
        download=True,
        transform=transform,
    )
    test_loader_MNIST = torch.utils.data.DataLoader(
        test_data_MNIST,
        batch_size=batch_size
    )
    if test_only is False:
        train_data_MNIST = torchvision.datasets.MNIST(
            root=datasets_path,
            train=True,
            download=True,
            transform=transform,
        )
        train_loader_MNIST = torch.utils.data.DataLoader(
            train_data_MNIST,
            batch_size=batch_size,
            shuffle=True
        )
        return train_data_MNIST, train_loader_MNIST, test_loader_MNIST
    else:
        return test_loader_MNIST


def square_creation(input_tensor: torch.Tensor):
    posible_values = [2, 20]
    mean = int(torch.mean(input_tensor[0]) * 100)
    random.seed(mean)
    x_rnd = random.randint(0, 1)
    x_start = posible_values[x_rnd]
    random.seed(mean - 1)
    y_rnd = random.randint(0, 1)
    y_start = posible_values[y_rnd]
    input_tensor[:, x_start:x_start + 6, y_start:y_start + 6] = torch.ones((1, 6, 6), dtype=torch.float32)
    return input_tensor


def load_MNIST_square(batch_size, datasets_path: Path, *args):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(square_creation)
        ]
    )
    test_data_MNIST_square = torchvision.datasets.MNIST(
        root=datasets_path,
        train=False,
        download=True,
        transform=transform,
    )
    test_loader_MNIST_square = torch.utils.data.DataLoader(
        test_data_MNIST_square,
        batch_size=batch_size
    )
    return test_loader_MNIST_square
