from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms

from SCP.utils.common import load_config
from SCP.utils.plots import plot_image, plot_grid


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
    transform = transforms.Compose(
        [
            # transforms.RandomRotation(15, ),
            # transforms.RandomCrop(400),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor()
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


if __name__ == "__main__":
    import os
    current_file_path = Path(os.path.dirname(__file__))
    datasets_path = current_file_path.parent.parent / 'datasets'
    loader = load_CIFAR10(64, datasets_path, test_only=True)
    imgs, targets = next(iter(loader))
    print(imgs.max())
    plot_grid(imgs)
    # plot_image(imgs[1].permute(1, 2, 0))
