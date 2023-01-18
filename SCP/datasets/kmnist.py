from pathlib import Path

import torch
import torchvision


def load_KMNIST(batch_size, datasets_path: Path, test_only=False, *args, **kwargs):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )

    test_data_KMNIST = torchvision.datasets.KMNIST(
        root=datasets_path,
        train=False,
        download=True,
        transform=transform,
    )
    test_loader_KMNIST = torch.utils.data.DataLoader(
        torchvision.datasets.KMNIST(
            root=datasets_path,
            train=False,
            transform=transform,
        ),
        batch_size=batch_size
    )
    if test_only is False:
        train_data_KMNIST = torchvision.datasets.KMNIST(
            root=datasets_path,
            train=True,
            download=True,
            transform=transform,
        )

        train_loader_KMNIST = torch.utils.data.DataLoader(
            train_data_KMNIST,
            batch_size=batch_size,
            shuffle=True
        )
        return train_data_KMNIST, train_loader_KMNIST, test_loader_KMNIST
    else:
        return test_loader_KMNIST
