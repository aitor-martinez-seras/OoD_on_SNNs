from pathlib import Path

import torch
import torchvision


def load_Fashion_MNIST(batch_size, datasets_path: Path, test_only=False):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )
    test_data_fashion_MNIST = torchvision.datasets.FashionMNIST(
        root=datasets_path,
        train=False,
        download=True,
        transform=transform,
    )
    test_loader_fashion_MNIST = torch.utils.data.DataLoader(
        test_data_fashion_MNIST,
        batch_size=batch_size
    )

    if test_only is False:
        train_data_fashion_MNIST = torchvision.datasets.FashionMNIST(
            root=datasets_path,
            train=True,
            download=True,
            transform=transform,
        )
        train_loader_fashion_MNIST = torch.utils.data.DataLoader(
            train_data_fashion_MNIST,
            batch_size=batch_size,
            shuffle=True
        )
        return train_data_fashion_MNIST, train_loader_fashion_MNIST, test_loader_fashion_MNIST
    else:
        return test_loader_fashion_MNIST


if __name__ == "__main__":
    load_Fashion_MNIST(64, Path(r'/datasets'))
