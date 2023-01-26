from pathlib import Path

from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

from SCP.datasets.presets import load_test_presets
from SCP.utils.common import load_config
from SCP.utils.plots import plot_image, plot_grid, show_img_from_dataloader, show_grid_from_dataloader


def load_CIFAR10_BW(batch_size, datasets_path: Path, *args, **kwargs):

    transform = load_test_presets(img_shape=[1, 28, 28])

    test_data_CIFAR10 = torchvision.datasets.CIFAR10(
        root=datasets_path,
        train=False,
        download=True,
        transform=transform,
    )
    test_loader_CIFAR10 = DataLoader(
        test_data_CIFAR10,
        batch_size=batch_size,
        shuffle=False
    )
    return test_loader_CIFAR10


def load_CIFAR10(batch_size, datasets_path: Path, test_only=False, image_shape=(3, 32, 32),
                 workers=4, *args, **kwargs):

    test_transform = load_test_presets(img_shape=image_shape)

    test_data_CIFAR10 = torchvision.datasets.CIFAR10(
        root=datasets_path,
        train=False,
        download=True,
        transform=test_transform,
    )
    test_loader_CIFAR10 = DataLoader(
        test_data_CIFAR10,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    if test_only is False:
        train_transform = T.Compose(
            [
                # T.RandomRotation(15, ),
                # T.RandomCrop(400),
                T.RandomHorizontalFlip(),
                # T.RandomVerticalFlip(),
                T.ToTensor()
            ]
        )
        train_data_CIFAR10 = torchvision.datasets.CIFAR10(
            root=datasets_path,
            train=True,
            download=True,
            transform=train_transform,
        )

        train_loader_CIFAR10 = DataLoader(
            train_data_CIFAR10,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,

        )
        return train_data_CIFAR10, train_loader_CIFAR10, test_loader_CIFAR10

    else:
        return test_loader_CIFAR10


def load_CIFAR100_BW(batch_size, datasets_path: Path, *args, **kwargs):

    transform = load_test_presets(img_shape=[1, 28, 28])

    test_data_CIFAR10 = torchvision.datasets.CIFAR100(
        root=datasets_path,
        train=False,
        download=True,
        transform=transform,
    )
    test_loader_CIFAR10 = DataLoader(
        test_data_CIFAR10,
        batch_size=batch_size,
        shuffle=False,
    )
    return test_loader_CIFAR10


def load_CIFAR100(batch_size, datasets_path: Path, test_only=False, image_shape=(3, 32, 32), *args, **kwargs):

    test_transform = load_test_presets(img_shape=image_shape)
    test_data_CIFAR100 = torchvision.datasets.CIFAR100(
        root=datasets_path,
        train=False,
        download=True,
        transform=test_transform,
    )
    test_loader_CIFAR100 = DataLoader(
        test_data_CIFAR100,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    if test_only is False:
        train_transform = T.Compose(
            [
                # T.RandomRotation(15, ),
                # T.RandomCrop(400),
                T.RandomHorizontalFlip(),
                # T.RandomVerticalFlip(),
                T.ToTensor()
            ]
        )
        train_data_CIFAR100 = torchvision.datasets.CIFAR100(
            root=datasets_path,
            train=True,
            download=True,
            transform=train_transform,
        )

        train_loader_CIFAR100 = DataLoader(
            train_data_CIFAR100,
            batch_size=batch_size,
            shuffle=True
        )
        return train_data_CIFAR100, train_loader_CIFAR100, test_loader_CIFAR100

    else:
        return test_loader_CIFAR100


def load_CIFAR100_non_overlapping(batch_size, datasets_path: Path, test_only=True,
                                  image_shape=(3, 32, 32), *args, **kwargs):
    test_transform = load_test_presets(img_shape=image_shape)
    target_transform = 'as'
    test_data_CIFAR100 = torchvision.datasets.CIFAR100(
        root=datasets_path,
        train=False,
        download=True,
        transform=test_transform,
    )
    test_loader_CIFAR100 = DataLoader(
        test_data_CIFAR100,
        batch_size=batch_size,
        shuffle=False,
    )

    if test_only is False:
        train_transform = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.ToTensor()
            ]
        )
        train_data_CIFAR100 = torchvision.datasets.CIFAR100(
            root=datasets_path,
            train=True,
            download=True,
            transform=train_transform,
            target_transform=target_transform,
        )

        train_loader_CIFAR100 = DataLoader(
            train_data_CIFAR100,
            batch_size=batch_size,
            shuffle=True
        )
        return train_data_CIFAR100, train_loader_CIFAR100, test_loader_CIFAR100

    else:
        return test_loader_CIFAR100


if __name__ == "__main__":
    import os

    current_file_path = Path(os.path.dirname(__file__))
    datasets_path = current_file_path.parent.parent / 'datasets'

    d_cif10, train_loader10, test_loader10 = load_CIFAR10(64, datasets_path, test_only=False)
    d_cif100, train_loader100, test_loader100 = load_CIFAR100(64, datasets_path, test_only=False)

    print(d_cif10.classes)
    print(d_cif100.classes)

    dat = 0
    if dat == 10:
        loader = load_CIFAR10(64, datasets_path, test_only=True)
        imgs, targets = next(iter(loader))
        print(imgs.max())
        plot_grid(imgs)
        # plot_image(imgs[1].permute(1, 2, 0))
    elif dat == 100:
        dataset, train_loader, test_loader = load_CIFAR100(64, datasets_path)
        show_img_from_dataloader(train_loader, img_pos=0, number_of_iterations=1)
        show_grid_from_dataloader(train_loader)
