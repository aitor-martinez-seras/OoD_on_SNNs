from pathlib import Path

import torchvision
import torchvision.transforms as T
from torchvision.datasets import VisionDataset

from SCP.datasets.utils import DatasetCustomLoader
from SCP.utils.plots import show_img_from_dataloader, show_grid_from_dataloader


class CIFAR10(DatasetCustomLoader):

    def __init__(self, root_path, *args, **kwargs):
        super().__init__(torchvision.datasets.CIFAR10, root_path=root_path)

    def _train_data(self, transform) -> VisionDataset:
        return self.dataset(
            root=self.root_path,
            train=True,
            download=True,
            transform=transform,
        )

    def _test_data(self, transform) -> VisionDataset:
        return self.dataset(
            root=self.root_path,
            train=False,
            download=True,
            transform=transform,
        )

    def _train_transformation(self, output_shape):
        return T.Compose(
            [
                T.ToTensor(),
                T.Resize(output_shape),
                # T.RandomCrop(output_shape[0], padding=4),
                # T.RandomRotation(15, ),
                T.RandomHorizontalFlip(),

            ]
        )

    def _test_transformation(self, output_shape):
        return T.Compose(
            [
                T.ToTensor(),
                T.Resize(output_shape),

            ]
        )


class CIFAR10BW(CIFAR10):

    def __init__(self, root_path, *args, **kwargs):
        super().__init__(root_path)

    def _train_transformation(self, output_shape):
        return T.Compose(
            [
                T.Resize(output_shape),
                T.RandomHorizontalFlip(),
                T.Grayscale(num_output_channels=1),
                T.ToTensor(),

            ]
        )

    def _test_transformation(self, output_shape):
        return T.Compose(
            [
                T.Resize(output_shape),
                T.Grayscale(num_output_channels=1),
                T.ToTensor(),
            ]
        )


class CIFAR100(CIFAR10):

    def __init__(self, root_path, *args, **kwargs):
        super().__init__(root_path)
        self.dataset = torchvision.datasets.CIFAR100


class CIFAR100BW(CIFAR10BW):

    def __init__(self, root_path, *args, **kwargs):
        super().__init__(root_path)
        self.dataset = torchvision.datasets.CIFAR100


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = CIFAR10BW(Path(r"C:/Users/110414/PycharmProjects/OoD_on_SNNs/datasets"))
    loader = DataLoader(
        dataset.load_data(split='test', transformation_option='test', output_shape=(32, 32)),
        batch_size=64,
        shuffle=False
    )
    print(loader.dataset.classes)
    show_img_from_dataloader(loader, img_pos=15, number_of_iterations=10)
    show_grid_from_dataloader(loader)

