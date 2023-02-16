from typing import Tuple, List
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
from torchvision.datasets import VisionDataset

from SCP.datasets.presets import load_test_presets, load_presets_test
from SCP.utils.plots import show_img_from_dataloader, show_grid_from_dataloader
from SCP.datasets.utils import DatasetCustomLoader


def load_caltech101(batch_size, datasets_path: Path, test_only=False, image_shape=(3, 96, 96), *args, **kwargs):
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    test_transform = load_test_presets(img_shape=image_shape)
    test_transform = T.Compose(
        [
            test_transform,
            T.Lambda(lambda x: x.repeat(3, 1, 1) if (x.shape[0] == 1) else x),
        ]
    )
    test_data = torchvision.datasets.Caltech101(
        root=datasets_path,
        target_type='category',
        download=True,
        transform=test_transform,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size
    )
    if test_only is False:
        train_transform = T.Compose(
            [
                T.Resize((96, 96)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                # To represent gray images as RGB images
                T.Lambda(lambda x: x.repeat(3, 1, 1) if (x.shape[0] == 1) else x),

            ]
        )
        train_data = torchvision.datasets.Caltech101(
            root=datasets_path,
            target_type='category',
            download=True,
            transform=train_transform,
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True
        )
        return train_data, train_loader, test_loader
    else:
        return test_loader


class Caltech101(DatasetCustomLoader):

    def __init__(self, root_path):
        super().__init__(torchvision.datasets.Caltech101, root_path=root_path)
        self.to_rgb_transform = T.Lambda(lambda x: x.repeat(3, 1, 1) if (x.shape[0] == 1) else x)

    def _train_data(self, transform) -> VisionDataset:
        return self.dataset(
            root=self.root_path,
            target_type='category',
            download=True,
            transform=transform,
        )

    def _test_data(self, transform) -> VisionDataset:
        return self.dataset(
            root=self.root_path,
            target_type='category',
            download=True,
            transform=transform,
        )

    def _train_transformation(self, output_shape):
        return T.Compose(
            [
                T.Resize(output_shape),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                self.to_rgb_transform
            ]
        )

    def _test_transformation(self, output_shape):
        return T.Compose(
            [
                T.Resize(output_shape),
                T.ToTensor(),
                self.to_rgb_transform
            ]
        )


if __name__ == "__main__":

    dataset = Caltech101(Path(r"C:/Users/110414/PycharmProjects/OoD_on_SNNs/datasets"))
    loader = torch.utils.data.DataLoader(
        dataset.load_data(split='test', transformation_option='train', output_shape=(256, 256)),
        batch_size=64,
        shuffle=True
    )
    print(loader.dataset.categories)
    show_img_from_dataloader(loader, img_pos=0, number_of_iterations=10)
    show_grid_from_dataloader(loader)
