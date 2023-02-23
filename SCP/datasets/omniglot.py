from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
from torchvision.datasets import VisionDataset

from SCP.datasets.utils import DatasetCustomLoader
from SCP.utils.plots import show_img_from_dataloader, show_grid_from_dataloader


class Omniglot(DatasetCustomLoader):

    def __init__(self, root_path, *args, **kwargs):
        super().__init__(torchvision.datasets.Omniglot, root_path=root_path)
        self.color_transformation = T.Lambda(lambda img: torchvision.transforms.functional.invert(img))

    def _train_data(self, transform) -> VisionDataset:
        return self.dataset(
            root=self.root_path,
            background=False,
            download=True,
            transform=transform,
        )

    def _test_data(self, transform) -> VisionDataset:
        return self.dataset(
            root=self.root_path,
            background=False,
            download=True,
            transform=transform,
        )

    def _train_transformation(self, output_shape):
        return T.Compose(
            [
                T.Resize(output_shape),
                self.color_transformation,
                T.ToTensor(),
                # self.color_transformation
            ]
        )

    def _test_transformation(self, output_shape):
        return self._train_transformation(output_shape)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = Omniglot(Path(r"C:/Users/110414/PycharmProjects/OoD_on_SNNs/datasets"))
    loader = DataLoader(
        dataset.load_data(split='test', transformation_option='test', output_shape=(28, 28)),
        batch_size=64,
        shuffle=False
    )
    d, t = next(iter(loader))
    print(d.mean(), d.std())
    # show_img_from_dataloader(loader, img_pos=15, number_of_iterations=5)
    # show_grid_from_dataloader(loader)

