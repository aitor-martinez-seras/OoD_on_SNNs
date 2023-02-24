from pathlib import Path

import torchvision
import torchvision.transforms as T
from torchvision.datasets import VisionDataset

from SCP.datasets.utils import DatasetCustomLoader
from SCP.utils.plots import show_img_from_dataloader, show_grid_from_dataloader


class SUN397(DatasetCustomLoader):

    def __init__(self, root_path, *args, **kwargs):
        super().__init__(torchvision.datasets.SUN397, root_path=root_path)

    def _train_data(self, transform) -> VisionDataset:
        return self.dataset(
            root=self.root_path,
            download=True,
            transform=transform,
        )

    def _test_data(self, transform) -> VisionDataset:
        return self.dataset(
            root=self.root_path,
            download=True,
            transform=transform,
        )

    def _train_transformation(self, output_shape):
        return T.Compose(
            [
                # T.Resize(output_shape),
                T.RandomHorizontalFlip(),
                T.RandomResizedCrop(size=output_shape, scale=(0.7, 1.0), ratio=(0.75, 1.0)),
                T.RandomRotation(15),
                T.ToTensor(),
            ]
        )


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = SUN397(Path(r"C:/Users/110414/PycharmProjects/OoD_on_SNNs/datasets"))
    loader = DataLoader(
        dataset.load_data(split='test', transformation_option='train', output_shape=(64, 64)),
        batch_size=64,
        shuffle=False
    )
    print(loader.dataset.classes)
    show_img_from_dataloader(loader, img_pos=15, number_of_iterations=10)
    show_grid_from_dataloader(loader)