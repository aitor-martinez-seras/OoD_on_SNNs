from pathlib import Path

import torchvision
import torchvision.transforms as T
from torchvision.datasets import VisionDataset

from SCP.utils.plots import show_img_from_dataloader, show_grid_from_dataloader
from SCP.datasets.utils import DatasetCustomLoader


class Caltech101(DatasetCustomLoader):

    def __init__(self, root_path, *args, **kwargs):
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
                T.ToTensor(),
                T.RandomHorizontalFlip(),
                T.Resize(output_shape),
                self.to_rgb_transform
            ]
        )

    def _test_transformation(self, output_shape):
        return T.Compose(
            [
                T.ToTensor(),
                T.Resize(output_shape),
                self.to_rgb_transform
            ]
        )


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = Caltech101(Path(r"C:/Users/110414/PycharmProjects/OoD_on_SNNs/datasets"))
    loader = DataLoader(
        dataset.load_data(split='test', transformation_option='train', output_shape=(256, 256)),
        batch_size=64,
        shuffle=True
    )
    print(loader.dataset.categories)
    show_img_from_dataloader(loader, img_pos=0, number_of_iterations=10)
    show_grid_from_dataloader(loader)
