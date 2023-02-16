from pathlib import Path

import torchvision
import torchvision.transforms as T
from torchvision.datasets import VisionDataset

from SCP.datasets.utils import DatasetCustomLoader
from SCP.utils.plots import show_img_from_dataloader, show_grid_from_dataloader


class CelebA(DatasetCustomLoader):

    def __init__(self, root_path):
        super().__init__(torchvision.datasets.CelebA, root_path=root_path)

    def _train_data(self, transform) -> VisionDataset:
        return self.dataset(
            root=self.root_path,
            split='train',
            target_type='identity',
            transform=transform,
            download=True,
        )

    def _test_data(self, transform) -> VisionDataset:
        return self.dataset(
            root=self.root_path,
            split='test',
            target_type='identity',
            transform=transform,
            download=True,
        )

    def _train_transformation(self, output_shape):
        return T.Compose(
            [
                T.Resize(output_shape),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]
        )


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = CelebA(Path(r"C:/Users/110414/PycharmProjects/OoD_on_SNNs/datasets"))
    loader = DataLoader(
        dataset.load_data(split='test', transformation_option='train', output_shape=(96, 96)),
        batch_size=64,
        shuffle=True
    )
    print(loader.dataset.categories)
    show_img_from_dataloader(loader, img_pos=0, number_of_iterations=10)
    show_grid_from_dataloader(loader)
