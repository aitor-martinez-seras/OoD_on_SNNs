from pathlib import Path

import torchvision
import torchvision.transforms as T
from torchvision.datasets import VisionDataset

from SCP.datasets.utils import DatasetCustomLoader
from SCP.utils.plots import show_img_from_dataloader, show_grid_from_dataloader


class GTSRB(DatasetCustomLoader):

    def __init__(self, root_path, *args, **kwargs):
        super().__init__(torchvision.datasets.GTSRB, root_path=root_path)

    def _train_data(self, transform) -> VisionDataset:
        return self.dataset(
            root=self.root_path,
            split='train',
            download=True,
            transform=transform,
        )

    def _test_data(self, transform) -> VisionDataset:
        return self.dataset(
            root=self.root_path,
            split='test',
            download=True,
            transform=transform,
        )

    def _train_transformation(self, output_shape):
        return T.Compose(
            [
                T.ToTensor(),
                T.Resize(output_shape),
                T.RandomHorizontalFlip(),
                # T.RandomVerticalFlip(),
                # T.RandomResizedCrop(size=image_shape[1:], scale=(0.7, 1.0), ratio=(0.75, 1.0)),
                # T.RandomRotation(15),

            ]
        )


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = GTSRB(Path(r"C:/Users/110414/PycharmProjects/OoD_on_SNNs/datasets"))
    loader = DataLoader(
        dataset.load_data(split='train', transformation_option='train', output_shape=(32, 32)),
        batch_size=64,
        shuffle=True
    )
    print(loader.dataset)
    show_img_from_dataloader(loader, img_pos=15, number_of_iterations=10)
    show_grid_from_dataloader(loader)

