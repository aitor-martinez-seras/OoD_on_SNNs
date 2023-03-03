from pathlib import Path

import torchvision
import torchvision.transforms as T
from torchvision.datasets import VisionDataset

from SCP.datasets.utils import DatasetCustomLoader
from SCP.utils.plots import show_img_from_dataloader, show_grid_from_dataloader


class DTD(DatasetCustomLoader):

    def __init__(self, root_path, *args, **kwargs):
        torchvision.datasets.DTD(root=root_path, download=True)
        self._dtd_images = 'dtd\dtd\images'
        super().__init__(torchvision.datasets.ImageFolder, root_path=root_path / self._dtd_images)

    def _train_data(self, transform) -> VisionDataset:
        return self.dataset(
            root=self.root_path,
            transform=transform,
        )

    def _test_data(self, transform) -> VisionDataset:
        return self._train_data(transform)

    def _train_transformation(self, output_shape):
        return T.Compose(
            [
                T.ToTensor(),
                T.Resize(output_shape),
                T.RandomHorizontalFlip(),
            ]
        )


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = DTD(Path(r"C:/Users/110414/PycharmProjects/OoD_on_SNNs/datasets"))
    loader = DataLoader(
        dataset.load_data(split='train', transformation_option='test', output_shape=(32, 32)),
        batch_size=64,
        shuffle=False
    )
    print(loader.dataset.classes)
    print(len(loader.dataset))
    show_img_from_dataloader(loader, img_pos=15, number_of_iterations=10)
    show_grid_from_dataloader(loader)
