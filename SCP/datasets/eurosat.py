from pathlib import Path
import torchvision
import torchvision.transforms as T
from torchvision.datasets import VisionDataset

from SCP.datasets.utils import DatasetCustomLoader
from SCP.utils.plots import show_img_from_dataloader, show_grid_from_dataloader


class EuroSAT(DatasetCustomLoader):

    def __init__(self, root_path, *args, **kwargs):
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        super().__init__(torchvision.datasets.EuroSAT, root_path=root_path)

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
                T.ToTensor(),
                T.Resize(output_shape),
                # transforms.RandomResizedCrop(size=image_shape[1:], scale=(0.7, 1.0), ratio=(0.75, 1.0)),
                T.RandomHorizontalFlip(),
                # transforms.RandomRotation(15),
            ]
        )


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = EuroSAT(Path(r"C:/Users/110414/PycharmProjects/OoD_on_SNNs/datasets"))
    loader = DataLoader(
        dataset.load_data(split='test', transformation_option='test', output_shape=(512, 512)),
        batch_size=64,
        shuffle=True
    )
    print(loader.dataset.classes)
    show_img_from_dataloader(loader, img_pos=15, number_of_iterations=10)
    show_grid_from_dataloader(loader)
