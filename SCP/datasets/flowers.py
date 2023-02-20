from pathlib import Path
import torchvision
import torchvision.transforms as T
from torchvision.datasets import VisionDataset

from SCP.datasets.utils import DatasetCustomLoader
from SCP.utils.plots import show_img_from_dataloader, show_grid_from_dataloader


class Flowers102(DatasetCustomLoader):

    def __init__(self, root_path, *args, **kwargs):
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        super().__init__(torchvision.datasets.Flowers102, root_path=root_path)

    def _train_data(self, transform) -> VisionDataset:
        return self.dataset(
            root=self.root_path,
            download=True,
            split='test',  # We use this as the test has more data in it
            transform=transform,
        )

    def _test_data(self, transform) -> VisionDataset:
        return self.dataset(
            root=self.root_path,
            download=True,
            split='val',
            transform=transform,
        )

    def _train_transformation(self, output_shape):
        return T.Compose(
            [
                T.ToTensor(),
                T.Resize(output_shape),
                T.RandomRotation(20, ),
                # T.RandomCrop(output_shape[0] - int(output_shape[0]*0.05), padding=int(output_shape[0]*0.05)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
            ]
        )


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = Flowers102(Path(r"C:/Users/110414/PycharmProjects/OoD_on_SNNs/datasets"))
    loader = DataLoader(
        dataset.load_data(split='train', transformation_option='test', output_shape=(64, 64)),
        batch_size=64,
        shuffle=True
    )
    print(loader.dataset)
    print(len(loader.dataset))
    # n = 0
    # for d, t in loader:
    #     n += len(t)
    # print(n)
    show_img_from_dataloader(loader, img_pos=15, number_of_iterations=10)
    show_grid_from_dataloader(loader)
