from pathlib import Path

import torchvision
import torchvision.transforms as T
from torchvision.datasets import VisionDataset

from SCP.datasets.utils import DatasetCustomLoader
from SCP.utils.plots import show_img_from_dataloader, show_grid_from_dataloader


class Letters(DatasetCustomLoader):

    def __init__(self, root_path, *args, **kwargs):
        super().__init__(torchvision.datasets.EMNIST, root_path=root_path)

    def _train_data(self, transform) -> VisionDataset:
        data = torchvision.datasets.EMNIST(
            root=self.root_path,
            split="letters",
            train=True,
            download=True,
            transform=transform,
            target_transform=T.Lambda(lambda y: y - 1),
        )

        data.classes = data.classes[1:]
        return data

    def _test_data(self, transform) -> VisionDataset:
        data = torchvision.datasets.EMNIST(
            root=self.root_path,
            split="letters",
            train=False,
            download=True,
            transform=transform,
            target_transform=T.Lambda(lambda y: y - 1),
        )
        data.classes = data.classes[1:]
        return data

    def _train_transformation(self, output_shape):
        return T.Compose(
            [
                lambda img: T.functional.rotate(img, -90),
                lambda img: T.functional.hflip(img),
                T.Resize(output_shape),
                T.ToTensor(),
            ]
        )

    def _test_transformation(self, output_shape):
        return self._train_transformation(output_shape)

    def select_transformation(self, transformation_option, output_shape):
        if transformation_option == 'train':
            return self._train_transformation(output_shape)
        elif transformation_option == 'test':
            return self._test_transformation(output_shape)
        else:
            raise NameError(
                f'Wrong transformation option selected ({transformation_option}). Possible choices: "train" or test")'
            )

    def load_data(self, split, transformation_option, output_shape):

        transform = self.select_transformation(transformation_option, output_shape)

        if split == 'train':
            return self._train_data(transform)

        elif split == 'test':
            return self._test_data(transform)

        else:
            raise NameError(
                f'Wrong split option selected ({split}). Possible choices: "train" or test")'
            )


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = Letters(Path(r"C:/Users/110414/PycharmProjects/OoD_on_SNNs/datasets"))
    loader = DataLoader(
        dataset.load_data(split='train', transformation_option='train', output_shape=(28, 28)),
        batch_size=64,
        shuffle=True
    )
    print(loader.dataset.classes)
    show_img_from_dataloader(loader, img_pos=15, number_of_iterations=10)
    show_grid_from_dataloader(loader)
