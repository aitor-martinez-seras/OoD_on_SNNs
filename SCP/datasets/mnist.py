from pathlib import Path
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torchvision.datasets import VisionDataset
import tonic
import tonic.transforms as tonic_tfrs

from SCP.datasets.utils import DatasetCustomLoader, download_dataset
from SCP.utils.plots import show_img_from_dataloader, show_grid_from_dataloader


def square_creation(input_tensor: torch.Tensor):
    posible_values = [2, 20]
    mean = int(torch.mean(input_tensor[0]) * 100)
    random.seed(mean)
    x_rnd = random.randint(0, 1)
    x_start = posible_values[x_rnd]
    random.seed(mean - 1)
    y_rnd = random.randint(0, 1)
    y_start = posible_values[y_rnd]
    input_tensor[:, x_start:x_start + 6, y_start:y_start + 6] = torch.ones((1, 6, 6), dtype=torch.float32)
    return input_tensor


class MNIST(DatasetCustomLoader):

    def __init__(self, root_path, *args, **kwargs):
        super().__init__(torchvision.datasets.MNIST, root_path=root_path)

    def _train_data(self, transform) -> VisionDataset:
        return self.dataset(
            root=self.root_path,
            train=True,
            download=True,
            transform=transform,
        )

    def _test_data(self, transform) -> VisionDataset:
        return self.dataset(
            root=self.root_path,
            train=False,
            download=True,
            transform=transform,
        )

    def _train_transformation(self, output_shape):
        return T.Compose(
            [
                # T.RandomHorizontalFlip(),
                T.Resize(output_shape),
                T.ToTensor(),
            ]
        )


class MNIST_Square(MNIST):

    def __init__(self, root_path):
        super().__init__(root_path)

    def _train_transformation(self, output_shape):
        return T.Compose(
            [
                # T.RandomHorizontalFlip(),
                T.Resize(output_shape),
                T.ToTensor(),
                T.Lambda(square_creation),
            ]
        )

    def _test_transformation(self, output_shape):
        return T.Compose(
            [
                T.Resize(output_shape),
                T.ToTensor(),
                T.Lambda(square_creation),
            ]
        )


class ArrayImageToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, img):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        return torch.from_numpy(img.transpose((2, 0, 1)) / 255)


class ArrayLabelToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, label):
        return torch.tensor(label)


class MNIST_C(VisionDataset):
    def __init__(self, root, img_transform=None, target_transform=None):
        """
        Args:
            root (string): Directory with the images of the selected option.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(root)
        self.root = Path(root)  # Through the dir is the option selected
        self.images = np.load(self.root / 'test_images.npy')
        self.targets = np.load(self.root / 'test_labels.npy')
        self.img_transform = img_transform
        self.label_transform = target_transform
        self.classes = [str(x) for x in range(10)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        images = self.images[idx]
        targets = self.targets[idx]
        if self.img_transform:
            images = self.img_transform(images)
        if self.label_transform:
            targets = self.label_transform(targets)
        return [images, targets]


class MNIST_C_Loader(DatasetCustomLoader):

    def __init__(self, root_path, option, *args, **kwargs):
        compressed_fname = 'mnist_c.zip'
        url = "https://tecnalia365-my.sharepoint.com/:u:/g/personal/" \
              "aitor_martinez_tecnalia_com/ERi3c4DxluJFqpv4wtlTkKEBvhdrY4WwqNRJWKyyVoTQqg?download=1"
        uncomp_fpath = download_dataset(compressed_fname, root_path, url)
        super().__init__(MNIST_C, root_path=uncomp_fpath / option)

    def _train_data(self, transform) -> VisionDataset:
        return self.dataset(
            root=self.root_path,
            img_transform=transform[0],
            target_transform=transform[1],
        )

    def _test_data(self, transform) -> VisionDataset:
        return self.dataset(
            root=self.root_path,
            img_transform=transform[0],
            target_transform=transform[1],
        )

    def _train_transformation(self, output_shape):
        return [
            T.Compose([ArrayImageToTensor(), T.Resize(output_shape)]),
            T.Compose([ArrayLabelToTensor()])
        ]

    def _test_transformation(self, output_shape):
        return [
            T.Compose([ArrayImageToTensor(), T.Resize(output_shape)]),
            T.Compose([ArrayLabelToTensor()])
        ]


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = MNIST_C_Loader(Path(r"C:/Users/110414/PycharmProjects/OoD_on_SNNs/datasets"), option='zigzag')
    # dataset = NMNIST(Path(r"C:/Users/110414/PycharmProjects/OoD_on_SNNs/datasets"))
    loader = DataLoader(
                dataset=dataset.load_data(split='test', transformation_option='test', output_shape=(34,34)),
                batch_size=6,
                shuffle=False,
                collate_fn=tonic.collation.PadTensors(batch_first=False)
            )
    # loader = DataLoader(
    #     dataset.load_data(split='test', transformation_option='test', output_shape=(28,28)),
    #     batch_size=6,
    #     shuffle=True,
    # )

    # print(loader.dataset.classes)
    # print(len(loader.dataset.images))
    # print(len(loader.dataset.targets))
    import matplotlib.pyplot as plt
    def plot_frames(frames):
        fig, axes = plt.subplots(1, len(frames))
        for axis, frame in zip(axes, frames):
            axis.imshow(frame[1] - frame[0], )
            axis.axis("off")
            # plt.tight_layout()
        plt.show()

    data, targets = next(iter(loader))
    frames = data[100:110, 0]
    plot_frames(frames)
    # show_img_from_dataloader(loader, img_pos=15, number_of_iterations=5)
    # show_grid_from_dataloader(loader)
