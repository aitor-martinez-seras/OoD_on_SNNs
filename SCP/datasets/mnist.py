import random
from pathlib import Path

import torchvision
import numpy as np
import torch
from torchvision.datasets import VisionDataset

from SCP.datasets.utils import download_dataset


def load_MNIST(batch_size, datasets_path: Path, test_only=False):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )
    test_data_MNIST = torchvision.datasets.MNIST(
        root=datasets_path,
        train=False,
        download=True,
        transform=transform,
    )
    test_loader_MNIST = torch.utils.data.DataLoader(
        test_data_MNIST,
        batch_size=batch_size
    )
    if test_only is False:
        train_data_MNIST = torchvision.datasets.MNIST(
            root=datasets_path,
            train=True,
            download=True,
            transform=transform,
        )
        train_loader_MNIST = torch.utils.data.DataLoader(
            train_data_MNIST,
            batch_size=batch_size,
            shuffle=True
        )
        return train_data_MNIST, train_loader_MNIST, test_loader_MNIST
    else:
        return test_loader_MNIST


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


def load_MNIST_square(batch_size, datasets_path: Path, *args):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(square_creation)
        ]
    )
    test_data_MNIST_square = torchvision.datasets.MNIST(
        root=datasets_path,
        train=False,
        download=True,
        transform=transform,
    )
    test_loader_MNIST_square = torch.utils.data.DataLoader(
        test_data_MNIST_square,
        batch_size=batch_size
    )
    return test_loader_MNIST_square


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1)) / 255
        return [torch.from_numpy(image), torch.tensor(label)]


class MNIST_C(VisionDataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with the images of the selected option.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = Path(root_dir)
        self.images = np.load(self.root / 'test_images.npy')
        self.targets = np.load(self.root / 'test_labels.npy').astype('uint8')
        self.transform = transform
        self.classes = [str(x) for x in range(10)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = [self.images[idx], self.targets[idx]]
        if self.transform:
            sample = self.transform(sample)
        return sample


def load_MNIST_C(batch_size, datasets_path: Path, option='zigzag'):
    compressed_fname = 'mnist_c.zip'
    url = "https://tecnalia365-my.sharepoint.com/:u:/g/personal/aitor_martinez_tecnalia_com/ERi3c4DxluJFqpv4wtlTkKEBvhdrY4WwqNRJWKyyVoTQqg?download=1"
    uncomp_fpath = download_dataset(compressed_fname, datasets_path, url)
    mnist_c_dataloader = torch.utils.data.DataLoader(
        MNIST_C(uncomp_fpath / option, ToTensor()),
        batch_size=batch_size,
        shuffle=False
    )
    return mnist_c_dataloader


if __name__ == '__main__':
    load_MNIST_C(64, Path(r'/datasets'))
