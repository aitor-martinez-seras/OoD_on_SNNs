from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import VisionDataset

from SCP.utils.datasets.data import download_dataset


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
    load_MNIST_C(64, Path(r'C:\Users\110414\PycharmProjects\OoD_on_SNNs\datasets'))


