from abc import ABC, abstractmethod
from pathlib import Path
import requests
from io import BytesIO
from typing import List
from zipfile import ZipFile

import torch
from torchvision.datasets import VisionDataset
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from SCP.utils.common import find_idx_of_class


def _download_dataset(fpath: Path, url: str):
    ftype = fpath.name.split('.')[-1]

    if ftype == 'gz':
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(fpath, 'wb') as f:
                f.write(r.raw.read())
        else:
            raise FileNotFoundError

    elif ftype == 'zip':
        r = requests.get(url)
        zipfile = ZipFile(BytesIO(r.content))
        zipfile.extractall(path=fpath.parent)


def download_dataset(compressed_fname, datasets_folder_path, url) -> Path:
    """
        Downloads the zip or tar.gz file from the url and extracts to the parent folder
        of the file fpath points to, returning the path to the folder of the dataset

        Attributes:
            compressed_fname: name of the file the url points to and which parent folder is the folder where the
                    files should be extracted to.
            datasets_folder_path: path to the folder where dataset should be stored
            url: string with the URL to the compressed file
        Returns:
            uncomp_fpath: the file path as Path object of the uncompressed file, that should be a
                    folder with the name of the dataset
        """
    compressed_fpath = datasets_folder_path / compressed_fname
    uncomp_fname = compressed_fname.split('.')[0]
    uncomp_fpath = datasets_folder_path / uncomp_fname
    if not uncomp_fpath.exists():
        print(f'Dataset {uncomp_fname} will be downloaded')
        _download_dataset(compressed_fpath, url)
    else:
        print(f'Skipping the download of {uncomp_fname} dataset as it is already downloaded')
    return uncomp_fpath


def parse_size_of_dataloader(dataloader_obj: DataLoader, batch_size, max_size=30000):
    if len(dataloader_obj.dataset) > max_size:
        subset = Subset(dataloader_obj.dataset, [x for x in range(max_size)])
        subset_loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
        return subset_loader


def indices_of_every_class_for_subset(train_loader, n_samples_per_class, dataset_name, init_pos=0):
    n_classes = len(train_loader.dataset.classes)
    selected_indices_per_class = []
    for cl_index in range(n_classes):
        # The +1 is because the targets transformation does not happen
        # when accesing the dataset object, only in the dataloader
        if dataset_name == 'Letters':
            cl_index = cl_index + 1

        indices = find_idx_of_class(
            cl_index, train_loader.dataset.targets, number_of_searched_samples=n_samples_per_class, initial_pos=init_pos
        )
        selected_indices_per_class += indices
    return selected_indices_per_class


def isolate_or_remove_mnistc(df, option):
    if option == 'Isolate':
        df = df.loc[
            df['Out-Distribution'].apply(lambda x: x.split('/')[0]) == 'MNIST-C'
            ]

    elif option == 'Remove':
        df = df.loc[
            df['Out-Distribution'].apply(lambda x: x.split('/')[0]) != 'MNIST-C'
            ]
    return df


def isolate_model(df, option):
    df = df.loc[df['Model'] == option]
    if df.size == 0:
        raise ValueError('Empty dataframe, probably wrong model name selected')

    return df


class DatasetCustomLoader(ABC):

    def __init__(self, dataset_class, root_path: Path, *args, **kwargs):
        self.dataset = dataset_class
        self.root_path = root_path

    @abstractmethod
    def _train_data(self, transform) -> VisionDataset:
        """
        To be overridden by the child
        """
        pass

    @abstractmethod
    def _test_data(self, transform) -> VisionDataset:
        """
        To be overridden by the child
        """
        pass

    @abstractmethod
    def _train_transformation(self, output_shape):
        """
        To be overridden by the child
        """
        raise NotImplementedError('Train transformation not implemented by the child')

    def _test_transformation(self, output_shape):
        """
        To be overridden by the child if the dataset needs any custom transformation
        """
        return T.Compose([
            MyBrightnessTransform(brightness_factor=1.5),
            T.ToTensor(),
            T.Resize(output_shape),
        ])

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


def load_dataloader(data, batch_size: int, shuffle: bool, num_workers=0, generator=None):
    if shuffle and generator:
        dataloader = DataLoader(
            dataset=data,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            generator=generator
        )
    else:
        dataloader = DataLoader(
            dataset=data,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
        )
    return dataloader


import torchvision.transforms.functional as TF


class MyBrightnessTransform:
    """Adjust brightness"""

    def __init__(self, brightness_factor):
        self.brightness_factor = brightness_factor

    def __call__(self, x):
        return TF.adjust_brightness(x, brightness_factor=self.brightness_factor)


def create_subset_of_specific_size_with_random_data(data, size_data, new_size, generator, batch_size):
    rnd_idxs = torch.randint(high=size_data, size=(new_size,), generator=generator)
    subset = Subset(data, [x for x in rnd_idxs.numpy()])
    loader = load_dataloader(subset, batch_size=batch_size, shuffle=False)
    return loader
