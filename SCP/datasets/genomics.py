import json
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
from tfrecord.torch.dataset import MultiTFRecordDataset


# Code copied from https://github.com/selflein/genomics_ood/blob/main/ood_genomics_dataset.py
class OODGenomicsDataset(IterableDataset):
    """PyTorch Dataset implementation for the Bacteria Genomics OOD dataset (https://github.com/google-research/google-research/tree/master/genomics_ood) proposed in

    J. Ren et al., “Likelihood Ratios for Out-of-Distribution Detection,” arXiv:1906.02845 [cs, stat], Available: http://arxiv.org/abs/1906.02845.
    Code copied from https://github.com/selflein/genomics_ood/blob/main/ood_genomics_dataset.py
    """

    splits = {
        "train": "before_2011_in_tr",
        "val": "between_2011-2016_in_val",
        "test": "after_2016_in_test",
        "val_ood": "between_2011-2016_ood_val",
        "test_ood": "after_2016_ood_test",
    }

    def __init__(self, data_root, split="train", transform=None, target_transform=None):
        if isinstance(data_root, str):
            data_root = Path(data_root)
        self.data_root = data_root / "llr_ood_genomics"

        assert split in self.splits, f"Split '{split}' does not exist."
        split_dir = self.data_root / self.splits[split]

        tf_record_ids = [f.stem for f in split_dir.iterdir() if f.suffix == ".tfrecord"]

        self.ds = MultiTFRecordDataset(
            data_pattern=str(split_dir / "{}.tfrecord"),
            index_pattern=str(split_dir / "{}.index"),
            splits={id_: 1 / len(tf_record_ids) for id_ in tf_record_ids},
            description={"x": "byte", "y": "int", "z": "byte"},
            infinite=False,
        )

        with open(self.data_root / "label_dict.json") as f:
            label_dict = json.load(f)
            self.label_dict = {v: k for k, v in label_dict.items()}

        transform = transform if transform is not None else lambda x: x
        target_transform = (
            target_transform if target_transform is not None else lambda x: x
        )
        self.data_transform = lambda x: self.full_transform(
            x, transform, target_transform
        )

    def full_transform(self, item, transform, target_transform):
        dec = np.array([int(i) for i in item["x"].tobytes().decode("utf-8").split(" ")])
        x = torch.from_numpy(transform(dec.copy())).float()
        x = self.spike_transformation(x)
        y = torch.from_numpy(target_transform(item["y"].copy())).long().squeeze()
        return x, y

    def __iter__(self):
        return map(self.data_transform, self.ds.__iter__())

    @staticmethod
    def spike_transformation(x):
        """
        Transform input to spikes
        """
        # Create the array to store the new form of data
        spike_encoded_data = torch.zeros(len(x), 4, dtype=torch.float)
        for i in range(4):
            # Get the positions where each unique input occurs
            # Possible inputs: adenine (A), cytosine (C), guanine (G) and thymine (T).
            # Each represented by a number from 0 to 3
            # positions = []
            pos = np.where(x == i)
            # The position in the sequence where the possible input occurs must have a spike
            # in his category the last position of the tensor
            spike_encoded_data[pos, i] = 1  # TODO: Maybe slow???
        return spike_encoded_data


def custom_collate(batch):
    for d in zip(*batch):
        if len(d[0].shape) > 0:
            data = torch.stack(d, dim=1)
        else:
            targets = torch.stack(d, dim=0)
    return data, targets


def load_oodgenomics(batch_size, datasets_path: Path, test_only=False):
    test_data = OODGenomicsDataset(
        data_root=datasets_path,
        split='val',
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        pin_memory=True,
        collate_fn=custom_collate,
    )

    if test_only is False:
        train_data = OODGenomicsDataset(
            data_root=datasets_path,
            split='train',
        )
        train_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            pin_memory=True,
            collate_fn=custom_collate,
        )
        return train_data, train_loader, test_loader

    else:
        return test_loader


if __name__ == "__main__":
    from sys import platform
    if platform == 'linux':
        ds = Path(r'/home/tri110414/nfs_home/OoD_on_SNNs/datasets')
    else:
        ds = Path(r'C:\Users\110414\PycharmProjects\OoD_on_SNNs\datasets')
    collate_fn = custom_collate
    train_data, train_loader, test_loader = load_oodgenomics(64, ds)
    # custom_loader = CustomDataloader(ds, batch_size=4)
    data, targets = next(iter(train_loader))
    # for data, targets in loader:
    #     print(len(data))
    print(next(iter(train_loader)))
    print(ds.label_dict)
