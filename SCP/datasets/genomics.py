import json
from pathlib import Path

import torch
from torch.utils.data import IterableDataset
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
        # x = self.spike_transformation(x)
        y = torch.from_numpy(target_transform(item["y"].copy())).long().squeeze()
        return x, y

    def __iter__(self):
        return map(self.data_transform, self.ds.__iter__())

    @staticmethod
    def spike_transformation(x):
        """
        Transform input to spikes
        """

        # Get the positions where each unique input occurs
        # Possible inputs: adenine (A), cytosine (C), guanine (G) and thymine (T).
        # Each represented by a number from 0 to 3
        positions = []
        for i in range(4):
            positions.append(np.where(x == i))

        # Create the array to store the new form of data
        temp_data = torch.zeros(len(x), 1, 4, dtype=torch.int32)

        # The position in the sequence where the possible input occurs must have a spike
        # in his category the last position of the tensor
        for i, pos in enumerate(positions):
            temp_data[pos, 1, i] = 1
        return temp_data


class CustomDataloader:

    def __init__(self, dataset: OODGenomicsDataset, batch_size=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.iterator = iter(dataset)
        self.count = 0

    def __iter__(self):
        data = []
        targets = []
        for _ in range(self.batch_size):
            self.count += 1
            next_data = next(self.iterator)
            data.append(next_data[0])
            targets.append(next_data[1])
        yield torch.stack(data), torch.stack(targets)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from sys import platform
    if platform == 'linux':
        ds = OODGenomicsDataset(r'/home/tri110414/nfs_home/OoD_on_SNNs/datasets', "train")
    else:
        ds = OODGenomicsDataset(r'C:\Users\110414\PycharmProjects\OoD_on_SNNs\datasets', "train")
    loader = DataLoader(ds, batch_size=4, num_workers=0)
    custom_loader = CustomDataloader(ds, batch_size=4)
    # for data, targets in loader:
    #     print(len(data))
    print(next(iter(ds)))
    print(ds.label_dict)
