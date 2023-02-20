from collections import defaultdict
from pathlib import Path
import os

# import imageio
from PIL import Image
import numpy as np
import torchvision
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
import torchvision.transforms as T

from SCP.datasets.utils import download_dataset, DatasetCustomLoader
from SCP.utils.plots import show_img_from_dataloader, show_grid_from_dataloader

dir_structure_help = r"""
TinyImageNetPath
├── test
│   └── images
│       ├── test_0.JPEG
│       ├── t...
│       └── ...
├── train
│   ├── n01443537
│   │   ├── images
│   │   │   ├── n01443537_0.JPEG
│   │   │   ├── n...
│   │   │   └── ...
│   │   └── n01443537_boxes.txt
│   ├── n01629819
│   │   ├── images
│   │   │   ├── n01629819_0.JPEG
│   │   │   ├── n...
│   │   │   └── ...
│   │   └── n01629819_boxes.txt
│   ├── n...
│   │   ├── images
│   │   │   ├── ...
│   │   │   └── ...
├── val
│   ├── images
│   │   ├── val_0.JPEG
│   │   ├── v...
│   │   └── ...
│   └── val_annotations.txt
├── wnids.txt
└── words.txt
"""


def download_and_unzip(URL, root_dir):
    error_message = "Download is not yet implemented. Please, go to {URL} urself."


    raise NotImplementedError(error_message.format(URL))


def _add_channels(img, total_channels=3):
    while len(img.shape) < 3:  # third axis is the channels
        img = np.expand_dims(img, axis=-1)
    while (img.shape[-1]) < 3:
        img = np.concatenate([img, img[:, :, -1:]], axis=-1)
    return img


"""Creates a paths datastructure for the tiny imagenet.
Args:
  root_dir: Where the data is located
  download: Download if the data is not there
Members:
  label_id:
  ids:
  nit_to_words:
  data_dict:
"""


class TinyImageNetPaths:
    def __init__(self, root_dir: Path, download=False):
        dataset_folder_name = 'tiny-imagenet-200'
        if download:
            dataset_root_path = download_dataset(
                compressed_fname=f'{dataset_folder_name}.zip',
                datasets_folder_path=root_dir,
                url='http://cs231n.stanford.edu/tiny-imagenet-200.zip'
            )
        else:
            dataset_root_path = root_dir / dataset_folder_name
            # download_and_unzip('http://cs231n.stanford.edu/tiny-imagenet-200.zip', root_dir)
        train_path = dataset_root_path / 'train'
        val_path = dataset_root_path / 'val'
        test_path = dataset_root_path / 'test'

        wnids_path = dataset_root_path / 'wnids.txt'
        words_path = dataset_root_path / 'words.txt'

        self._make_paths(train_path, val_path, test_path,
                         wnids_path, words_path)

    def _make_paths(self, train_path, val_path, test_path,
                    wnids_path, words_path):
        self.ids = []
        with open(wnids_path, 'r') as idf:
            for nid in idf:
                nid = nid.strip()
                self.ids.append(nid)
        self.nid_to_words = defaultdict(list)
        with open(words_path, 'r') as wf:
            for line in wf:
                nid, labels = line.split('\t')
                labels = list(map(lambda x: x.strip(), labels.split(',')))
                self.nid_to_words[nid].extend(labels)

        self.paths = {
            'train': [],  # [img_path, id, nid, box]
            'val': [],  # [img_path, id, nid, box]
            'test': []  # img_path
        }

        # Get the test paths
        self.paths['test'] = list(map(lambda x: test_path / x,
                                      os.listdir(test_path)))
        # Get the validation paths and labels
        with open(os.path.join(val_path, 'val_annotations.txt')) as valf:
            for line in valf:
                fname, nid, x0, y0, x1, y1 = line.split()
                fname = os.path.join(val_path, 'images', fname)
                bbox = int(x0), int(y0), int(x1), int(y1)
                label_id = self.ids.index(nid)
                self.paths['val'].append((fname, label_id, nid, bbox))

        # Get the training paths
        train_nids = os.listdir(train_path)
        for nid in train_nids:
            anno_path = os.path.join(train_path, nid, nid + '_boxes.txt')
            imgs_path = os.path.join(train_path, nid, 'images')
            label_id = self.ids.index(nid)
            with open(anno_path, 'r') as annof:
                for line in annof:
                    fname, x0, y0, x1, y1 = line.split()
                    fname = os.path.join(imgs_path, fname)
                    bbox = int(x0), int(y0), int(x1), int(y1)
                    self.paths['train'].append((fname, label_id, nid, bbox))


class TinyImageNetDataset(Dataset):
    """Datastructure for the tiny image dataset.
    Args:
      root_dir: Root directory for the data
      mode: One of "train", "test", or "val"
      preload: Preload into memory
      load_transform: Transformation to use at the preload time
      transform: Transformation to use at the retrieval time
      download: Download the dataset
    Members:
      tinp: Instance of the TinyImageNetPaths
      img_data: Image data
      label_data: Label data
    """
    def __init__(self, root_dir: Path, mode='train', preload=False, load_transform=None,
                 transform=None, download=True, max_samples=None):
        tinp = TinyImageNetPaths(root_dir, download)
        self.mode = mode
        self.label_idx = 1  # from [image, id, nid, box]
        self.preload = preload
        self.transform = transform
        self.transform_results = dict()

        self.IMAGE_SHAPE = (64, 64, 3)

        self.img_data = []
        self.label_data = []

        self.max_samples = max_samples
        self.samples = tinp.paths[mode]
        self.samples_num = len(self.samples)

        if self.max_samples is not None:
            self.samples_num = min(self.max_samples, self.samples_num)
            self.samples = np.random.permutation(self.samples)[:self.samples_num]

        if self.preload:
            load_desc = "Preloading {} data...".format(mode)
            self.img_data = np.zeros((self.samples_num,) + self.IMAGE_SHAPE,
                                     dtype=np.float32)
            self.label_data = np.zeros((self.samples_num,), dtype='int')
            for idx in range(self.samples_num):
                s = self.samples[idx]
                # img = imageio.imread(s[0])
                img = np.array(Image.open(s[0]).convert("RGB"))
                img = _add_channels(img)
                self.img_data[idx] = img
                if mode != 'test':
                    self.label_data[idx] = s[self.label_idx]

            if load_transform:
                for lt in load_transform:
                    result = lt(self.img_data, self.label_data)
                    self.img_data, self.label_data = result[:2]
                    if len(result) > 2:
                        self.transform_results.update(result[2])

    def __len__(self):
        return self.samples_num

    def __getitem__(self, idx):
        if self.preload:
            img = self.img_data[idx]
            img = _add_channels(img)
            lbl = None if self.mode == 'test' else self.label_data[idx]
        else:
            s = self.samples[idx]
            # img = imageio.imread(s[0])
            img = np.array(Image.open(s[0]).convert("RGB"))
            lbl = None if self.mode == 'test' else s[self.label_idx]
        # sample = {'img': img, 'label': lbl}

        if self.transform:
            img = self.transform(img)
        return img, lbl


class TinyImageNetLoader(DatasetCustomLoader):

    def __init__(self, root_path, *args, **kwargs):
        super().__init__(TinyImageNetDataset, root_path=root_path)

    def _train_data(self, transform) -> VisionDataset:
        return self.dataset(
            root_dir=self.root_path,
            mode='train',
            download=True,
            transform=transform,
        )

    def _test_data(self, transform) -> VisionDataset:
        return self.dataset(
            root_dir=self.root_path,
            mode='test',
            download=True,
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
            ]
        )


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = TinyImageNetLoader(Path(r"C:/Users/110414/PycharmProjects/OoD_on_SNNs/datasets"))
    loader = DataLoader(
        dataset.load_data(split='train', transformation_option='train', output_shape=(64, 64)),
        batch_size=64,
        shuffle=True
    )
    print(loader.dataset)
    show_img_from_dataloader(loader, img_pos=15, number_of_iterations=10)
    show_grid_from_dataloader(loader)
