from os.path import join
from pathlib import Path
from zipfile import ZipFile
import subprocess
from urllib.request import Request, urlopen
import shutil

import torchvision
import torchvision.transforms as T
from torchvision.datasets import VisionDataset

from SCP.datasets.utils import DatasetCustomLoader
from SCP.utils.plots import show_img_from_dataloader, show_grid_from_dataloader


def list_categories():
    url = 'http://dl.yf.io/lsun/categories.txt'
    with urlopen(Request(url)) as response:
        return response.read().decode().strip().split('\n')


def download(out_dir, category, set_name):
    url = 'http://dl.yf.io/lsun/scenes/{category}_' \
          '{set_name}_lmdb.zip'.format(**locals())
    if set_name == 'test':
        out_name = 'test_lmdb.zip'
        url = 'http://dl.yf.io/lsun/scenes/test_lmdb.zip'
    else:
        out_name = '{category}_{set_name}_lmdb.zip'.format(**locals())
    out_path = join(out_dir, out_name)
    if Path(out_path).exists():
        print(f'Not downloading {out_path}')
    else:
        cmd = ['curl', '-C', '-', url, '-o', out_path]
        print('Downloading', category, set_name, 'set')
        subprocess.call(cmd)


def download_and_extract(lsun_root_dir_path: Path):

    lsun_root_dir_path.mkdir(exist_ok=True)

    categories = list_categories()
    print('Loading', len(categories), 'categories and downloading them if necessary')
    for category in categories:
        # download(args.out_dir, category, 'train')
        if category == 'test':
            download(str(lsun_root_dir_path), category, 'test')
        else:
            download(str(lsun_root_dir_path), category, 'val')

    for cat in categories:
        if cat == 'test':
            fname = f'test_lmdb.zip'
        else:
            fname = f'{cat}_val_lmdb.zip'

        # Extract in download
        fpath = lsun_root_dir_path / fname
        cat_dir_path = lsun_root_dir_path / fpath.stem

        # New location is in the root of LSUN
        cat_data_new_location = lsun_root_dir_path / fpath.stem
        if not cat_data_new_location.exists():
            zipfile = ZipFile(fpath)
            zipfile.extractall(path=lsun_root_dir_path)


class LSUN(DatasetCustomLoader):

    def __init__(self, root_path, *args, **kwargs):
        self.dataset_dir_name = 'lsun'
        super().__init__(torchvision.datasets.LSUN, root_path=root_path / self.dataset_dir_name)
        download_and_extract(root_path / self.dataset_dir_name)

    def _train_data(self, transform) -> VisionDataset:
        return self.dataset(
            root=self.root_path,
            classes='val',
            transform=transform,
        )

    def _test_data(self, transform) -> VisionDataset:
        return self.dataset(
            root=self.root_path,
            classes='test',
            transform=transform,
        )

    def _train_transformation(self, output_shape):
        return T.Compose(
            [
                # T.Resize(output_shape),
                T.RandomHorizontalFlip(),
                T.RandomResizedCrop(size=output_shape, scale=(0.7, 1.0), ratio=(0.75, 1.0)),
                T.RandomRotation(15),
                T.ToTensor(),
            ]
        )


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = LSUN(Path(r"C:/Users/110414/PycharmProjects/OoD_on_SNNs/datasets"))
    loader = DataLoader(
        dataset.load_data(split='test', transformation_option='train', output_shape=(64, 64)),
        batch_size=64,
        shuffle=False
    )
    print(len(loader.dataset))
    show_img_from_dataloader(loader, img_pos=15, number_of_iterations=10)
    show_grid_from_dataloader(loader)
