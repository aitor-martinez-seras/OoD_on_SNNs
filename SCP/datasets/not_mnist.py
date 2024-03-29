from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
from torchvision.datasets import VisionDataset
from sklearn.utils import shuffle as skl_shuffle

from SCP.datasets.utils import DatasetCustomLoader, download_dataset
from SCP.utils.plots import show_img_from_dataloader, show_grid_from_dataloader


class notMNIST(VisionDataset):
    def __init__(self, root: Path, transform=None, samples_per_class=None):
        """
        Args:
            root (string): Directory with the images of the selected option.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        classes = []
        images = []
        targets = []
        # Every directory is a class in the structure of notMNINST
        for cl_index, class_dir_name in enumerate(sorted(root.iterdir())):
            # Extract the class name from the path
            class_dir_name = class_dir_name.as_posix().split('/')[-1]
            classes.append(class_dir_name)
            class_dir_path = root / class_dir_name
            if samples_per_class is None:
                # We put the limit way above the length of the dataset to not limit
                # the data collected at all, as None indicates that we want all the
                # data available
                limit = 1000000
            else:
                limit = samples_per_class
            for index, png_im in enumerate(sorted(class_dir_path.iterdir())):
                if index >= limit:
                    break
                else:
                    # Some images are corrupted, so we skip those images
                    try:
                        # Get images in range 0-1
                        images.append(torchvision.io.read_image(png_im.as_posix()) / 255)
                        targets.append(cl_index)
                    except RuntimeError:
                        # we have to update the limit to obtain the desired
                        # number of imgs per class
                        limit += 1
                        continue

        # Transform list to tensors
        images = torch.stack(images, dim=0)
        targets = torch.tensor(targets, dtype=torch.uint8)

        # Shuffle fixed for reproducibility reasons
        self.images, self.targets = skl_shuffle(images, targets, random_state=7)
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = [self.images[idx], self.targets[idx]]
        if self.transform:
            sample = self.transform(sample)
        return sample


class notMNISTLoader(DatasetCustomLoader):

    def __init__(self, root_path, *args, **kwargs):
        compressed_fname = 'notMNIST_small.zip'
        url = "https://tecnalia365-my.sharepoint.com/:u:/g/personal/aitor_martinez_tecnalia_com/EXzRbeXSE2tKvYlLdML9mSkBDq7r8GVoy27n70_5HxUi-A?download=1"
        uncomp_fpath = download_dataset(compressed_fname, root_path, url)
        super().__init__(notMNIST, root_path=uncomp_fpath)

    def _train_data(self, transform) -> VisionDataset:
        return self.dataset(
            root=self.root_path,
            samples_per_class=2000,
            transform=None
        )

    def _test_data(self, transform) -> VisionDataset:
        return self.dataset(
            root=self.root_path,
            samples_per_class=2000,
            transform=None
        )

    def _train_transformation(self, output_shape):
        return T.Compose([T.Resize(output_shape), T.ToTensor()])

    def _test_transformation(self, output_shape):
        return self._train_transformation(output_shape)


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = notMNISTLoader(Path(r"C:/Users/110414/PycharmProjects/OoD_on_SNNs/datasets"))
    loader = DataLoader(
        dataset.load_data(split='test', transformation_option='test', output_shape=(64, 64)),
        batch_size=64,
        shuffle=True
    )
    print(loader.dataset.classes)
    print(len(loader.dataset.images))
    print(len(loader.dataset.targets))
    show_img_from_dataloader(loader, img_pos=15, number_of_iterations=5)
    show_grid_from_dataloader(loader)


