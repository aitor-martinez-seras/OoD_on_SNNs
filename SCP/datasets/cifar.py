from pathlib import Path

import tonic
import tonic.transforms as tonic_tfrs
import torchvision
import torchvision.transforms as T
from torchvision.datasets import VisionDataset

from SCP.datasets.utils import DatasetCustomLoader
from SCP.utils.plots import show_img_from_dataloader, show_grid_from_dataloader


class CIFAR10(DatasetCustomLoader):

    def __init__(self, root_path, *args, **kwargs):
        super().__init__(torchvision.datasets.CIFAR10, root_path=root_path)

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
                # T.RandomCrop(output_shape[0], padding=4),
                # T.RandomRotation(15, ),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Resize(output_shape),

            ]
        )
        # return T.Compose(
        #     [
        #         T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
        #         T.ToTensor(),
        #         T.Resize(output_shape),
        #     ]
        # )

    def _test_transformation(self, output_shape):
        return T.Compose(
            [
                T.ToTensor(),
                T.Resize(output_shape),
            ]
        )


class CIFAR10BW(CIFAR10):

    def __init__(self, root_path, *args, **kwargs):
        super().__init__(root_path)

    def _train_transformation(self, output_shape):
        return T.Compose(
            [
                T.Resize(output_shape),
                T.RandomHorizontalFlip(),
                T.Grayscale(num_output_channels=1),
                T.ToTensor(),

            ]
        )

    def _test_transformation(self, output_shape):
        return T.Compose(
            [
                T.Resize(output_shape),
                T.Grayscale(num_output_channels=1),
                T.ToTensor(),
            ]
        )


class CIFAR100(CIFAR10):

    def __init__(self, root_path, *args, **kwargs):
        super().__init__(root_path)
        self.dataset = torchvision.datasets.CIFAR100


class CIFAR100BW(CIFAR10BW):

    def __init__(self, root_path, *args, **kwargs):
        super().__init__(root_path)
        self.dataset = torchvision.datasets.CIFAR100


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    dataset = MNIST_C_Loader(Path(r"C:/Users/110414/PycharmProjects/OoD_on_SNNs/datasets"), option='zigzag')
    # dataset = CIFAR10DVS(Path(r"C:/Users/110414/PycharmProjects/OoD_on_SNNs_Gitlab/datasets"))
    loader = DataLoader(
                dataset=dataset.load_data(split='test', transformation_option='test', output_shape=(34,34,2)),
                batch_size=2,
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

    # dataset = CIFAR10(Path(r"C:/Users/110414/PycharmProjects/OoD_on_SNNs/datasets"))
    # loader = DataLoader(
    #     dataset.load_data(split='test', transformation_option='test', output_shape=(32, 32)),
    #     batch_size=64,
    #     shuffle=False
    # )
    # print(loader.dataset.classes)
    # d, t = next(iter(loader))
    # print(d.mean(), d.std())
    # show_img_from_dataloader(loader, img_pos=15, number_of_iterations=10)
    # show_grid_from_dataloader(loader)

