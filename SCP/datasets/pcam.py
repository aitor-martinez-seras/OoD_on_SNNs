from pathlib import Path

import torchvision
import torchvision.transforms as T
from torchvision.datasets import VisionDataset

from SCP.datasets.utils import DatasetCustomLoader
from SCP.utils.plots import show_img_from_dataloader, show_grid_from_dataloader


class PCAM(DatasetCustomLoader):

    def __init__(self, root_path, *args, **kwargs):
        super().__init__(torchvision.datasets.PCAM, root_path=root_path)

    def _train_data(self, transform) -> VisionDataset:
        return self.dataset(
            root=self.root_path,
            split='train',
            download=True,
            transform=transform,
        )

    def _test_data(self, transform) -> VisionDataset:
        return self.dataset(
            root=self.root_path,
            split='test',
            download=True,
            transform=transform,
        )

    def _train_transformation(self, output_shape):
        return T.Compose(
            [
                T.ToTensor(),
                T.Resize(output_shape),
                T.RandomHorizontalFlip(),
                T.RandomResizedCrop(size=output_shape, scale=(0.7, 1.0), ratio=(0.75, 1.0)),
                T.RandomRotation(15),

            ]
        )


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = PCAM(Path(r"C:/Users/110414/PycharmProjects/OoD_on_SNNs/datasets"))
    loader = DataLoader(
        dataset.load_data(split='test', transformation_option='test', output_shape=(64, 64)),
        batch_size=64,
        shuffle=True
    )
    print(loader.dataset.classes)
    show_img_from_dataloader(loader, img_pos=15, number_of_iterations=10)
    show_grid_from_dataloader(loader)
#
# from pathlib import Path
#
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
#
# from SCP.datasets.presets import load_test_presets
# from SCP.utils.plots import show_img_from_dataloader, show_grid_from_dataloader
#
#
# def load_pcam(batch_size, datasets_path: Path, test_only=False, image_shape=(3, 96, 96), *args, **kwargs):
#
#     test_transform = load_test_presets(img_shape=image_shape)
#     test_data = torchvision.datasets.PCAM(
#         root=datasets_path,
#         split='test',
#         transform=test_transform,
#         download=True,
#     )
#     test_loader = DataLoader(
#         test_data,
#         batch_size=batch_size,
#     )
#     if test_only is False:
#         train_transform = transforms.Compose(
#             [
#                 transforms.Resize(image_shape[1:]),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomResizedCrop(size=image_shape[1:], scale=(0.7, 1.0), ratio=(0.75, 1.0)),
#                 transforms.RandomRotation(15),
#                 transforms.ToTensor(),
#             ]
#         )
#
#         train_data = torchvision.datasets.PCAM(
#             root=datasets_path,
#             split='train',
#             download=True,
#             transform=train_transform,
#         )
#
#         train_loader = DataLoader(
#             train_data,
#             batch_size=batch_size,
#             shuffle=True,
#             pin_memory=True,
#         )
#         return train_data, train_loader, test_loader
#     else:
#         return test_loader
#
#
# if __name__ == "__main__":
#     dataset, train_loader, test_loader = load_pcam(
#         64, Path(r"C:/Users/110414/PycharmProjects/OoD_on_SNNs/datasets"), test_only=False, image_shape=[3, 96, 96]
#     )
#     show_img_from_dataloader(train_loader, img_pos=0, number_of_iterations=1)
#     show_grid_from_dataloader(train_loader)
