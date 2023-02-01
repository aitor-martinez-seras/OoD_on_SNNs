from pathlib import Path

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from SCP.datasets.presets import load_test_presets
from SCP.utils.plots import show_img_from_dataloader, show_grid_from_dataloader


def load_FGVCAircraft(batch_size, datasets_path: Path, test_only=False, image_shape=(3, 32, 32),
                      workers=4, *args, **kwargs):

    test_transform = load_test_presets(img_shape=image_shape)
    test_data = torchvision.datasets.FGVCAircraft(
        root=datasets_path,
        split='test',
        transform=test_transform,
        download=True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=workers,
    )
    if test_only is False:
        train_transform = transforms.Compose(
            [
                # transforms.ToTensor(),
                transforms.Resize((112, 224)),
                # transforms.RandomRotation(30, ),
                # transforms.RandomCrop(400),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # To represent gray images as RGB images
                # transforms.Lambda(lambda x: x.repeat(3, 1, 1) if (x.shape[0] == 1) else x),
            ]
        )
        train_data = torchvision.datasets.FGVCAircraft(
            root=datasets_path,
            split='train',
            download=True,
            transform=train_transform,
        )

        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )
        return train_data, train_loader, test_loader
    else:
        return test_loader


if __name__ == "__main__":
    dataset, _, test_loader = load_FGVCAircraft(
        64, Path(r"C:/Users/110414/PycharmProjects/OoD_on_SNNs/datasets"), test_only=False
    )
    test_transform = load_test_presets(img_shape=[3, 32, 32])
    dataset.transform = test_transform

    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    show_img_from_dataloader(train_loader, img_pos=0, number_of_iterations=1)
    show_grid_from_dataloader(train_loader)
