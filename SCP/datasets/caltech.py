from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms

from SCP.datasets.presets import load_test_presets


def load_caltech101(batch_size, datasets_path: Path, test_only=False, image_shape=(3, 32, 32), *args, **kwargs):

    test_transform = load_test_presets(img_shape=image_shape)
    test_data = torchvision.datasets.Caltech101(
        root=datasets_path,
        target_type='category',
        download=True,
        transform=test_transform,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size
    )
    if test_only is False:
        train_transform = transforms.Compose(
            [
                # transforms.ToTensor(),
                transforms.Resize((224, 224)),
                # transforms.RandomRotation(30, ),
                # transforms.RandomCrop(400),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # To represent gray images as RGB images
                transforms.Lambda(lambda x: x.repeat(3, 1, 1) if (x.shape[0] == 1) else x),
            ]
        )
        train_data = torchvision.datasets.Caltech101(
            root=datasets_path,
            target_type='category',
            download=True,
            transform=train_transform,
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True
        )
        return train_data, train_loader, test_loader
    else:
        return test_loader


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # data_loader = load_flowers(64, Path(r"C:/Users/110414/PycharmProjects/OoD_on_SNNs/datasets"), test_only=True)
    dataset, train_loader, test_loader = load_caltech101(
        64, Path(r"C:/Users/110414/PycharmProjects/OoD_on_SNNs/datasets"), test_only=False
    )
    images, targets = next(iter(train_loader))
    # images, targets = next(iter(test_loader))
    n = 25
    print(images[n].max(), images[n].min())
    plt.imshow(images[n].permute(1, 2, 0))
    plt.show()
    grid = torchvision.utils.make_grid(images)
    print(targets)
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()
