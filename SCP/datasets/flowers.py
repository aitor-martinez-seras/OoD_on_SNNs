from pathlib import Path

from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from SCP.datasets.presets import load_test_presets


# This dataset needs Scipy to load target files form .mat format
def load_flowers(batch_size, datasets_path: Path, test_only=False, image_shape=(3, 32, 32),
                 workers=1, *args, **kwargs):
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    test_transform = load_test_presets(img_shape=image_shape)

    test_data = torchvision.datasets.Flowers102(
        root=datasets_path,
        split='val',
        download=True,
        transform=test_transform,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
    )
    if test_only is False:
        train_transform = transforms.Compose(
            [
                transforms.Resize((500, 500)),
                transforms.RandomRotation(30, ),
                transforms.RandomCrop(400),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
            ]
        )
        train_data = torchvision.datasets.Flowers102(
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
    import matplotlib.pyplot as plt
    # data_loader = load_flowers(64, Path(r"C:/Users/110414/PycharmProjects/OoD_on_SNNs/datasets"), test_only=True)
    dataset, train_loader, test_loader = load_flowers(64, Path(r"C:/Users/110414/PycharmProjects/OoD_on_SNNs/datasets"), test_only=False)
    images, targets = next(iter(train_loader))
    # images, targets = next(iter(test_loader))
    n = 25
    print(images[n].max(), images[n].min())
    # fig, axes = plt.subplots(nrows=8, ncols=8, sharex=True, sharey=True, figsize=(32, 32))
    grid = torchvision.utils.make_grid(images)
    print(targets)
    # plt.imshow(images[n].permute(1, 2, 0))
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()
