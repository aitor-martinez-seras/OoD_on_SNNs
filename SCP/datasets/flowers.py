from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms


# This datasets needs Scipy to load target files form .mat format
def load_flowers(batch_size, datasets_path: Path, test_only=False, *args, **kwargs):
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    transform = transforms.Compose(
        [
            transforms.Resize((500, 500)),
            transforms.RandomRotation(30, ),
            transforms.RandomCrop(400),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ]
    )

    test_data = torchvision.datasets.Flowers102(
        root=datasets_path,
        split='val',
        download=True,
        transform=transform,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size
    )
    if test_only is False:
        train_data = torchvision.datasets.Flowers102(
            root=datasets_path,
            split='train',
            download=True,
            transform=transform,
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
    _, train_loader, test_loader = load_flowers(64, Path(r"C:/Users/110414/PycharmProjects/OoD_on_SNNs/datasets"), test_only=False)
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



