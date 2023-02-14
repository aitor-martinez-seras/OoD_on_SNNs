from pathlib import Path

import torch
import torchvision
from torchvision.transforms import Lambda

from SCP.datasets.utils import parse_size_of_dataloader
from SCP.utils.plots import show_img_from_dataloader, show_grid_from_dataloader


def load_omniglot(batch_size, datasets_path: Path, *args, **kwargs):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((28, 28)),
            Lambda(lambda img: torchvision.transforms.functional.invert(img)),
        ]
    )
    test_data = torchvision.datasets.Omniglot(
        root=datasets_path,
        background=False,
        download=True,
        transform=transform,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )
    return test_loader


if __name__ == "__main__":
    test_loader = load_omniglot(
        64, Path(r"C:/Users/110414/PycharmProjects/OoD_on_SNNs/datasets"),
    )
    print(len(test_loader.dataset))
    show_img_from_dataloader(test_loader, img_pos=0, number_of_iterations=1)
    show_grid_from_dataloader(test_loader)
