from pathlib import Path

import torch
import torchvision
from torchvision.transforms import Lambda

from SCP.datasets.utils import parse_size_of_dataloader


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
    test_loader = parse_size_of_dataloader(test_loader, batch_size)
    return test_loader
