import os
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import shutil

import requests


def searchIndicesOfClass(searched_class, labels, n=0, initial_pos=0):
    """
    Function that outputs a list with all the indices of the searched_class in the labelsList
    If n is provided, only the first n coincidences are outputted
    If initial_pos is provided, the search starts by this position.
    searched_class, n, initial_pos -> integer
    labels -> array
    """
    indices = []
    if n == 0:
        # Case of searching for all the array
        for index, labels in enumerate(labels[initial_pos:]):
            if labels == searched_class:
                indices.append(initial_pos + index)
    else:
        # Case of searching only n number of indices
        i = 0
        for index, labels in enumerate(labels[initial_pos:]):
            if i >= n:
                break
            if labels == searched_class:
                indices.append(initial_pos + index)
                i += 1
    return indices


def download_pretrained_weights(pretrained_weights_path: Path, url: str):
    print('Downloading pretrained weights...')
    r = requests.get(url)
    zipfile = ZipFile(BytesIO(r.content))
    zipfile.extractall(path=pretrained_weights_path)

    for weight_pth in (pretrained_weights_path / "Pretrained_weights").iterdir():
        shutil.move(weight_pth, pretrained_weights_path / weight_pth.name)

    os.rmdir(pretrained_weights_path / "Pretrained_weights")
    print('Downloading completed!')


if __name__ == "__main__":
    pret_w_path = Path(r"C:\Users\110414\PycharmProjects\OoD_on_SNNs\weights\pretrained")
    download_pretrained_weights(
        pretrained_weights_path=pret_w_path,
        url="https://tecnalia365-my.sharepoint.com/:u:/g/personal/aitor_martinez_tecnalia_com/Ea2uSuEbePRIklCHnUxhAB0BLUVrQ4IxcnLyu4BnU_i8ag?download=1"
    )
