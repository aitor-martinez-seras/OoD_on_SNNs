import requests
from zipfile import ZipFile
from io import BytesIO
from pathlib import Path
import shutil
import os


def download_pretrained_weights(pretrained_weights_path: Path):
    print('Downloading pretrained weights...')
    url = "https://tecnalia365-my.sharepoint.com/:u:/g/personal/aitor_martinez_tecnalia_com/Eas7K6U9YJVHvFwcyOJTzwYBQVJYZ9Ibq0Rqq0cG132xKg?download=1"
    r = requests.get(url)
    zipfile = ZipFile(BytesIO(r.content))
    zipfile.extractall(path=pretrained_weights_path)

    for weight_pth in (pretrained_weights_path / "Pretrained_weights").iterdir():
        shutil.move(weight_pth, pretrained_weights_path / weight_pth.name)

    os.rmdir(pretrained_weights_path / "Pretrained_weights")
    print('Downloading completed!')


if __name__ == "__main__":
    cwd = os.getcwd()
    cwd = Path(cwd[:cwd.find("OoD_on_SNNs")+11])
    pret_w_path = cwd / "weights/pretrained"
    download_pretrained_weights(pretrained_weights_path=pret_w_path)
