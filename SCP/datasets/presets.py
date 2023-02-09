from typing import List, Tuple

import torchvision.transforms as T


def load_test_presets(img_shape: List[int], real_shape=None):
    if len(img_shape) != 3:
        raise Exception(f'Resize must be a two element list with integers, got {img_shape} instead')

    color_channels = img_shape[0]
    img_height_and_width = img_shape[1:]

    if color_channels == 3:
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize(img_height_and_width)
            ]
        )

    elif color_channels == 1:
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize(img_height_and_width)
            ]
        )
        if real_shape:
            if real_shape[0] != 1:  # Case where a RGB dataset must be converted to Grayscale
                transform = T.Compose(
                    [
                        T.ToTensor(),
                        T.Grayscale(num_output_channels=1),
                        T.Resize(img_height_and_width)
                    ]
                )
    else:
        raise NameError(f'Wrong number of channels: {color_channels}. Can either be 1 (BW) or 3 (RGB)')

    return transform


if __name__ == "__main__":
    load_test_presets([8, 25, 5])
