from pathlib import Path

import tonic
import tonic.transforms as tonic_tfrs
import torchvision
import torchvision.transforms as T
from torchvision.datasets import VisionDataset

from SCP.datasets.utils import DatasetCustomLoader
from SCP.utils.plots import show_img_from_dataloader, show_grid_from_dataloader


class _NeuromorphicBaseTrainTest(DatasetCustomLoader):

    def __init__(self, root_path, dataset_obj: tonic.Dataset, *args, **kwargs):
        super().__init__(dataset_obj, root_path=root_path)
        self.sensor_size = dataset_obj.sensor_size
        self.neuromorphic_data = True

    def _train_data(self, transform) -> VisionDataset:
        return self.dataset(
            save_to=self.root_path,
            train=True,
            transform=transform,
        )

    def _test_data(self, transform) -> VisionDataset:
        return self.dataset(
            save_to=self.root_path,
            train=False,
            transform=transform,
        )

    def _train_transformation(self, output_shape):
        if output_shape[0] != self.sensor_size[0]:
            trfs = T.Compose(
                [
                    # tonic_tfrs.Denoise(filter_time=10000),
                    tonic.transforms.Downsample(spatial_factor=output_shape[0]/self.sensor_size[0]),
                    tonic_tfrs.ToFrame(sensor_size=(output_shape[0], output_shape[1], 2),
                                       # time_window=1000,
                                       # # n_time_bins=64,
                                       n_event_bins=64,
                                       )
                ]
            )

        else:
            trfs = T.Compose(
                [
                    # tonic_tfrs.Denoise(filter_time=10000),
                    tonic.transforms.Downsample(spatial_factor=output_shape[0]/self.sensor_size[0]),
                    tonic_tfrs.ToFrame(sensor_size=self.sensor_size,
                                       # time_window=1000,
                                       # # n_time_bins=64,
                                       n_event_bins=64,
                                       )
                ]
            )
        return trfs

    def _test_transformation(self, output_shape):
        # In case they are different, we must take the output size as the sensor size
        if output_shape[0] != self.sensor_size[0]:
            trfs = T.Compose(
                [
                    # tonic_tfrs.Denoise(filter_time=10000),
                    tonic.transforms.Downsample(spatial_factor=output_shape[0]/self.sensor_size[0]),
                    tonic_tfrs.ToFrame(sensor_size=(output_shape[0], output_shape[1], 2),
                                       # time_window=1000,
                                       # # n_time_bins=64,
                                       n_event_bins=64,
                                       )
                ]
            )

        else:
            trfs = T.Compose(
                [
                    # tonic_tfrs.Denoise(filter_time=10000),
                    tonic.transforms.Downsample(spatial_factor=output_shape[0]/self.sensor_size[0]),
                    tonic_tfrs.ToFrame(sensor_size=self.sensor_size,
                                       # time_window=1000,
                                       # # n_time_bins=64,
                                       n_event_bins=64,
                                       )
                ]
            )
        return trfs

    def load_data(self, split, transformation_option, output_shape) -> VisionDataset:
        
        transform = self.select_transformation(transformation_option, output_shape)

        if split =='train':
            # CON CACHE DATASET TIENE EL PROBLEMA DE SOLO SE APLICA LA TRANSFORMACION PRIMERA
            # return tonic.DiskCachedDataset(self._train_data(transform), cache_path=self.root_path / 'cache/nmnist/')
            return self._train_data(transform)

        elif split == 'test':
            return self._test_data(transform)

        else:
            raise NameError(
                f'Wrong split option selected ({split}). Possible choices: "train" or test")'
            )


class _NeuromorphicBaseNoSplits(DatasetCustomLoader):

    def __init__(self, root_path, dataset_obj: tonic.Dataset, *args, **kwargs):
        super().__init__(dataset_obj, root_path=root_path)
        self.sensor_size = dataset_obj.sensor_size
        self.neuromorphic_data = True

    def _train_data(self, transform) -> VisionDataset:
        return self.dataset(
            save_to=self.root_path,
            transform=transform,
        )

    def _test_data(self, transform) -> VisionDataset:
        return self.dataset(
            save_to=self.root_path,
            transform=transform,
        )

    def _train_transformation(self, output_shape):
        if output_shape[0] != self.sensor_size[0]:
            trfs = T.Compose(
                [
                    # tonic_tfrs.Denoise(filter_time=10000),
                    tonic.transforms.Downsample(spatial_factor=output_shape[0]/self.sensor_size[0]),
                    tonic_tfrs.ToFrame(sensor_size=(output_shape[0], output_shape[1], 2),
                                       # time_window=1000,
                                       # # n_time_bins=64,
                                       n_event_bins=64,
                                       )
                ]
            )

        else:
            trfs = T.Compose(
                [
                    # tonic_tfrs.Denoise(filter_time=10000),
                    tonic.transforms.Downsample(spatial_factor=output_shape[0]/self.sensor_size[0]),
                    tonic_tfrs.ToFrame(sensor_size=self.sensor_size,
                                       # time_window=1000,
                                       # # n_time_bins=64,
                                       n_event_bins=64,
                                       )
                ]
            )
        return trfs

    def _test_transformation(self, output_shape):
        # In case they are different, we must take the output size as the sensor size
        if output_shape[0] != self.sensor_size[0]:
            trfs = T.Compose(
                [
                    # tonic_tfrs.Denoise(filter_time=10000),
                    tonic.transforms.Downsample(spatial_factor=output_shape[0]/self.sensor_size[0]),
                    tonic_tfrs.ToFrame(sensor_size=(output_shape[0], output_shape[1], 2),
                                       # time_window=1000,
                                       # # n_time_bins=64,
                                       n_event_bins=64,
                                       )
                ]
            )

        else:
            trfs = T.Compose(
                [
                    # tonic_tfrs.Denoise(filter_time=10000),
                    tonic.transforms.Downsample(spatial_factor=output_shape[0]/self.sensor_size[0]),
                    tonic_tfrs.ToFrame(sensor_size=self.sensor_size,
                                       # time_window=1000,
                                       # # n_time_bins=64,
                                       n_event_bins=64,
                                       )
                ]
            )
        return trfs

    def load_data(self, split, transformation_option, output_shape) -> VisionDataset:
        
        transform = self.select_transformation(transformation_option, output_shape)

        if split =='train':
            # CON CACHE DATASET TIENE EL PROBLEMA DE SOLO SE APLICA LA TRANSFORMACION PRIMERA
            # return tonic.DiskCachedDataset(self._train_data(transform), cache_path=self.root_path / 'cache/nmnist/')
            return self._train_data(transform)

        elif split == 'test':
            return self._test_data(transform)

        else:
            raise NameError(
                f'Wrong split option selected ({split}). Possible choices: "train" or test")'
            )


class CIFAR10DVS(_NeuromorphicBaseNoSplits):

    def __init__(self, root_path, *args, **kwargs):
        super().__init__(root_path, tonic.datasets.CIFAR10DVS, *args, **kwargs)


class NCALTECH101(_NeuromorphicBaseNoSplits):

    def __init__(self, root_path, *args, **kwargs):
        super().__init__(root_path, tonic.datasets.NCALTECH101, *args, **kwargs)
        self.sensor_size = (2, 128, 128)


class NMNIST(_NeuromorphicBaseTrainTest):

    def __init__(self, root_path, *args, **kwargs):
        super().__init__(root_path, tonic.datasets.NMNIST, *args, **kwargs)


class DVSGesture(_NeuromorphicBaseTrainTest):

    def __init__(self, root_path, *args, **kwargs):
        super().__init__(root_path, tonic.datasets.DVSGesture, *args, **kwargs)


class POKERDVS(_NeuromorphicBaseTrainTest):

    def __init__(self, root_path, *args, **kwargs):
        super().__init__(root_path, tonic.datasets.POKERDVS, *args, **kwargs)


class DVSLip(_NeuromorphicBaseTrainTest):

    def __init__(self, root_path, *args, **kwargs):
        super().__init__(root_path, tonic.datasets.DVSLip, *args, **kwargs)


# class NMNIST(DatasetCustomLoader):

#     def __init__(self, root_path, *args, **kwargs):
#         super().__init__(tonic.datasets.NMNIST, root_path=root_path)
#         self.sensor_size = tonic.datasets.NMNIST.sensor_size
#         self.neuromorphic_data = True

#     def _train_data(self, transform) -> VisionDataset:
#         return self.dataset(
#             save_to=self.root_path,
#             train=True,
#             transform=transform,
#         )

#     def _test_data(self, transform) -> VisionDataset:
#         return self.dataset(
#             save_to=self.root_path,
#             train=False,
#             transform=transform,
#         )

#     def _train_transformation(self, output_shape):
#         if output_shape[0] != self.sensor_size[0]:
#             trfs = T.Compose(
#                 [
#                     # tonic_tfrs.Denoise(filter_time=10000),
#                     tonic.transforms.Downsample(spatial_factor=output_shape[0]/self.sensor_size[0]),
#                     tonic_tfrs.ToFrame(sensor_size=(output_shape[0], output_shape[1], 2),
#                                        # time_window=1000,
#                                        # # n_time_bins=64,
#                                        n_event_bins=64,
#                                        )
#                 ]
#             )

#         else:
#             trfs = T.Compose(
#                 [
#                     # tonic_tfrs.Denoise(filter_time=10000),
#                     tonic.transforms.Downsample(spatial_factor=output_shape[0]/self.sensor_size[0]),
#                     tonic_tfrs.ToFrame(sensor_size=self.sensor_size,
#                                        # time_window=1000,
#                                        # # n_time_bins=64,
#                                        n_event_bins=64,
#                                        )
#                 ]
#             )
#         return trfs

#     def _test_transformation(self, output_shape):
#         # In case they are different, we must take the output size as the sensor size
#         if output_shape[0] != self.sensor_size[0]:
#             trfs = T.Compose(
#                 [
#                     # tonic_tfrs.Denoise(filter_time=10000),
#                     tonic.transforms.Downsample(spatial_factor=output_shape[0]/self.sensor_size[0]),
#                     tonic_tfrs.ToFrame(sensor_size=(output_shape[0], output_shape[1], 2),
#                                        # time_window=1000,
#                                        # # n_time_bins=64,
#                                        n_event_bins=64,
#                                        )
#                 ]
#             )

#         else:
#             trfs = T.Compose(
#                 [
#                     # tonic_tfrs.Denoise(filter_time=10000),
#                     tonic.transforms.Downsample(spatial_factor=output_shape[0]/self.sensor_size[0]),
#                     tonic_tfrs.ToFrame(sensor_size=self.sensor_size,
#                                        # time_window=1000,
#                                        # # n_time_bins=64,
#                                        n_event_bins=64,
#                                        )
#                 ]
#             )
#         return trfs

#     def load_data(self, split, transformation_option, output_shape) -> VisionDataset:

#         transform = self.select_transformation(transformation_option, output_shape)

#         if split =='train':
#             # CON CACHE DATASET TIENE EL PROBLEMA DE SOLO SE APLICA LA TRANSFORMACION PRIMERA
#             # return tonic.DiskCachedDataset(self._train_data(transform), cache_path=self.root_path / 'cache/nmnist/')
#             return self._train_data(transform)

#         elif split == 'test':
#             return self._test_data(transform)

#         else:
#             raise NameError(
#                 f'Wrong split option selected ({split}). Possible choices: "train" or test")'
#             )
        

class NMNISTMissingEvents(DatasetCustomLoader):

    def __init__(self, root_path, *args, **kwargs):
        super().__init__(tonic.datasets.NMNIST, root_path=root_path)
        self.sensor_size = tonic.datasets.NMNIST.sensor_size
        self.neuromorphic_data = True

    def _train_data(self, transform) -> VisionDataset:
        return self.dataset(
            save_to=self.root_path,
            train=True,
            transform=transform,
        )

    def _test_data(self, transform) -> VisionDataset:
        return self.dataset(
            save_to=self.root_path,
            train=False,
            transform=transform,
        )

    def _train_transformation(self, output_shape):
        if output_shape[0] != self.sensor_size[0]:
            trfs = T.Compose(
                [
                    # tonic_tfrs.Denoise(filter_time=10000),
                    tonic.transforms.Downsample(spatial_factor=output_shape[0]/self.sensor_size[0]),
                    tonic.transforms.DropEventByArea(sensor_size=(output_shape[0], output_shape[1], 2), area_ratio=0.4),
                    tonic_tfrs.ToFrame(sensor_size=(output_shape[0], output_shape[1], 2),
                                       # time_window=1000,
                                       # # n_time_bins=64,
                                       n_event_bins=64,
                                       )
                ]
            )

        else:
            trfs = T.Compose(
                [
                    # tonic_tfrs.Denoise(filter_time=10000),
                    tonic.transforms.Downsample(spatial_factor=output_shape[0]/self.sensor_size[0]),
                    tonic.transforms.DropEventByArea(sensor_size=self.sensor_size, area_ratio=0.4),
                    tonic_tfrs.ToFrame(sensor_size=self.sensor_size,
                                       # time_window=1000,
                                       # # n_time_bins=64,
                                       n_event_bins=64,
                                       )
                ]
            )
        return trfs

    def _test_transformation(self, output_shape):
        # In case they are different, we must take the output size as the sensor size
        if output_shape[0] != self.sensor_size[0]:
            trfs = T.Compose(
                [
                    # tonic_tfrs.Denoise(filter_time=10000),
                    tonic.transforms.Downsample(spatial_factor=output_shape[0]/self.sensor_size[0]),
                    tonic.transforms.DropEventByArea(sensor_size=(output_shape[0], output_shape[1], 2), area_ratio=0.4),
                    tonic_tfrs.ToFrame(sensor_size=(output_shape[0], output_shape[1], 2),
                                       # time_window=1000,
                                       # # n_time_bins=64,
                                       n_event_bins=64,
                                       )
                ]
            )

        else:
            trfs = T.Compose(
                [
                    # tonic_tfrs.Denoise(filter_time=10000),
                    tonic.transforms.Downsample(spatial_factor=output_shape[0]/self.sensor_size[0]),
                    tonic.transforms.DropEventByArea(sensor_size=self.sensor_size, area_ratio=0.4),
                    tonic_tfrs.ToFrame(sensor_size=self.sensor_size,
                                       # time_window=1000,
                                       # # n_time_bins=64,
                                       n_event_bins=64,
                                       )
                ]
            )
        return trfs

    def load_data(self, split, transformation_option, output_shape) -> VisionDataset:

        transform = self.select_transformation(transformation_option, output_shape)

        if split =='train':
            # CON CACHE DATASET TIENE EL PROBLEMA DE SOLO SE APLICA LA TRANSFORMACION PRIMERA
            # return tonic.DiskCachedDataset(self._train_data(transform), cache_path=self.root_path / 'cache/nmnist/')
            return self._train_data(transform)

        elif split == 'test':
            return self._test_data(transform)

        else:
            raise NameError(
                f'Wrong split option selected ({split}). Possible choices: "train" or test")'
            )
        

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    # dataset = MNIST_C_Loader(Path(r"C:/Users/110414/PycharmProjects/OoD_on_SNNs/datasets"), option='zigzag')
    # dataset = DVSGesture(Path(r"C:/Users/110414/PycharmProjects/OoD_on_SNNs_Gitlab/datasets"))
    dataset = POKERDVS(Path(r"C:/Users/110414/PycharmProjects/OoD_on_SNNs_Gitlab/datasets"))
    # dataset = NCALTECH(Path(r"datasets"))
    loader = DataLoader(
                dataset=dataset.load_data(split='test', transformation_option='test', output_shape=(34,34,2)),
                batch_size=2,
                shuffle=False,
                collate_fn=tonic.collation.PadTensors(batch_first=False)
            )
    # loader = DataLoader(
    #     dataset.load_data(split='test', transformation_option='test', output_shape=(28,28)),
    #     batch_size=6,
    #     shuffle=True,
    # )

    # print(loader.dataset.classes)
    # print(len(loader.dataset.images))
    # print(len(loader.dataset.targets))
    import matplotlib.pyplot as plt
    def plot_frames(frames):
        fig, axes = plt.subplots(1, len(frames))
        for axis, frame in zip(axes, frames):
            axis.imshow(frame[1] - frame[0], )
            axis.axis("off")
            # plt.tight_layout()
        plt.show()

    data, targets = next(iter(loader))
    frames = data[100:110, 0]
    plot_frames(frames)
    # show_img_from_dataloader(loader, img_pos=15, number_of_iterations=5)
    # show_grid_from_dataloader(loader)

    # dataset = CIFAR10(Path(r"C:/Users/110414/PycharmProjects/OoD_on_SNNs/datasets"))
    # loader = DataLoader(
    #     dataset.load_data(split='test', transformation_option='test', output_shape=(32, 32)),
    #     batch_size=64,
    #     shuffle=False
    # )
    # print(loader.dataset.classes)
    # d, t = next(iter(loader))
    # print(d.mean(), d.std())
    # show_img_from_dataloader(loader, img_pos=15, number_of_iterations=10)
    # show_grid_from_dataloader(loader)

