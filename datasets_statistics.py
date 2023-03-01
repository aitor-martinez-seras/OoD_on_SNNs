from pathlib import Path
import argparse

from tqdm import tqdm
import pandas as pd
import torch
import numpy as np
from scipy.stats import norm
from scipy.special import kl_div
from matplotlib import pyplot as plt

from SCP.datasets import datasets_loader
from SCP.datasets.utils import load_dataloader
from SCP.utils.common import load_config, get_batch_size


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OOD detection on SNNs", add_help=True)

    parser.add_argument("-c", "--conf", default="config", type=str, required=True,
                        help="name of the configuration in config folder")
    parser.add_argument("--img-shape", type=int, required=True, dest='img_shape',
                        help="the size of the img for the resize (for BW 28, for RGB 32)")
    parser.add_argument("--ind-seed", default=7, type=int, dest='ind_seed',
                        help="seed for the selection of train instances")
    parser.add_argument("--ood-seed", default=8, type=int, dest='ood_seed',
                        help="seed for the selection of test instances")
    return parser


def compute_kb_divergence_per_pixel(images):
    """
    Images of shape (n_images, channels, height, width)
    """
    return


def extract_all_images_to_array(img_loader):
    images = []
    for data, target in img_loader:
        images.append(data.numpy())

    return np.concatenate(images)


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def main(args):

    print(f'Loading configuration from {args.conf}.toml')
    config = load_config(args.conf)

    # Paths
    paths_conf = load_config('paths')
    results_path = Path(paths_conf["paths"]["results"])
    datasets_path = Path(paths_conf["paths"]["datasets"])

    # Datasets config
    datasets_conf = load_config('datasets')

    # Datasets to test
    in_dist_dataset_to_test = config["in_distribution_datasets"]
    ood_datasets_to_test = config["out_of_distribution_datasets"]
    # datasets_to_test = in_dist_dataset_to_test + ood_datasets_to_test

    COLUMNS = ['In-Dataset', 'OoD-Dataset', 'Split', 'Number of OoD examples', 'Mean diff', 'Median diff',
               'InD mean', 'InD Median', 'InD STD', 'OoD mean', 'OoD Median',
               'OoD STD', 'KL Divergence custom metric']
    df_results = pd.DataFrame(columns=COLUMNS)

    for in_dataset in tqdm(in_dist_dataset_to_test, desc=f'In-Distribution datasets loop'):
        results_list = []
        # Get the batch size and data loaders to obtain the data splits
        batch_size = 512
        in_dataset_data_loader = datasets_loader[in_dataset](datasets_path)

        # Load data
        ind_test_data = in_dataset_data_loader.load_data(
            split='test', transformation_option='test', output_shape=(args.img_shape, args.img_shape)
        )

        # Define loaders
        g_test = torch.Generator()
        g_test.manual_seed(args.ind_seed)
        ind_test_loader = load_dataloader(ind_test_data, batch_size, shuffle=True, generator=g_test)

        ind_samples = extract_all_images_to_array(ind_test_loader)

        mean_ind_samples = np.mean(ind_samples, axis=0)
        std_ind_samples = np.std(ind_samples, axis=0)

        x = np.linspace(0, 1, 5000)
        ind_pdf = np.zeros((5000, 3, 32, 32))
        for ch in range(3):
            for h in range(32):
                for w in range(32):
                    ind_pdf[:, ch, h, w] = norm.pdf(x, loc=mean_ind_samples[ch, h, w],
                                                    scale=std_ind_samples[ch, h, w])

        for ood_dataset in tqdm(ood_datasets_to_test, desc=f'Out-of-Distribution datasets loop'):

            # Load OOD data
            ood_dataset_data_loader = datasets_loader[ood_dataset](datasets_path)
            ood_test_data = ood_dataset_data_loader.load_data(
                split='test', transformation_option='test', output_shape=(args.img_shape, args.img_shape)
            )
            g_ood = torch.Generator()
            g_ood.manual_seed(args.ood_seed)
            ood_test_loader = load_dataloader(ood_test_data, batch_size, shuffle=True, generator=g_ood)

            ood_samples = extract_all_images_to_array(ood_test_loader)

            mean_ood_samples = np.mean(ood_samples, axis=0)
            std_ood_samples = np.std(ood_samples, axis=0)

            ood_pdf = np.zeros((5000, 3, 32, 32))
            for ch in range(3):
                for h in range(32):
                    for w in range(32):
                        ood_pdf[:, ch, h, w] = norm.pdf(x, loc=mean_ood_samples[ch, h, w],
                                                        scale=std_ood_samples[ch, h, w])

            kl_all = np.zeros((3, 32, 32))
            for ch in range(3):
                for h in range(32):
                    for w in range(32):
                        kl_all[ch, h, w] = kl_divergence(ind_pdf[:, ch, h, w], ood_pdf[:, ch, h, w])

            # plt.imshow(np.moveaxis(kl_all, 0, -1))  # Move axis 0 to last dim
            # plt.show(block=True)
            #
            # # _ = plt.hist(ind_samples[:, 1, 1, 1], bins='auto', density=False)
            # # _ = plt.hist(ood_samples[:, 1, 1, 1], bins='auto', density=False, c='red')
            # # plt.show(block=True)
            #
            # plt.plot(x, ind_pdf)
            # plt.plot(x, ood_pdf, c='red')
            # plt.show(block=True)

            # ['In-Dataset', 'OoD-Dataset', 'Split', 'Number of OoD examples', 'Mean diff', 'Median diff',
            # 'InD mean', 'InD Median', 'InD STD', 'InD KL Metric' 'OoD mean', 'OoD Median', 'OoD STD', 'OoD KL metric'
            # 'KL Divergence custom metric']

            df_results_one_dataset = pd.DataFrame(results_list, columns=COLUMNS)
            results_list.append(
                    [
                        in_dataset,
                        ood_dataset,
                        'test',
                        len(ood_samples),
                        np.mean(mean_ind_samples) - np.mean(mean_ood_samples),
                        np.median(mean_ind_samples) - np.median(mean_ood_samples),

                        np.mean(mean_ind_samples),
                        np.median(mean_ind_samples),
                        np.std(mean_ind_samples),

                        np.mean(mean_ood_samples),
                        np.median(mean_ood_samples),
                        np.std(mean_ood_samples),

                        np.sum(kl_all),
                    ]
            )

            df_results = pd.concat([df_results, df_results_one_dataset])

    df_results.to_excel(results_path / f'statistics_{args.conf}.xlsx')


if __name__ == "__main__":
    main(get_args_parser().parse_args())
