from pathlib import Path
import argparse

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

from SCP.datasets import datasets_loader
from SCP.datasets.utils import load_dataloader
from SCP.utils.common import load_config, get_batch_size


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OOD detection on SNNs", add_help=True)

    parser.add_argument("--conf", default="config", type=str, help="name of the configuration in config folder")
    parser.add_argument("--train-seed", default=7, type=int, dest='train_seed',
                        help="seed for the selection of train instances")
    parser.add_argument("--test-seed", default=8, type=int, dest='test_seed',
                        help="seed for the selection of test instances")
    return parser


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
    datasets_to_test = in_dist_dataset_to_test + ood_datasets_to_test

    COLUMNS = ['Dataset', 'Split', 'Number of examples', 'Mean', 'Median', 'STD']
    df_results = pd.DataFrame(columns=COLUMNS)

    for in_dataset in tqdm(datasets_to_test, desc=f'In-Distribution dataset loop'):
        results_list = []
        # Get the batch size and data loaders to obtain the data splits
        batch_size = get_batch_size(config, in_dataset)
        in_dataset_data_loader = datasets_loader[in_dataset](datasets_path)

        # Load both splits
        train_data = in_dataset_data_loader.load_data(
            split='train', transformation_option='test', output_shape=datasets_conf[in_dataset]['input_size'][1:]
        )
        test_data = in_dataset_data_loader.load_data(
            split='test', transformation_option='test', output_shape=datasets_conf[in_dataset]['input_size'][1:]
        )

        # Define loaders. Use a seed for train loader
        g_train = torch.Generator()
        g_train.manual_seed(args.train_seed)
        g_test = torch.Generator()
        g_test.manual_seed(args.test_seed)
        train_loader = load_dataloader(train_data, batch_size, shuffle=True, generator=g_train)
        test_loader = load_dataloader(test_data, batch_size, shuffle=True, generator=g_test)

        train_samples = []
        for data, target in train_loader:
            train_samples.append(data.numpy())

        test_samples = []
        for data, target in test_loader:
            test_samples.append(data.numpy())

        train_samples = np.concatenate(train_samples)
        test_samples = np.concatenate(test_samples)

        results_list.append(
            [
                in_dataset,
                'train',
                len(train_samples),
                train_samples.mean(),
                np.median(train_samples),
                train_samples.std()
            ]
        )

        results_list.append(
            [
                in_dataset,
                'test',
                len(test_samples),
                test_samples.mean(),
                np.median(test_samples),
                test_samples.std()
            ]
        )

        df_results_one_dataset = pd.DataFrame(results_list, columns=COLUMNS)

        df_results = pd.concat([df_results, df_results_one_dataset])

    df_results.to_excel(results_path / f'mean_and_standard_{config}.xlsx')


if __name__ == "__main__":
    main(get_args_parser().parse_args())
