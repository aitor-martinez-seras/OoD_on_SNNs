import logging
import sys
from pathlib import Path
import datetime

import pytz
from tqdm import tqdm
import tomli
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from SCP.datasets import in_distribution_datasets_loader, out_of_distribution_datasets_loader
from SCP.datasets.utils import indices_of_every_class_for_subset
from SCP.models.model import load_model
from SCP.utils.clusters import create_clusters, average_per_class_and_cluster, distance_to_clusters_averages
from SCP.utils.metrics import thresholds_per_class_for_each_TPR, compute_precision_tpr_fpr_for_test_and_ood, \
    thresholds_for_each_TPR_likelihood, likelihood_method_compute_precision_tpr_fpr_for_test_and_ood
from SCP.benchmark import MSP, ODIN, EnergyOOD
from test import validate_one_epoch


def get_args_parser():
    import argparse

    parser = argparse.ArgumentParser(description="OOD detection on SNNs", add_help=True)

    parser.add_argument("--pretrained", action="store_true", default=False,
                        help="Can only be set if no SNN is used, and in that case the pretrained weights for"
                             "RPN and Detector will be used")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--n-hidden-layers", default=1, type=int,
                        dest="n_hidden_layers", help="number of hidden layers of the models")
    parser.add_argument("--samples-for-cluster-per-class", default=1200, type=int,
                        dest="samples_for_cluster_per_class", help="number of samples for validation per class")
    parser.add_argument("--samples-for-thr-per-class", default=1000, type=int,
                        dest="samples_for_thr_per_class", help="number of samples for validation per class")



def my_custom_logger(logger_name, level=logging.INFO):
    """
    Method to return a custom logger with the given name and level
    """
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(logger_name, mode='w')

    # Set handler levels
    console_handler.setLevel(level)
    file_handler.setLevel(level)

    # Create formatter and assign to handlers
    format_string = "%(asctime)s — %(levelname)s — %(message)s"
    log_format = logging.Formatter(format_string)
    console_handler.setFormatter(log_format)
    file_handler.setFormatter(log_format)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def get_batch_size(config: dict, in_dataset: str, logger: logging.Logger):
    try:  # If the key exists, it means a specific batch size is defined for the dataset
        batch_size = config["hyperparameters"][in_dataset]
    except KeyError:
        batch_size = config["hyperparameters"]["batch_size"]
        logging.warning(f"Using custom batch_size = {batch_size} for {in_dataset}")
    return batch_size


def main(args):

    # -----------------
    # Settings
    # -----------------
    # Load config
    with open(Path(r"config\config.toml"), mode="rb") as fp:
        config = tomli.load(fp)

    # Paths
    results_pth = Path(config["paths"]["results_path"])
    logs_pth = Path(config["paths"]["logs_path"])
    pretrained_weights_path = Path(config["paths"]["pretrained_weights_path"])

    # Datasets to test
    in_dist_dataset_to_test = config["in_distribution_datasets"]
    ood_datasets_to_test = config["out_of_distribution_datasets"]

    # Model architectures
    model_archs = config["model_arch"]
    archs_to_test = [k for k in model_archs.keys()]

    # Dataframe to store the results
    COLUMNS = ['Timestamp', 'In-Distribution', 'Out-Distribution', 'Model',
               'Training accuracy', 'Accuracy in OOD test set', 'OoD Method',
               'AUROC', 'AUPR', 'FPR95', 'FPR80', 'Temperature']
    df_results = pd.DataFrame(columns=COLUMNS)

    # Device for computation
    device = args.device if torch.cuda.is_available() else torch.device('cpu')

    for in_dataset in tqdm(in_dist_dataset_to_test, desc=f'In-Distribution dataset loop'):

        # For each in dataset we reset the results list, previously saving the results
        # in dataframe and in a checkpoint file
        # results_list = []

        # New logger for each In-Distribution Dataset
        logger = my_custom_logger(f"Logger_{in_dataset}.log")

        # Load in-distribution data from the dictionary
        batch_size = get_batch_size(config, in_dataset, logger)
        train_data, train_loader, test_loader = in_distribution_datasets_loader[in_dataset](batch_size)

        logger.info(f'Starting In-Distribution dataset {in_dataset}')
        for model_name in tqdm(archs_to_test, desc='Model loop'):

            logger.info(f'Logs for benchmark with the model {model_name}')

            # Load model arch
            hidden_neurons = model_archs[model_name][in_dataset][0]
            output_neurons = model_archs[model_name][in_dataset][1]
            model = load_model(model_name, device, hidden_neurons, output_neurons, args.n_hidden_layers)

            # Load weights
            weights_path = Path(
                f'state_dict_{in_dataset}_{model_name}_{hidden_neurons}_{output_neurons}_{args.n_hidden_layers} _layers.pth'
            )
            if args.pretrained:
                weights_path = pretrained_weights_path / weights_path
            model.load_state_dict(torch.load(weights_path))

            # Show test accuracy and extract the test logits and spikes
            model.eval()
            test_accuracy, preds_test, logits_test, _spk_count_test = validate_one_epoch(
                model, device, test_loader, return_logits=True,
            )
            logger.info(f"The accuracy of the model with loaded weights of {in_dataset} is {test_accuracy} %")

            # Train subset for creating the clusters
            number_of_samples_per_class = 1200  # TODO: No se a que se debía el hecho de utilizar un subset
            selected_indices_per_class = indices_of_every_class_for_subset(
                train_loader,
                n_samples_per_class=args.samples_for_cluster_per_class,
                dataset_name=in_dataset
            )
            training_subset_clusters = Subset(train_data, [x for x in selected_indices_per_class])
            subset_train_loader_clusters = DataLoader(
                training_subset_clusters, batch_size=batch_size, shuffle=False
            )
            accuracy_subset_train_clusters, preds_train_clusters, _, _spk_count_train_clusters = validate_one_epoch(
                model, device, subset_train_loader_clusters, return_logits=True
            )
            logger.info(f'Accuracy for the train clusters subset is {accuracy_subset_train_clusters:.3f} %')

            # Train subset to create the thresholds
            # Introduce a the init_pos parameters to not select the same indices that for
            # the subset for creating the clusters
            selected_indices_per_class = indices_of_every_class_for_subset(
                train_loader,
                n_samples_per_class=args.samples_for_thr_per_class,
                dataset_name=in_dataset,
                init_pos=number_of_samples_per_class * len(train_loader.dataset.classes)
            )
            training_subset = Subset(train_data, [x for x in selected_indices_per_class])
            subset_train_loader = DataLoader(training_subset, batch_size=batch_size, shuffle=False)

            # Extract the logits and the hidden spikes
            accuracy_subset_train, preds_train, logits_train, _spk_count_train = validate_one_epoch(
                model, device, subset_train_loader, return_logits=True
            )
            logger.info(f'Accuracy for the train subset is {accuracy_subset_train_clusters:.3f} %')

            # Convert spikes to counts
            if isinstance(_spk_count_train_clusters, tuple):
                logger.warning('He usado el ISINSTANCE')
                _spk_count_train_clusters, _ = _spk_count_train_clusters
                _spk_count_train, _ = _spk_count_train
                _spk_count_test, _ = _spk_count_test
            spk_count_train_clusters = np.sum(_spk_count_train_clusters, axis=0, dtype='uint16')
            spk_count_train = np.sum(_spk_count_train, axis=0, dtype='uint16')
            spk_count_test = np.sum(_spk_count_test, axis=0, dtype='uint16')
            logger.info(f'Train subset for clusters: {spk_count_train_clusters.shape}')
            logger.info(f'Train subset for threshold: {spk_count_train.shape}')
            logger.info(f'Test set: {spk_count_test.shape}')

            # Create clusters
            '''
            if hidden_neurons == 200:
              dist_clustering = (1000,2000)
            elif hidden_neurons == 300:
              dist_clustering = (1500,2500)
            '''
            dist_clustering = (500, 5000)

            # TODO: Tengo que conseguir que se use el args.samples_for_cluster_per_class sin que de error.
            #   El problema viene de que se coge el subset de train para hacer los clusters, pero se escoge
            #   antes de hacer el predict, y luego cuando se predice nos quedamos con menos.
            #   Tengo que hacer la funcion create clusters robusta ante sizes mas pequeños, añadiendo un warning
            #   para cuando se ejecute
            clusters_per_class, logging_info = create_clusters(
                subset_train_loader_clusters,
                preds_train_clusters,
                spk_count_train_clusters,
                distance_for_clustering=dist_clustering,
                size=args.samples_for_cluster_per_class,
                verbose=1
            )
            logger.info(logging_info)

            # Initialize for every model, as we save the results for every model
            results_list = []

            for ood_dataset in tqdm(ood_datasets_to_test, desc='Out-of-Distribution dataset loop'):

                logger.info(f'Logs for benchmark with the OoD dataset {ood_dataset}')

                # Load OoD dataset from the dictionary. In case it is MNIST-C, load the
                # selected option
                batch_size_ood = get_batch_size(config, ood_dataset, logger)
                if ood_dataset.split('/')[0] == 'MNIST-C':
                    test_loader_ood = out_of_distribution_datasets_loader[ood_dataset.split('/')[0]](
                        batch_size_ood,
                        test_only=True,
                        option=ood_dataset.split('/')[1]
                    )
                else:
                    test_loader_ood = out_of_distribution_datasets_loader[ood_dataset](batch_size_ood, test_only=True)

                # Extract the spikes and logits for OoD
                accuracy_ood, preds_ood, logits_ood, _spk_count_ood = validate_one_epoch(
                    model, device, test_loader_ood, return_logits=True
                )
                logger.info(f'Accuracy for the ood dataset {ood_dataset} is {accuracy_ood:.3f} %')

                # Convert spikes to counts
                if isinstance(_spk_count_ood, tuple):
                    _spk_count_ood, _ = _spk_count_ood
                spk_count_ood = np.sum(_spk_count_ood, axis=0, dtype='uint16')
                logger.info(f'OoD set: {spk_count_ood.shape}')

                # Create the median aggregations for each cluster of each class
                agg_counts_per_class_cluster = average_per_class_and_cluster(
                    spk_count_train_clusters,
                    preds_train_clusters, clusters_per_class,
                    n_samples=1000, option='median'
                )

                # Computation of the distances of train, test and ood
                distances_train_per_class, _ = distance_to_clusters_averages(
                    spk_count_train, preds_train, agg_counts_per_class_cluster
                )
                distances_test_per_class, _ = distance_to_clusters_averages(
                    spk_count_test, preds_test, agg_counts_per_class_cluster
                )
                distances_ood_per_class, _ = distance_to_clusters_averages(
                    spk_count_ood, preds_ood, agg_counts_per_class_cluster
                )

                # -----------
                # Metrics
                # -----------
                # Creation of the array with the thresholds for each TPR (class, dist_per_TPR)
                distance_thresholds_train = thresholds_per_class_for_each_TPR(distances_train_per_class)
                # Conmputing precision, tpr and fpr
                precision, tpr_values, fpr_values = compute_precision_tpr_fpr_for_test_and_ood(distances_test_per_class,
                                                                                               distances_ood_per_class,
                                                                                               distance_thresholds_train)
                # Appending that when FPR = 1 the TPR is also 1:
                tpr_values_auroc = np.append(tpr_values, 1)
                fpr_values_auroc = np.append(fpr_values, 1)
                # Metrics
                auroc = round(np.trapz(tpr_values_auroc, fpr_values_auroc) * 100, 2)
                aupr = round(np.trapz(precision, tpr_values) * 100, 2)
                fpr95 = round(fpr_values_auroc[95] * 100, 2)
                fpr80 = round(fpr_values_auroc[80] * 100, 2)

                # Save results to list
                local_time = datetime.datetime.now(pytz.timezone('Europe/Madrid')).ctime()
                results_list.append([local_time, in_dataset, ood_dataset, model_name,
                                     test_accuracy, accuracy_ood, 'Ours', auroc, aupr, fpr95, fpr80, 0.0])

                # ------ Other approaches ------
                # TODO: For every approach, move code to benchmark and define in configs the methods to test

                # --- Baseline method ---
                baseline = MSP()
                auroc, aupr, fpr95, fpr80 = baseline(logits_train, logits_test, logits_ood)

                # Save results to list
                local_time = datetime.datetime.now(pytz.timezone('Europe/Madrid')).ctime()
                results_list.append([local_time, in_dataset, ood_dataset, model_name,
                                     test_accuracy, accuracy_ood, 'Baseline', auroc, aupr, fpr95, fpr80, 0.0])

                # --- ODIN ---
                odin = ODIN()
                auroc, aupr, fpr95, fpr80, temp = odin(logits_train, logits_test, logits_ood)
                # Save results to list
                local_time = datetime.datetime.now(pytz.timezone('Europe/Madrid')).ctime()
                results_list.append([local_time, in_dataset, ood_dataset, model_name,
                                     test_accuracy, accuracy_ood, 'ODIN', auroc, aupr, fpr95, fpr80, temp])

                # --- Energy ---
                energy = EnergyOOD()
                auroc, aupr, fpr95, fpr80, temp = energy(logits_train, logits_test, logits_ood)
                # Save results to list
                local_time = datetime.datetime.now(pytz.timezone('Europe/Madrid')).ctime()
                results_list.append([local_time, in_dataset, ood_dataset, model_name,
                                     test_accuracy, accuracy_ood, 'Free energy', auroc, aupr, fpr95, fpr80, temp])

            # Save the results in the results list to a dataframe and the save it to
            # a file
            df_results_one_run = pd.DataFrame(results_list, columns=COLUMNS)
            # Save into .csv format
            # df_results_one_run.to_csv(results_path / f'checkpoint_{in_dataset}.csv')
            df_results = pd.concat([df_results, df_results_one_run])
            # Save into .csv
            # df_results.to_csv(results_path / f'Benchmark_results.csv')

    # Save all the results to excel
    df_results.to_excel('benchmark_results_full.xlsx')


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main()