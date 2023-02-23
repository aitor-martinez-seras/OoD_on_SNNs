from pathlib import Path
import datetime
import argparse

import pytz
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

from SCP.detection.ensembles import EnsembleOdinSCP, EnsembleOdinEnergy
from SCP.detection.weights import download_pretrained_weights
from SCP.datasets import datasets_loader
from SCP.datasets.utils import load_dataloader, create_subset_of_specific_size_with_random_data
from SCP.models.model import load_model
from SCP.utils.clusters import create_clusters, aggregation_per_class_and_cluster, distance_to_clusters_averages,\
    silhouette_score_log
from SCP.utils.common import load_config, get_batch_size, my_custom_logger, create_str_for_ood_method_results, \
    find_idx_of_class
from SCP.detection import MSP, ODIN, EnergyOOD, SCPMethod
from SCP.utils.metrics import compare_distances_per_class_to_distance_thr_per_class, thresholds_per_class_for_each_TPR
from test import validate_one_epoch


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OOD detection on SNNs", add_help=True)

    parser.add_argument("--conf", default="config", type=str, help="name of the configuration in config folder")
    parser.add_argument("--pretrained", action="store_true", default=False,
                        help="Can only be set if no SNN is used, and in that case the pretrained weights for"
                             "RPN and Detector will be used")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--f-max", default=100, type=int, dest='f_max',
                        help="max frecuency of the input neurons per second")
    parser.add_argument("--n-time-steps", default=50, type=int, dest='n_time_steps',
                        help="number of timesteps for the simulation")
    parser.add_argument("--n-hidden-layers", default=1, type=int,
                        dest="n_hidden_layers", help="number of hidden layers of the models")
    parser.add_argument("--samples-for-cluster-per-class", default=1200, type=int,
                        dest="samples_for_cluster_per_class", help="number of samples for validation per class")
    parser.add_argument("--samples-for-thr-per-class", default=1000, type=int,
                        dest="samples_for_thr_per_class", help="number of samples for validation per class")
    parser.add_argument("--max-number-of-test-images", default=10000, type=int,
                        dest="max_number_of_test_images", help="max number of test samples for OOD detection")
    parser.add_argument("--cluster-mode", default="predictions", type=str, dest='cluster_mode',
                        choices=["predictions", "labels", "correct-predictions"],
                        help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--use-test-labels", action='store_true', dest='use_test_labels',
                        help="if passed, the labels used to determine which aggregated clusters to compare to"
                             "are the real labels, not the predictions as in real world scenario")
    parser.add_argument("--use-only-correct-test-images", action='store_true', dest='use_only_correct_test_images',
                        help="if passed, the labels used to determine which aggregated clusters to compare to are"
                             "only correctly predicted images, not all the predictions as in real world scenario")
    parser.add_argument("--samples-for-thr", type=str, dest='samples_for_thr', default='disjoint',
                        choices=['disjoint', 'random', 'same'],
                        help="if passed, the thresholds are defined using a random subset of train")
    parser.add_argument("--save-histograms-for", default=[], type=str, nargs='+', dest="save_histograms_for",
                        help="saves histogram plots for the specified methods. Options: SCP, Baseline, ODIN, Energy")
    parser.add_argument("--save-metric-plots", action='store_true', dest='save_metric_plots',
                        help="if passed, AUROC and AUPR Curves are saved")
    parser.add_argument("--ind-seed", default=6, type=int, dest='ind_seed',
                        help="seed for the In-Distribution dataset")
    parser.add_argument("--thr-seed", default=7, type=int, dest='thr_seed',
                        help="seed for the selection of the instances for creating the thresholds")
    parser.add_argument("--ood-seed", default=8, type=int, dest='ood_seed',
                        help="seed for the selection of ood instances in case train instances are needed")
    parser.add_argument("--fn-vs-bad-clasification", action='store_true', dest='fn_vs_bad_clasification',
                        help="if passed, obtain a new result where info about False Negative is present")
    return parser


def main(args: argparse.Namespace):
    # -----------------
    # Settings
    # -----------------
    # Load config
    print(f'Loading configuration from {args.conf}.toml')
    config = load_config(args.conf)

    # Parse histogram option
    save_scp_hist = save_baseline_hist = save_odin_hist = False
    save_energy_hist = save_ensemble_odin_scp = save_ensemble_odin_energy = False
    args.save_histograms_for = [method.lower() for method in args.save_histograms_for]
    if "scp" in args.save_histograms_for:
        save_scp_hist = True
    if "baseline" in args.save_histograms_for:
        save_baseline_hist = True
    if "odin" in args.save_histograms_for:
        save_odin_hist = True
    if "energy" in args.save_histograms_for:
        save_energy_hist = True
    if "ensemble-odin-scp" in args.save_histograms_for:
        save_ensemble_odin_scp = True
    if "ensemble-odin-energy" in args.save_histograms_for:
        save_ensemble_odin_energy = True

    # Paths
    paths_conf = load_config('paths')
    results_path = Path(paths_conf["paths"]["results"])
    logs_path = Path(paths_conf["paths"]["logs"])
    weights_folder_path = Path(paths_conf["paths"]["weights"])
    pretrained_weights_folder_path = Path(paths_conf["paths"]["pretrained_weights"])
    datasets_path = Path(paths_conf["paths"]["datasets"])
    figures_path = Path(paths_conf["paths"]["figures"])

    # Datasets config
    datasets_conf = load_config('datasets')

    # Datasets to test
    in_dist_dataset_to_test = config["in_distribution_datasets"]
    ood_datasets_to_test = config["out_of_distribution_datasets"]

    # Model architectures
    model_archs = config["model_arch"]
    archs_to_test = [k for k in model_archs.keys()]

    # Check if pretrained weights are downloaded when required
    if args.pretrained:
        exist = False
        for in_dataset in in_dist_dataset_to_test:
            for model_name in archs_to_test:
                hidden_neurons = model_archs[model_name][in_dataset][0]
                output_neurons = model_archs[model_name][in_dataset][1]
                weights_path = pretrained_weights_folder_path / f'state_dict_{in_dataset}_{model_name}' \
                                                                f'_{hidden_neurons}_{output_neurons}_' \
                                                                f'{args.n_hidden_layers}_layers.pth'
                if not weights_path.exists():
                    print(f'As {weights_path} does not exist, pretrained weights will be downloaded')
                    download_pretrained_weights(pretrained_weights_path=pretrained_weights_folder_path)
                    exist = True
                    break
                else:
                    exist = True
        if exist:
            print('Pretrained weights are correctly in path')

    # Dataframes to store results
    COLUMNS = ['Timestamp', 'In-Distribution', 'Out-Distribution', 'Model',
               'Test set accuracy', 'Accuracy in OOD test set', 'OoD Method',
               'AUROC', 'AUPR', 'FPR95', 'FPR80', 'Temperature']
    df_results = pd.DataFrame(columns=COLUMNS)

    if args.fn_vs_bad_clasification:
        columns_fn_incorrect_vs_correct = [
            'In-Dataset', 'OOD-Dataset', 'Total test samples', 'TPR [%]', 'FN [%]', 'FN [Total]',
            'FN correctly classified [%]', 'FN misclassified [%]', 'Accuracy of the model'
        ]
        df_fn_vs_bad_classification = pd.DataFrame(columns=columns_fn_incorrect_vs_correct)

    # Device for computation
    device = args.device if torch.cuda.is_available() else torch.device('cpu')

    # To enable downloading some datasets from pytorch
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    for in_dataset in tqdm(in_dist_dataset_to_test, desc=f'In-Distribution dataset loop'):

        # For each in dataset we reset the results list, previously saving the results
        # in dataframe and in a checkpoint file
        # results_list = []

        # New logger for each In-Distribution Dataset
        logger = my_custom_logger(logger_name=f'{in_dataset}_{args.cluster_mode}', logs_pth=logs_path)
        logger.info(args)

        # ---------------------------------------------------------------
        # Load in-distribution data
        # ---------------------------------------------------------------
        # Get the batch size and data loaders to obtain the data splits
        batch_size = get_batch_size(config, in_dataset, logger)
        in_dataset_data_loader = datasets_loader[in_dataset](datasets_path)

        # Load both splits
        train_data = in_dataset_data_loader.load_data(
            split='train', transformation_option='test', output_shape=datasets_conf[in_dataset]['input_size'][1:]
        )
        test_data = in_dataset_data_loader.load_data(
            split='test', transformation_option='test', output_shape=datasets_conf[in_dataset]['input_size'][1:]
        )

        # Define loaders. Use a seed for train loader
        g_ind = torch.Generator()
        g_ind.manual_seed(args.ind_seed)
        train_loader = load_dataloader(train_data, batch_size, shuffle=True, generator=g_ind)
        test_loader = load_dataloader(test_data, batch_size, shuffle=True, generator=g_ind)

        # Extract useful variables for future operations
        class_names = train_data.classes
        n_classes = len(class_names)

        logger.info(f'Starting In-Distribution dataset {in_dataset}')
        for model_name in tqdm(archs_to_test, desc='Model loop'):
            # Initialize for every model, as we save the results for every model
            results_list = []

            logger.info(f'Logs for benchmark with the model {model_name}')

            # ---------------------------------------------------------------
            # Load model and its weights
            # ---------------------------------------------------------------
            input_size = datasets_conf[in_dataset]['input_size']
            hidden_neurons = model_archs[model_name][in_dataset][0]
            output_neurons = datasets_conf[in_dataset]['classes']
            # TODO: Voy a tener que cambiar la forma de cargarlo ya que si finalmente metemos mas datasets
            #   entonces el caso para reproducir resultados va a ser diferente para color y para no color
            model = load_model(
                model_arch=model_name,
                input_size=input_size,
                hidden_neurons=hidden_neurons,
                output_neurons=output_neurons,
                n_hidden_layers=args.n_hidden_layers,
                f_max=args.f_max,  # Default value is for reproducing results
                encoder='poisson',
                n_time_steps=args.n_time_steps,  # Default value is for reproducing results
            )
            model = model.to(device)
            logger.info('* - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
            logger.info(model)
            logger.info('* - - - - - - - - - - - - - - - - - - - - - - - - - - - -')

            # Load weights
            weights_path = Path(
                f'state_dict_{in_dataset}_{model_name}_{hidden_neurons}'
                f'_{output_neurons}_{args.n_hidden_layers}_layers.pth'
            )
            if args.pretrained:
                weights_path = pretrained_weights_folder_path / weights_path
            else:
                weights_path = weights_folder_path / weights_path

            state_dict = torch.load(weights_path)
            if 'model' in state_dict.keys():  # Handle the case where it has been saved in the updated version
                state_dict = state_dict['model']
            model.load_state_dict(state_dict)

            # ---------------------------------------------------------------
            # Create clusters
            # ---------------------------------------------------------------
            # Process train instances and generate logits and spike counts
            accuracy_train, preds_train, logits_train, _spk_count_train, labels_train = validate_one_epoch(
                model, device, train_loader, return_logits=True, return_targets=True
            )
            logger.info(f'Accuracy for the train clusters subset is {accuracy_train:.3f} %')
            spk_count_train = np.sum(_spk_count_train, axis=0, dtype='uint16')

            # Define cluster mode
            spk_count_train_clusters = spk_count_train
            if args.cluster_mode == "predictions":
                labels_for_clustering = preds_train
            elif args.cluster_mode == "labels":
                labels_for_clustering = labels_train
            elif args.cluster_mode == "correct-predictions":
                correctly_classfied_idx = np.where(preds_train == labels_train)[0]
                labels_for_clustering = preds_train[correctly_classfied_idx]
                spk_count_train_clusters = spk_count_train_clusters[correctly_classfied_idx]
            else:
                raise NameError(f"Wrong cluster mode {args.cluster_mode}")

            logger.info(f"Available train samples' shape: {spk_count_train_clusters.shape}")

            # Create cluster models
            # TODO: Tengo que conseguir que se use el args.samples_for_cluster_per_class sin que de error.
            #   El problema viene de que se coge el subset de train para hacer los clusters, pero se escoge
            #   antes de hacer el predict, y luego cuando se predice nos quedamos con menos.
            #   Tengo que hacer la funcion create clusters robusta ante sizes mas pequeños, añadiendo un warning
            #   para cuando se ejecute
            dist_clustering = (500, 5000)
            file_name = figures_path / f'{in_dataset}_{model_name}_{args.cluster_mode}_{hidden_neurons}' \
                                       f'_{output_neurons}_{args.n_hidden_layers}_layers'
            clusters_per_class, logging_info = create_clusters(
                labels_for_clustering,
                spk_count_train_clusters,
                class_names,
                distance_for_clustering=dist_clustering,
                n_samples_per_class=args.samples_for_cluster_per_class,
                verbose=2,
                name=file_name,
            )
            logger.info(f'Mean number of clusters in total: {np.mean([cl.n_clusters_ for cl in clusters_per_class])}')
            logger.info(logging_info)

            scores_perf = silhouette_score_log(
                clusters_per_class, labels_for_clustering, spk_count_train_clusters, args.samples_for_cluster_per_class,
            )
            logger.info(f'Score per class: {scores_perf}')

            # ---------------------------------------------------------------
            # Select a subset of training to calculate the thresholds
            # ---------------------------------------------------------------
            if args.samples_for_thr == 'disjoint':
                preds_train_thr = preds_train[args.samples_for_cluster_per_class * n_classes:]
                spk_count_train_thr = spk_count_train[args.samples_for_cluster_per_class * n_classes:]
                logits_train_thr = logits_train[args.samples_for_cluster_per_class * n_classes:]

            elif args.samples_for_thr == 'random':
                g_thr = torch.Generator()
                g_thr.manual_seed(args.thr_seed)
                shuffle_idx = torch.randperm(len(train_data), generator=g_thr)
                preds_train_thr = preds_train[shuffle_idx]
                spk_count_train_thr = spk_count_train[shuffle_idx]
                logits_train_thr = logits_train[shuffle_idx]

            elif args.samples_for_thr == 'same':
                preds_train_thr = preds_train
                spk_count_train_thr = spk_count_train
                logits_train_thr = logits_train

            else:
                raise NameError('Bad choice')

            logger.info(f'Train set to select thresholds: {spk_count_train_thr.shape}')

            # ---------------------------------------------------------------
            # Extract predictions, logits and hidden spikes from test InD data
            # ---------------------------------------------------------------
            test_accuracy, preds_test, logits_test, _spk_count_test, test_labels = validate_one_epoch(
                model, device, test_loader, return_logits=True, return_targets=True
            )
            # Option to use the test labels for the metrics (only affects SPC)
            if args.use_test_labels:
                preds_test = test_labels
            logger.info(f"The accuracy of the model with loaded weights of {in_dataset} is {test_accuracy} %")
            spk_count_test = np.sum(_spk_count_test, axis=0, dtype='uint16')
            logger.info(f'Test set: {spk_count_test.shape}')

            # Option to use only the correctly predicted test images for the metrics
            if args.use_only_correct_test_images:
                pos_correct_preds_test = np.where(preds_test == test_labels)[0]
                preds_test = preds_test[pos_correct_preds_test]
                spk_count_test = spk_count_test[pos_correct_preds_test]
                logits_test = logits_test[pos_correct_preds_test]
                new_number_of_samples_for_metrics = len(preds_test)
                logger.info(f'Only using correctly classified test samples... '
                            f'New number of samples for metrics: {new_number_of_samples_for_metrics}')

            len_test_images = len(preds_test)
            if args.max_number_of_test_images < len_test_images:
                preds_test = preds_test[:args.max_number_of_test_images]
                spk_count_test = spk_count_test[:args.max_number_of_test_images]
                logits_test = logits_test[:args.max_number_of_test_images]
                new_number_of_samples_for_metrics = len(preds_test)
                logger.info(f'As there are more test images available ({len_test_images}) than the predefined limit'
                            f' ({args.max_number_of_test_images}), the size of the test images will be decreased. '
                            f'New number of samples for metrics: {new_number_of_samples_for_metrics}')

            # Create the median aggregations (centroids) for each cluster of each class
            print('Spk counts:', spk_count_train_clusters.shape)
            print('Labels:', labels_for_clustering.shape)
            print('Clusters', [len(x.labels_) for x in clusters_per_class])
            agg_counts_per_class_cluster = aggregation_per_class_and_cluster(
                spk_count_train_clusters,
                labels_for_clustering,
                clusters_per_class,
                n_classes,
                n_samples=args.samples_for_cluster_per_class, option='median'
            )

            # Computation of the distances of train
            distances_train_per_class, _ = distance_to_clusters_averages(
                spk_count_train_thr, preds_train_thr, agg_counts_per_class_cluster, n_classes
            )

            # ---------------------------------------------------------------
            # Evaluate OOD performance
            # ---------------------------------------------------------------
            # This flag handles the case where the number of test samples is decreased to match the number
            # of samples in the ood dataset
            number_of_test_samples_decreased = False
            for ood_dataset in tqdm(ood_datasets_to_test, desc='Out-of-Distribution dataset loop'):

                logger.info(f'Logs for benchmark with the OoD dataset {ood_dataset}')

                new_figures_path = figures_path / f'{in_dataset}_vs_{ood_dataset}_{model_name}_{args.cluster_mode}' \
                                                  f'_{hidden_neurons}_{output_neurons}_{args.n_hidden_layers}_layers'

                # In case the number of samples has been decreased, use the backup to reload all the predictions
                # logits and spike counts for the next dataset, as it may not need the test set to be reduced
                # to match its size
                if number_of_test_samples_decreased:
                    logger.info(f'Using the backups to replenish all the test tensors')
                    preds_test = np.copy(backup_preds_test)
                    logits_test = np.copy(backup_logits_test)
                    spk_count_test = np.copy(backup_spk_count_test)

                    # This way, next iteration will only enter this code if again the number of samples
                    # of the test set has been reduced to match the number of OOD samples
                    number_of_test_samples_decreased = False

                    # Free up memory
                    backup_preds_test = None
                    backup_logits_test = None
                    backup_spk_count_test = None

                # ---------------------------------------------------------------
                # Load dataset and extract spikes and logits
                # ---------------------------------------------------------------
                size_test_data = 0
                size_ood_data = 0
                # Load OoD dataset. In case it is MNIST-C, load the selected option
                # In case the OOD test dataset has not enough instances, the train dataset is loaded
                batch_size_ood = get_batch_size(config, ood_dataset, logger)

                if ood_dataset.split('/')[0] == 'MNIST-C':

                    ood_dataset_data_loader = datasets_loader['MNIST-C'](
                        datasets_path, option=ood_dataset.split('/')[1]
                    )

                else:
                    ood_dataset_data_loader = datasets_loader[ood_dataset](datasets_path)

                ood_data = ood_dataset_data_loader.load_data(
                    split='test', transformation_option='test',
                    output_shape=datasets_conf[in_dataset]['input_size'][1:]
                )

                # Define loaders. Use a seed for ood loader
                g_ood = torch.Generator()
                g_ood.manual_seed(8)
                ood_loader = load_dataloader(ood_data, batch_size, shuffle=True, generator=g_ood)

                size_test_data = len(preds_test)
                size_ood_data = len(ood_data)

                # Ensure we have same number of samples for test and ood
                if size_ood_data == size_test_data:
                    pass

                elif size_ood_data < size_test_data:
                    logger.info(f"Using training data as test OOD data for {ood_dataset} dataset")

                    # Load the train data of OOD dataset
                    ood_data = ood_dataset_data_loader.load_data(
                        split='train', transformation_option='test',
                        output_shape=datasets_conf[in_dataset]['input_size'][1:]
                    )

                    size_ood_train_data = len(ood_data)
                    if size_ood_train_data < size_test_data:
                        logger.info(
                            f"There is still not sufficient OOD data in the training set"
                            f" {size_ood_train_data}. Therefore, the size of the test set is going to decrease "
                            f"for {ood_dataset} from {size_test_data} to {size_ood_train_data}")
                        number_of_test_samples_decreased = True
                        backup_preds_test = np.copy(preds_test)
                        backup_logits_test = np.copy(logits_test)
                        backup_spk_count_test = np.copy(spk_count_test)

                        preds_test = preds_test[:size_ood_train_data]
                        logits_test = logits_test[:size_ood_train_data]
                        spk_count_test = spk_count_test[:size_ood_train_data]

                        # Define the new size for the test data for this OOD dataset
                        size_test_data = len(preds_test)

                    # Create the subset of the train OOD data, where it will have the same size as
                    # the size of the test data.
                    ood_loader = create_subset_of_specific_size_with_random_data(
                        data=ood_data, size_data=size_ood_train_data, new_size=size_test_data,
                        generator=g_ood, batch_size=batch_size_ood
                    )

                else:  # size_ood_data > size_test_data
                    logger.info(f"Reducing the number of samples for OOD dataset {ood_dataset} to match "
                                f"the number of samples of test data, equal to {size_test_data}")
                    ood_loader = create_subset_of_specific_size_with_random_data(
                        data=ood_data, size_data=size_ood_data, new_size=size_test_data,
                        generator=g_ood, batch_size=batch_size_ood
                    )

                # Compute distances of test instances after possibly reducing its size
                distances_test_per_class, _ = distance_to_clusters_averages(
                    spk_count_test, preds_test, agg_counts_per_class_cluster, n_classes
                )

                # Extract the spikes and logits for OoD
                accuracy_ood, preds_ood, logits_ood, _spk_count_ood = validate_one_epoch(
                    model, device, ood_loader, return_logits=True
                )
                accuracy_ood = f'{accuracy_ood:.3f}'
                logger.info(f'Accuracy for the ood dataset {ood_dataset} is {accuracy_ood} %')

                # Convert spikes to counts
                if isinstance(_spk_count_ood, tuple):
                    _spk_count_ood, _ = _spk_count_ood
                spk_count_ood = np.sum(_spk_count_ood, axis=0, dtype='uint16')
                logger.info(f'OoD set: {spk_count_ood.shape}')

                # *************** SCP ***************
                # Computation of the distances of ood instances
                distances_ood_per_class, _ = distance_to_clusters_averages(
                    spk_count_ood, preds_ood, agg_counts_per_class_cluster, n_classes
                )
                logger.info(f'Shape of test and ood tensors:')
                logger.info(f'  Spike count test:\t{spk_count_test.shape}')
                logger.info(f'  Spike count ood:\t{spk_count_ood.shape}')
                logger.info(f'  Logits test:\t{logits_test.shape}')
                logger.info(f'  Logits ood:\t{logits_ood.shape}')
                scp = SCPMethod()
                if args.fn_vs_bad_clasification:
                    # Reorder preds and test labels to match the order of in_or_out_distribution_per_tpr_test
                    test_labels_per_predicted_class = []
                    for class_index in range(10):
                        test_labels_per_predicted_class.append(test_labels[find_idx_of_class(class_index, preds_test)])
                    test_labels_reordered = np.concatenate(test_labels_per_predicted_class)
                    preds_test_per_predicted_class = []
                    for class_index in range(10):
                        preds_test_per_predicted_class.append(preds_test[find_idx_of_class(class_index, preds_test)])
                    preds_test_reordered = np.concatenate(preds_test_per_predicted_class)
                    # Compare predictions and labels and output 1 where is correctly predicted, 0 where not
                    correct_incorrect_clasification = np.where(preds_test_reordered == test_labels_reordered, 1, 0)

                    # Obtain array with ind or ood decision for test instances and for specific TPR values
                    # Creation of the array with the thresholds for each TPR (class, dist_per_TPR)
                    dist_thresholds = thresholds_per_class_for_each_TPR(
                        len(class_names), distances_train_per_class,
                    )
                    # Compute if test instances are classified as InD or OoD for every tpr
                    in_or_out_distribution_per_tpr_test = compare_distances_per_class_to_distance_thr_per_class(
                        distances_test_per_class,
                        dist_thresholds
                    )
                    # Extract the list with only the TPR values we are interested in: 25, 50, 75 and 95 per cent
                    tprs_to_extract = (25, 50, 75, 95)
                    in_or_out_distribution_per_tpr_test = in_or_out_distribution_per_tpr_test[tprs_to_extract, :]

                    # Now compare the test labels with the InD or OoD decision and obtain a [4, number_of_samples]
                    # list, where 1 will mean the False Negative was correctly classified and 0 will mean
                    # the False Negative was misclassified
                    fn_correct_vs_incorrect_per_tpr = []
                    for idx, in_or_out_one_tpr in enumerate(in_or_out_distribution_per_tpr_test):
                        fn_position = np.where(in_or_out_one_tpr == 0)[0]
                        fn_correct_vs_incorrect_per_tpr.append(
                            np.where(correct_incorrect_clasification[fn_position] == 1, 1, 0)
                        )

                    df_fn_incorrect_vs_correct_one_dataset = pd.DataFrame(columns=columns_fn_incorrect_vs_correct)
                    for i, fn_correct_vs_incorrect in enumerate(fn_correct_vs_incorrect_per_tpr):
                        df_fn_incorrect_vs_correct_one_dataset.loc[len(df_fn_incorrect_vs_correct_one_dataset)] = [
                            in_dataset,
                            ood_dataset,
                            len(preds_test),
                            tprs_to_extract[i],
                            len(fn_correct_vs_incorrect) / len(preds_test),
                            len(fn_correct_vs_incorrect),
                            len(np.nonzero(fn_correct_vs_incorrect)[0]) / len(fn_correct_vs_incorrect),
                            (len(fn_correct_vs_incorrect) - len(np.nonzero(fn_correct_vs_incorrect)[0])) / len(fn_correct_vs_incorrect),
                            test_accuracy,
                        ]

                    df_fn_vs_bad_classification = pd.concat(
                        [df_fn_vs_bad_classification, df_fn_incorrect_vs_correct_one_dataset]
                    )

                auroc, aupr, fpr95, fpr80 = scp(
                    distances_train_per_class, distances_test_per_class, distances_ood_per_class,
                    save_histogram=save_scp_hist, name=new_figures_path, class_names=class_names, preds_ood=preds_ood
                )

                if args.save_metric_plots:
                    scp.save_auroc_fig(new_figures_path)
                    scp.save_aupr_fig(new_figures_path)

                # Save results to list
                local_time = datetime.datetime.now(pytz.timezone('Europe/Madrid')).ctime()
                results_list.append([local_time, in_dataset, ood_dataset, model_name,
                                     test_accuracy, accuracy_ood, 'Ours', auroc, aupr, fpr95, fpr80, 0.0])
                results_log = create_str_for_ood_method_results('SPC', auroc, aupr, fpr95, fpr80)
                logger.info(results_log)

                # *************** Baseline method ***************
                baseline = MSP()
                auroc, aupr, fpr95, fpr80 = baseline(
                    logits_train_thr, logits_test, logits_ood, save_histogram=save_baseline_hist, name=new_figures_path,
                )
                if args.save_metric_plots:
                    baseline.save_auroc_fig(new_figures_path)
                    baseline.save_aupr_fig(new_figures_path)
                results_log = create_str_for_ood_method_results('Baseline', auroc, aupr, fpr95, fpr80)
                logger.info(results_log)
                # Save results to list
                local_time = datetime.datetime.now(pytz.timezone('Europe/Madrid')).ctime()
                results_list.append([local_time, in_dataset, ood_dataset, model_name,
                                     test_accuracy, accuracy_ood, 'Baseline', auroc, aupr, fpr95, fpr80, 0.0])

                # *************** ODIN ***************
                odin = ODIN()
                auroc, aupr, fpr95, fpr80, temp = odin(
                    logits_train_thr, logits_test, logits_ood, save_histogram=save_odin_hist, name=new_figures_path,
                )
                if args.save_metric_plots:
                    odin.save_auroc_fig(new_figures_path)
                    odin.save_aupr_fig(new_figures_path)
                results_log = create_str_for_ood_method_results('ODIN', auroc, aupr, fpr95, fpr80,temp)
                logger.info(results_log)
                # Save results to list
                local_time = datetime.datetime.now(pytz.timezone('Europe/Madrid')).ctime()
                results_list.append([local_time, in_dataset, ood_dataset, model_name,
                                     test_accuracy, accuracy_ood, 'ODIN', auroc, aupr, fpr95, fpr80, temp])

                # *************** Energy ***************
                energy = EnergyOOD()
                auroc, aupr, fpr95, fpr80, temp = energy(
                    logits_train_thr, logits_test, logits_ood, save_histogram=save_energy_hist, name=new_figures_path,
                )
                if args.save_metric_plots:
                    energy.save_auroc_fig(new_figures_path)
                    energy.save_aupr_fig(new_figures_path)
                results_log = create_str_for_ood_method_results('Energy', auroc, aupr, fpr95, fpr80, temp)
                logger.info(results_log)
                # Save results to list
                local_time = datetime.datetime.now(pytz.timezone('Europe/Madrid')).ctime()
                results_list.append([local_time, in_dataset, ood_dataset, model_name,
                                     test_accuracy, accuracy_ood, 'Free energy', auroc, aupr, fpr95, fpr80, temp])

                # *************** Ensemble ODIN-SCP method ***************
                ensemble_odin_scp = EnsembleOdinSCP()
                auroc, aupr, fpr95, fpr80, temp = ensemble_odin_scp(
                    distances_train_per_class, distances_test_per_class, distances_ood_per_class,
                    logits_train, logits_test, logits_ood,
                    save_histogram=save_ensemble_odin_scp, name=new_figures_path, class_names=class_names
                )
                results_log = create_str_for_ood_method_results('Ensemble-Odin-SCP', auroc, aupr, fpr95, fpr80)
                logger.info(results_log)
                # Save results to list
                local_time = datetime.datetime.now(pytz.timezone('Europe/Madrid')).ctime()
                results_list.append([local_time, in_dataset, ood_dataset, model_name,
                                     test_accuracy, accuracy_ood, 'Ensemble-Odin-SCP', auroc, aupr, fpr95, fpr80, temp])

                # *************** Ensemble ODIN-Energy method ***************
                ensemble_odin_energy = EnsembleOdinEnergy()
                auroc, aupr, fpr95, fpr80, temp = ensemble_odin_energy(
                    logits_train, logits_test, logits_ood,
                    save_histogram=save_ensemble_odin_energy, name=new_figures_path, class_names=class_names
                )
                results_log = create_str_for_ood_method_results('Ensemble-Odin-Energy', auroc, aupr, fpr95, fpr80)
                logger.info(results_log)
                # Save results to list
                local_time = datetime.datetime.now(pytz.timezone('Europe/Madrid')).ctime()
                results_list.append([local_time, in_dataset, ood_dataset, model_name,
                                     test_accuracy, accuracy_ood, 'Ensemble-Odin-SCP', auroc, aupr, fpr95, fpr80, temp])

            # ---------------------------------------------------------------
            # Save results for every model arch
            # ---------------------------------------------------------------
            # Save the results in the results list to a dataframe and the save it to a file
            df_results_one_run = pd.DataFrame(results_list, columns=COLUMNS)
            # Save checkpoint into .csv format
            # df_results_one_run.to_csv(results_path / f'checkpoint_{in_dataset}.csv')
            df_results = pd.concat([df_results, df_results_one_run])

    # Save all the results to excel
    results_filename = f'benchmark_results_{args.conf}_{args.cluster_mode}_fmax_' \
                       f'{args.f_max}_timesteps_{args.n_time_steps}.xlsx'
    df_results.to_excel(results_path / results_filename)

    if args.fn_vs_bad_clasification:
        df_fn_vs_bad_classification.to_excel(results_path / f'fn_vs_bad_classification_{args.conf}_{args.cluster_mode}'
                                                            f'_fmax_{args.f_max}_timesteps_{args.n_time_steps}.xlsx')


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
