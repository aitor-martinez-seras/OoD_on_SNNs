from pathlib import Path
import datetime
import argparse

import pytz
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose, Lambda
import torchvision.transforms as T

from SCP.benchmark.scp import SCP
from SCP.benchmark.weights import download_pretrained_weights
from SCP.datasets import datasets_loader
from SCP.datasets.presets import load_test_presets
from SCP.models.model import load_model
from SCP.utils.clusters import create_clusters, aggregation_per_class_and_cluster, distance_to_clusters_averages,\
    silhouette_score_log
from SCP.utils.common import load_config, get_batch_size, my_custom_logger, create_str_for_ood_method_results, \
    find_idx_of_class
from SCP.benchmark import MSP, ODIN, EnergyOOD
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
    save_scp_hist = save_baseline_hist = save_odin_hist = save_energy_hist = False
    args.save_histograms_for = [method.lower() for method in args.save_histograms_for]
    if "scp" in args.save_histograms_for:
        save_scp_hist = True
    if "baseline" in args.save_histograms_for:
        save_baseline_hist = True
    if "odin" in args.save_histograms_for:
        save_odin_hist = True
    if "energy" in args.save_histograms_for:
        save_energy_hist = True

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

    # Dataframe to store the results
    COLUMNS = ['Timestamp', 'In-Distribution', 'Out-Distribution', 'Model',
               'Test set accuracy', 'Accuracy in OOD test set', 'OoD Method',
               'AUROC', 'AUPR', 'FPR95', 'FPR80', 'Temperature']
    df_results = pd.DataFrame(columns=COLUMNS)

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
        # TODO: Think of a better way to handle datasets
        batch_size = get_batch_size(config, in_dataset, logger)
        train_data, train_loader, test_loader = datasets_loader[in_dataset](batch_size, datasets_path)
        g_ind = torch.Generator()
        g_ind = g_ind.manual_seed(args.ind_seed)
        # Define the test presets for the train data as we need the data to be of the same distribution
        # as In Distribution test data
        train_data.transform = load_test_presets(img_shape=datasets_conf[in_dataset]['input_size'],
                                                 real_shape=train_data[0][0].shape)
        if in_dataset == 'Letters':
            train_data.transform = T.Compose(
                [
                    lambda img: T.functional.rotate(img, -90),
                    lambda img: T.functional.hflip(img),
                    T.transforms.ToTensor(),
                ]
            )
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            generator=g_ind,
        )
        class_names = train_loader.dataset.classes
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

            # TODO: Mejorar la forma de acceder al dataset... El argumento hidden layers podría empezar a llamarse
            #   de otra manera quiza...
            # Load weights
            weights_path = Path(
                f'state_dict_{in_dataset}_{model_name}_{hidden_neurons}_{output_neurons}_{args.n_hidden_layers}_layers.pth'
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
                spk_count_train_clusters = spk_count_train[correctly_classfied_idx]
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
            file_name = figures_path / f'{in_dataset}_{model_name}_{args.cluster_mode}_{hidden_neurons}_{output_neurons}_{args.n_hidden_layers}_layers'
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
            if args.use_test_labels:
                preds_test = test_labels
            logger.info(f"The accuracy of the model with loaded weights of {in_dataset} is {test_accuracy} %")
            spk_count_test = np.sum(_spk_count_test, axis=0, dtype='uint16')
            logger.info(f'Test set: {spk_count_test.shape}')
            
            if args.use_only_correct_test_images:
                pos_correct_preds_test = np.where(preds_test == test_labels)[0]
                preds_test = preds_test[pos_correct_preds_test]
                spk_count_test = spk_count_test[pos_correct_preds_test]
                new_number_of_samples_for_metrics = len(preds_test)
                logger.info(f'Only using correctly classified samples... '
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

            # Computation of the distances of train and test
            distances_train_per_class, _ = distance_to_clusters_averages(
                spk_count_train_thr, preds_train_thr, agg_counts_per_class_cluster, n_classes
            )
            distances_test_per_class, _ = distance_to_clusters_averages(
                spk_count_test, preds_test, agg_counts_per_class_cluster, n_classes
            )

            # ---------------------------------------------------------------
            # Evaluate OOD performance
            # ---------------------------------------------------------------
            for ood_dataset in tqdm(ood_datasets_to_test, desc='Out-of-Distribution dataset loop'):

                logger.info(f'Logs for benchmark with the OoD dataset {ood_dataset}')

                new_figures_path = figures_path / f'{in_dataset}_vs_{ood_dataset}_{model_name}_{hidden_neurons}_{output_neurons}_{args.n_hidden_layers}_layers'

                # ---------------------------------------------------------------
                # Load dataset and extract spikes and logits
                # ---------------------------------------------------------------
                size_test_data = 0
                size_ood_data = 0
                # Load OoD dataset from the dictionary. In case it is MNIST-C, load the selected option
                # In case the OOD test dataset has not enough instances, the train dataset is loaded
                batch_size_ood = get_batch_size(config, ood_dataset, logger)
                if ood_dataset.split('/')[0] == 'MNIST-C':
                    ood_loader = datasets_loader[ood_dataset.split('/')[0]](
                        batch_size_ood,
                        datasets_path,
                        test_only=True,
                        option=ood_dataset.split('/')[1]
                    )
                else:
                    ood_loader = datasets_loader[ood_dataset](
                        batch_size_ood, datasets_path,
                        test_only=True, image_shape=datasets_conf[in_dataset]['input_size']
                    )
                    # TODO: Test for BW datasets if causes errors, as test_only is not present in some datasets
                    size_test_data = len(preds_test)
                    size_ood_data = len(ood_loader.dataset)

                    if size_ood_data == size_test_data:
                        pass

                    elif size_ood_data < size_test_data:
                        logger.info(f"Using training data as test OOD data for {ood_dataset} dataset")
                        ood_train_data, _, _ = datasets_loader[ood_dataset](
                            batch_size_ood, datasets_path,
                            test_only=False, image_shape=datasets_conf[in_dataset]['input_size']
                        )
                        ood_transform = load_test_presets(img_shape=datasets_conf[in_dataset]['input_size'],
                                                          real_shape=ood_train_data[0][0].shape)
                        # ood_transform = load_test_presets(datasets_conf[in_dataset]['input_size'])
                        # TODO: Find a better way to do this
                        if ood_dataset == 'Caltech101' or ood_dataset == 'FER2013':
                            ood_transform = Compose(
                                [
                                    ood_transform,
                                    Lambda(lambda x: x.repeat(3, 1, 1) if (x.shape[0] == 1) else x),
                                ]
                            )
                        ood_train_data.transform = ood_transform
                        if ood_dataset == 'Letters':  # TODO: Handle in a different way
                            ood_train_data.transform = T.Compose(
                                [
                                    lambda img: T.functional.rotate(img, -90),
                                    lambda img: T.functional.hflip(img),
                                    T.transforms.ToTensor(),
                                ]
                            )
                        g_ood = torch.Generator()
                        g_ood.manual_seed(8)
                        rnd_idxs = torch.randint(
                            high=len(ood_train_data), size=(size_test_data,), generator=g_ood)
                        ood_subset = Subset(ood_train_data, [x for x in rnd_idxs.numpy()])
                        ood_loader = DataLoader(ood_subset, batch_size=batch_size_ood, shuffle=False)

                    else:  # size_ood_data > size_test_data
                        logger.info(f"Reducing the number of samples for OOD dataset {ood_dataset} to match"
                                    f"the number of samples of test data, equal to {size_test_data}")
                        g_ood = torch.Generator()
                        g_ood.manual_seed(args.ood_seed)
                        rnd_idxs = torch.randint(high=len(ood_loader.dataset), size=(size_test_data,), generator=g_ood)
                        if ood_dataset == 'Letters':  # TODO: Handle in a different way
                            ood_loader.dataset.transform = T.Compose(
                                [
                                    lambda img: T.functional.rotate(img, -90),
                                    lambda img: T.functional.hflip(img),
                                    T.transforms.ToTensor(),
                                ]
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
                if size_ood_data > size_test_data:
                    preds_ood = preds_ood[rnd_idxs]
                    spk_count_ood = spk_count_ood[rnd_idxs]
                logger.info(f'OoD set: {spk_count_ood.shape}')

                # ---------------------------------------------------------------
                # OOD Detection
                # ---------------------------------------------------------------
                # *************** SCP ***************
                # Computation of the distances of ood instances
                distances_ood_per_class, _ = distance_to_clusters_averages(
                    spk_count_ood, preds_ood, agg_counts_per_class_cluster, n_classes
                )

                scp = SCP()
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
                        len(class_names), distances_train_per_class
                    )
                    # Compute if test instances are classified as InD or OoD for every tpr
                    in_or_out_distribution_per_tpr_test = compare_distances_per_class_to_distance_thr_per_class(
                        distances_test_per_class,
                        dist_thresholds)
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

                    columns = ['Total test samples', 'TPR [%]', 'FN [%]', 'FN [Total]',
                               'FN correctly classified [%]', 'FN misclassified [%]', 'Accuracy of the model']
                    df_fn_incorrect_vs_correct = pd.DataFrame(columns=columns)
                    for i, fn_correct_vs_incorrect in enumerate(fn_correct_vs_incorrect_per_tpr):
                        df_fn_incorrect_vs_correct.loc[len(df_fn_incorrect_vs_correct)] = [
                            len(preds_test),
                            tprs_to_extract[i],
                            len(fn_correct_vs_incorrect) / len(preds_test),
                            len(fn_correct_vs_incorrect),
                            len(np.nonzero(fn_correct_vs_incorrect)[0]) / len(fn_correct_vs_incorrect),
                            (len(fn_correct_vs_incorrect) - len(np.nonzero(fn_correct_vs_incorrect)[0])) / len(fn_correct_vs_incorrect),
                            test_accuracy,
                        ]
                    df_fn_incorrect_vs_correct.to_excel(results_path / f'fn_misclassified_{args.cluster_mode}_fmax_'
                                                                       f'{args.f_max}_timesteps_{args.n_time_steps}.xlsx')

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

            # ---------------------------------------------------------------
            # Save results
            # ---------------------------------------------------------------
            # Save the results in the results list to a dataframe and the save it to a file
            df_results_one_run = pd.DataFrame(results_list, columns=COLUMNS)
            # Save checkpoint into .csv format
            df_results_one_run.to_csv(results_path / f'checkpoint_{in_dataset}.csv')
            df_results = pd.concat([df_results, df_results_one_run])

    # Save all the results to excel
    df_results.to_excel(results_path / f'benchmark_results_{args.conf}_{args.cluster_mode}_fmax_{args.f_max}_'
                                       f'timesteps_{args.n_time_steps}.xlsx')


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
