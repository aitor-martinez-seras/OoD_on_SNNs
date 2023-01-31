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

from SCP.benchmark.scp import SCP
from SCP.benchmark.weights import download_pretrained_weights
from SCP.datasets import datasets_loader
from SCP.datasets.presets import load_test_presets
from SCP.datasets.utils import indices_of_every_class_for_subset
from SCP.models.model import load_model
from SCP.utils.clusters import create_clusters, average_per_class_and_cluster, distance_to_clusters_averages
from SCP.utils.common import load_config, get_batch_size, my_custom_logger, create_str_for_ood_method_results
from SCP.benchmark import MSP, ODIN, EnergyOOD
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
                        help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("--use-test-labels", action='store_true', dest='use_test_labels',
                        help="if passed, the labels used to determine which aggregated clusters to compare to"
                             "are the real labels, not the predictions as in real world scenario")
    parser.add_argument("--use-only-correct-test-images", action='store_true', dest='use_only_correct_test_images',
                        help="if passed, the labels used to determine which aggregated clusters to compare to are"
                             "only correctly predicted images, not all the predictions as in real world scenario")
    parser.add_argument("--random-samples-for-thr", action='store_true', dest='random_samples_for_thr',
                        help="if passed, the thresholds are defined using a random subset of train")
    parser.add_argument("--save-histograms-for", default=[], type=str, nargs='+', dest="save_histograms_for",
                        help="saves histogram plots for the specified methods. Options: SCP, Baseline, ODIN, Energy")
    parser.add_argument("--save-metric-plots", action='store_true', dest='save_metric_plots',
                        help="if passed, AUROC and AUPR Curves are saved")
    return parser


def main(args: argparse.Namespace):
    # -----------------
    # Settings
    # -----------------
    print(args)

    # Load config
    print(f'Loading configuration from {args.conf}.toml')
    config = load_config(args.conf)

    # Parsing some options
    if args.cluster_mode not in ["predictions", "labels", "correct-predictions"]:
        raise AssertionError(f"Options for cluster-mode are: labels or correct-predictions, not {args.cluster_mode}")

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
                weights_path = Path(
                    f'state_dict_{in_dataset}_{model_name}_{hidden_neurons}_{output_neurons}_{args.n_hidden_layers}_layers.pth'
                )
                if not weights_path.exists():
                    print(f'As {weights_path} does not exist, pretrained weights will be downloaded')
                    download_pretrained_weights(pretrained_weights_path=pretrained_weights_folder_path)
                else:
                    exist = True
        if exist:
            print('Pretrained weights are correctly in path')

    # Dataframe to store the results
    COLUMNS = ['Timestamp', 'In-Distribution', 'Out-Distribution', 'Model',
               'Training accuracy', 'Accuracy in OOD test set', 'OoD Method',
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

        # ---------------------------------------------------------------
        # Load in-distribution data from the dictionary
        # ---------------------------------------------------------------
        batch_size = get_batch_size(config, in_dataset, logger)
        train_data, train_loader, test_loader = datasets_loader[in_dataset](batch_size, datasets_path)
        # TODO: Add generator and change the way of loading the dataloader and the dataset
        #   This is to create the clusters from the same images as the one being tested
        g_ind = torch.Generator()
        g_ind = g_ind.manual_seed(6)
        train_data.transform = load_test_presets(img_shape=datasets_conf[in_dataset]['input_size'])
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
            number_of_samples_per_class = 1200
            selected_indices_per_class = indices_of_every_class_for_subset(
                train_loader,  # Here is generated the variability in the cluster creation if no generator is defined
                n_samples_per_class=args.samples_for_cluster_per_class,
                dataset_name=in_dataset
            )
            training_subset_clusters = Subset(train_data, [x for x in selected_indices_per_class])
            subset_train_loader_clusters = DataLoader(
                training_subset_clusters, batch_size=batch_size, shuffle=False
            )
            accuracy_subset_train_clusters, preds_train_clusters, _, _spk_count_train_clusters, labels_subset_train_clusters = validate_one_epoch(
                model, device, subset_train_loader_clusters, return_logits=True, return_targets=True
            )
            logger.info(f'Accuracy for the train clusters subset is {accuracy_subset_train_clusters:.3f} %')
            # Convert spikes to counts
            spk_count_train_clusters = np.sum(_spk_count_train_clusters, axis=0, dtype='uint16')
            logger.info(f'Train subset for clusters: {spk_count_train_clusters.shape}')

            # Define cluster mode
            if args.cluster_mode == "predictions":
                labels_for_clustering = preds_train_clusters
            elif args.cluster_mode == "labels":
                labels_for_clustering = labels_subset_train_clusters
            elif args.cluster_mode == "correct-predictions":
                raise NotImplementedError("Not yet implemented the correct-predictions mode")
            else:
                raise NameError(f"Wrong cluster mode {args.cluster_mode}")

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
            silh_scores_name = figures_path / f'{in_dataset}_{model_name}_{hidden_neurons}_{output_neurons}_{args.n_hidden_layers}_layers'
            clusters_per_class, logging_info = create_clusters(
                labels_for_clustering,
                spk_count_train_clusters,
                class_names,
                distance_for_clustering=dist_clustering,
                size=args.samples_for_cluster_per_class,
                verbose=2,
                name=silh_scores_name,
            )
            logger.info(logging_info)
            
            # ---------------------------------------------------------------
            # Create a subset of training to calculate the thresholds
            # ---------------------------------------------------------------
            # Train subset to create the thresholds
            # Introduce a the init_pos parameters to not select the same indices that for
            # the subset for creating the clusters
            # TODO: Handle cases where we don't have sufficient training data
            if args.random_samples_for_thr:
                g_thr = torch.Generator()
                g_thr.manual_seed(7)
                rnd_idxs = torch.randint(high=len(train_data), size=(args.samples_for_thr_per_class,), generator=g_thr)
                training_subset = Subset(train_data, [x for x in rnd_idxs.numpy()])
                subset_train_loader = DataLoader(training_subset, batch_size=batch_size, shuffle=False)
            else:
                selected_indices_per_class = indices_of_every_class_for_subset(
                    train_loader,
                    n_samples_per_class=args.samples_for_thr_per_class,
                    dataset_name=in_dataset,
                    init_pos=number_of_samples_per_class * n_classes
                )
                training_subset = Subset(train_data, [x for x in selected_indices_per_class])
                subset_train_loader = DataLoader(training_subset, batch_size=batch_size, shuffle=False)

            # Extract the logits and the hidden spikes
            accuracy_subset_train_thr, preds_train_thr, logits_train_thr, _spk_count_train_thr = validate_one_epoch(
                model, device, subset_train_loader, return_logits=True
            )
            logger.info(f'Accuracy for the train subset is {accuracy_subset_train_thr:.3f} %')
            # Convert spikes to counts
            spk_count_train_thr = np.sum(_spk_count_train_thr, axis=0, dtype='uint16')
            logger.info(f'Train subset for threshold: {spk_count_train_thr.shape}')

            # Initialize for every model, as we save the results for every model
            results_list = []

            # ---------------------------------------------------------------
            # Extract predictions and hidden spikes from test InD data
            # ---------------------------------------------------------------

            model.eval()
            test_accuracy, preds_test, logits_test, _spk_count_test, test_labels = validate_one_epoch(
                model, device, test_loader, return_logits=True, return_targets=True
            )
            if args.use_test_labels:
                preds_test = test_labels
            logger.info(f"The accuracy of the model with loaded weights of {in_dataset} is {test_accuracy} %")
            # Convert spikes to counts
            spk_count_test = np.sum(_spk_count_test, axis=0, dtype='uint16')
            logger.info(f'Test set: {spk_count_test.shape}')
            
            if args.use_only_correct_test_images:
                preds_test = np.where(preds_test == test_labels)[0]
                new_number_of_samples_for_metrics = len(preds_test)
                spk_count_test = spk_count_test[:new_number_of_samples_for_metrics]
                logger.info(f'Only using correctly classified samples... '
                            f'New number of samples for metrics: {new_number_of_samples_for_metrics}')

            # ---------------------------------------------------------------
            # Evaluate OOD performance
            # ---------------------------------------------------------------
            for ood_dataset in tqdm(ood_datasets_to_test, desc='Out-of-Distribution dataset loop'):

                logger.info(f'Logs for benchmark with the OoD dataset {ood_dataset}')

                new_figures_path = figures_path / f'{in_dataset}_vs_{ood_dataset}_{model_name}_{hidden_neurons}_{output_neurons}_{args.n_hidden_layers}_layers'
                # ---------------------------------------------------------------
                # Load dataset and extract spikes and logits
                # ---------------------------------------------------------------
                # Load OoD dataset from the dictionary. In case it is MNIST-C, load the
                # selected option
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
                    size_test_data = len(test_loader.dataset)
                    size_ood_data = len(ood_loader.dataset)

                    if size_ood_data == size_test_data:
                        pass

                    elif size_ood_data < size_test_data:
                        logger.info(f"Using training data as test OOD data for {ood_dataset} dataset")
                        ood_train_data, _, _ = datasets_loader[ood_dataset](
                            batch_size_ood, datasets_path,
                            test_only=False, image_shape=datasets_conf[ood_dataset]['input_size']
                        )
                        ood_transform = load_test_presets(datasets_conf[in_dataset]['input_size'])
                        if ood_dataset == 'Caltech101':  # TODO: Find a better way to do this
                            ood_transform = Compose(
                                [
                                    ood_transform,
                                    Lambda(lambda x: x.repeat(3, 1, 1) if (x.shape[0] == 1) else x),
                                ]
                            )
                        ood_train_data.transform = ood_transform
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
                        g_ood.manual_seed(8)
                        rnd_idxs = torch.randint(
                            high=len(ood_loader.dataset), size=(size_test_data,), generator=g_ood)
                        ood_subset = Subset(ood_loader.dataset, [x for x in rnd_idxs.numpy()])
                        ood_loader = DataLoader(ood_subset, batch_size=batch_size_ood, shuffle=False)

                # Extract the spikes and logits for OoD
                accuracy_ood, preds_ood, logits_ood, _spk_count_ood = validate_one_epoch(
                    model, device, ood_loader, return_logits=True
                )
                accuracy_ood = f'{accuracy_ood:.3f}'
                logger.info(f'Accuracy for the ood dataset {ood_dataset} is {accuracy_ood} %')

                if args.use_only_correct_test_images:
                    preds_ood = preds_ood[:new_number_of_samples_for_metrics]
                    _spk_count_ood = _spk_count_ood[:new_number_of_samples_for_metrics]

                # ---------------------------------------------------------------
                # OOD Detection
                # ---------------------------------------------------------------
                # *************** SCP ***************
                # Convert spikes to counts
                if isinstance(_spk_count_ood, tuple):
                    _spk_count_ood, _ = _spk_count_ood
                spk_count_ood = np.sum(_spk_count_ood, axis=0, dtype='uint16')
                logger.info(f'OoD set: {spk_count_ood.shape}')

                # Create the median aggregations for each cluster of each class
                agg_counts_per_class_cluster = average_per_class_and_cluster(
                    spk_count_train_clusters,
                    labels_for_clustering,
                    clusters_per_class,
                    n_classes,
                    n_samples=1000, option='median'
                )
                # Computation of the distances of train, test and ood
                distances_train_per_class, _ = distance_to_clusters_averages(
                    spk_count_train_thr, preds_train_thr, agg_counts_per_class_cluster, n_classes
                )
                distances_test_per_class, _ = distance_to_clusters_averages(
                    spk_count_test, preds_test, agg_counts_per_class_cluster, n_classes
                )
                distances_ood_per_class, _ = distance_to_clusters_averages(
                    spk_count_ood, preds_ood, agg_counts_per_class_cluster, n_classes
                )

                scp = SCP()
                auroc, aupr, fpr95, fpr80 = scp(
                    distances_train_per_class, distances_test_per_class, distances_ood_per_class,
                    save_histogram=save_scp_hist, name=new_figures_path, class_names=class_names, preds_ood=preds_ood
                )
                if args.save_metric_plots:
                    scp.save_auroc_fig(new_figures_path)
                    scp.save_aupr_fig(new_figures_path)

                # # Creation of the array with the thresholds for each TPR (class, dist_per_TPR)
                # distance_thresholds_train = thresholds_per_class_for_each_TPR(
                #     n_classes, distances_train_per_class
                # )
                # # Computing precision, tpr and fpr
                # precision, tpr_values, fpr_values = compute_precision_tpr_fpr_for_test_and_ood(
                #     distances_test_per_class, distances_ood_per_class, distance_thresholds_train
                # )
                # # Appending that when FPR = 1 the TPR is also 1:
                # tpr_values_auroc = np.append(tpr_values, 1)
                # fpr_values_auroc = np.append(fpr_values, 1)
                # # Metrics
                # auroc = round(np.trapz(tpr_values_auroc, fpr_values_auroc) * 100, 2)
                # aupr = round(np.trapz(precision, tpr_values) * 100, 2)
                # fpr95 = round(fpr_values_auroc[95] * 100, 2)
                # fpr80 = round(fpr_values_auroc[80] * 100, 2)

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
    df_results.to_excel(results_path / f'benchmark_results_{args.cluster_mode}_fmax_{args.f_max}_'
                                       f'timesteps_{args.n_time_steps}.xlsx')


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
