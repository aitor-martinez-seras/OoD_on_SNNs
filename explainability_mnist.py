from pathlib import Path
import argparse
from typing import Dict

import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

from SCP.datasets import datasets_loader
from SCP.datasets.utils import load_dataloader
from SCP.models.model import load_model, load_weights
from SCP.utils.clusters import create_clusters, aggregation_per_class_and_cluster, distance_to_clusters_averages
from SCP.utils.common import load_config, find_idx_of_class
from SCP.explainable.utils import compute_reconstruction_per_class, extract_positive_part_per_class, \
    rearrange_to_ftmaps, rearrange_to_ftmaps_per_class, auroc_aupr
from SCP.utils.plots import plot_ax
from test import validate_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser(description="Explainability of OOD detector on SNN", add_help=True)

    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-m", "--model", required=True, type=str, choices=['Fully_connected', 'ConvNet'],
                        help="name of the model")
    parser.add_argument("-b", "--batch-size", default=128, type=int, dest='batch_size', help="batch_size")
    parser.add_argument("-n", "--number-of-plots", default=20, type=int, dest='number_of_plots',
                        help="number of figures to plot")
    parser.add_argument("--pretrained", action="store_true", default=False, help="For using the weights of the paper")
    parser.add_argument("-mc", "--mnist-c-option",dest='mnist_c_option', default="zigzag", type=str, help="Option of mnist-c",
                        choices=['zigzag', 'canny_edges', 'dotted_line', 'fog', 'glass_blur', 'impulse_noise',
                                 'motion_blur', 'rotate', 'scale', 'shear', 'shot_noise', 'spatter',
                                 'stripe', 'translate', 'brightness'])
    parser.add_argument("--encoder", default="poisson", type=str,
                        help="encoder to use. Options 'poisson' and 'constant'")
    parser.add_argument("--n-time-steps", default=50, type=int, dest='n_time_steps',
                        help="number of timesteps for the simulation")
    parser.add_argument("--f-max", default=100, type=int, dest='f_max',
                        help="max frecuency of the input neurons per second")
    parser.add_argument("--n-hidden-layers", default=1, type=int,
                        dest="arch_selector", help="number of hidden layers of the models")
    parser.add_argument("--samples-for-cluster-per-class", default=1000, type=int,
                        dest="samples_for_cluster_per_class", help="number of samples for validation per class")
    parser.add_argument("--samples-for-thr-per-class", default=1000, type=int,
                        dest="samples_for_thr_per_class", help="number of samples for validation per class")
    parser.add_argument("--ind-seed", default=7, type=int, dest='ind_seed',
                        help="seed for the selection of the instances for MNIST")
    parser.add_argument("--ood-seed", default=8, type=int, dest='ood_seed',
                        help="seed for the selection of ood test images")
    return parser


def main(args):
    print('****************** Starting explainability script ******************')
    print(args)

    # ---------------------------------------------------------------
    # Load configuration
    # ---------------------------------------------------------------
    config = load_config('explainability-mnist')

    # Device for computation
    device = args.device if torch.cuda.is_available() else torch.device('cpu')

    # Paths
    paths_config = load_config('paths')
    logs_path = Path(paths_config["paths"]["logs"])
    weights_folder_path = Path(paths_config["paths"]["weights"])
    datasets_path = Path(paths_config["paths"]["datasets"])
    pretrained_weights_folder_path = Path(paths_config["paths"]["pretrained_weights"])
    figures_path = Path(paths_config["paths"]["figures"])

    in_dataset = config["in_distribution_datasets"][0]
    mnist_sq, mnist_c = config["out_of_distribution_datasets"]

    # ---------------------------------------------------------------
    # Load In and OoD datasets
    # ---------------------------------------------------------------
    # Load MNIST
    print(f'Loading {in_dataset}...')
    in_dataset_data_loader = datasets_loader[in_dataset](datasets_path)

    all_datasets_conf = load_config('datasets')
    mnist_conf = all_datasets_conf[in_dataset]

    model_archs = config["model_type"]
    input_size = mnist_conf['input_size']
    hidden_neurons = model_archs[args.model][in_dataset][0]
    output_neurons = mnist_conf['classes']

    train_data = in_dataset_data_loader.load_data(
        split='train', transformation_option='test', output_shape=mnist_conf['input_size'][1:]
    )
    test_data = in_dataset_data_loader.load_data(
        split='test', transformation_option='test', output_shape=mnist_conf['input_size'][1:]
    )

    class_names = train_data.classes

    # Define loaders. Use a seeds
    g_train = torch.Generator()
    g_train.manual_seed(args.ind_seed)
    g_test = torch.Generator()
    g_test.manual_seed(args.ind_seed)
    train_loader = load_dataloader(train_data, args.batch_size, shuffle=True, generator=g_train)
    test_loader = load_dataloader(test_data, args.batch_size, shuffle=True, generator=g_test)
    print(f'Load of {in_dataset} completed!')

    # OOD DATA
    g_ood = torch.Generator()
    g_ood.manual_seed(args.ood_seed)
    # Load MNIST-Square
    mnist_sq_dataset_loader = datasets_loader[mnist_sq](datasets_path)
    mnist_sq_data = mnist_sq_dataset_loader.load_data(
        split='test', transformation_option='test', output_shape=mnist_conf['input_size'][1:]
    )
    mnist_sq_loader = load_dataloader(mnist_sq_data, args.batch_size, shuffle=True, generator=g_ood)
    # Load MNIST-C
    mnist_c_dataset_loader = datasets_loader[mnist_c](datasets_path, option=args.mnist_c_option)
    mnist_c_data = mnist_c_dataset_loader.load_data(
        split='test', transformation_option='test', output_shape=mnist_conf['input_size'][1:]
    )
    mnist_c_loader = load_dataloader(mnist_c_data, args.batch_size, shuffle=True, generator=g_ood)

    # ---------------------------------------------------------------
    # Load model and weights
    # ---------------------------------------------------------------
    model = load_model(
        model_type=args.model,
        input_size=input_size,
        hidden_neurons=hidden_neurons,
        output_neurons=output_neurons,
        arch_selector=args.arch_selector,
        encoder=args.encoder,
        n_time_steps=args.n_time_steps,
        f_max=args.f_max
    )

    weights_path = Path(
        f'state_dict_{in_dataset}_{args.model}_{hidden_neurons}'
        f'_{output_neurons}_{args.arch_selector}_layers.pth'
    )

    if args.pretrained:
        weights_path = pretrained_weights_folder_path / weights_path
    else:
        weights_path = weights_folder_path / weights_path

    state_dict = torch.load(weights_path)
    if 'model' in state_dict.keys():  # Handle the case where it has been saved in the updated version
        state_dict = state_dict['model']
    model.load_state_dict(state_dict)
    model = model.to(device)

    # ---------------------------------------------------------------
    # Create clusters and aggregations
    # ---------------------------------------------------------------
    accuracy_train, preds_train, logits_train, _spk_count_train, labels_train = validate_one_epoch(
        model, device, train_loader, return_logits=True, return_targets=True
    )
    spk_count_train = np.sum(_spk_count_train, axis=0, dtype='uint16')

    # Extract spike counts and predictions for the thresholds
    preds_train_thr = preds_train[args.samples_for_cluster_per_class * len(class_names):]
    spk_count_train_thr = spk_count_train[args.samples_for_cluster_per_class * len(class_names):]
    logits_train_thr = logits_train[args.samples_for_cluster_per_class * len(class_names):]

    # Define cluster mode
    spk_count_train_clusters = spk_count_train
    correctly_classfied_idx = np.where(preds_train == labels_train)[0]
    labels_for_clustering = preds_train[correctly_classfied_idx]
    spk_count_train_clusters = spk_count_train_clusters[correctly_classfied_idx]
    labels_for_clustering = preds_train

    # Create clusters
    dist_clustering = (500, 5000)
    clusters_per_class, logging_info = create_clusters(
        labels_for_clustering,
        spk_count_train_clusters,
        class_names,
        distance_for_clustering=dist_clustering,
        n_samples_per_class=args.samples_for_cluster_per_class,
        verbose=1
    )

    # Create the median aggregations (centroids) for each cluster of each class
    print('Spk counts:', spk_count_train_clusters.shape)
    print('Labels:', labels_for_clustering.shape)
    print('Clusters', [len(x.labels_) for x in clusters_per_class])
    agg_counts_per_class_cluster = aggregation_per_class_and_cluster(
        spk_count_train_clusters,
        labels_for_clustering,
        clusters_per_class,
        len(class_names),
        n_samples=args.samples_for_cluster_per_class, option='median'
    )

    # ---------------------------------------------------------------
    # Extract spike counts
    # ---------------------------------------------------------------
    test_accuracy, preds_test, logits_test, _spk_count_test, test_labels = validate_one_epoch(
        model, device, test_loader, return_logits=True, return_targets=True
    )
    mnist_sq_accuracy, preds_mnist_sq, logits_mnist_sq, _spk_count_mnist_sq, mnist_sq_labels = validate_one_epoch(
        model, device, mnist_sq_loader, return_logits=True, return_targets=True
    )
    mnist_c_accuracy, preds_mnist_c, logits_mnist_c, _spk_count_mnist_c, mnist_c_labels = validate_one_epoch(
        model, device, mnist_c_loader, return_logits=True, return_targets=True
    )
    spk_count_test = np.sum(_spk_count_test, axis=0, dtype='uint16')
    spk_count_mnist_sq = np.sum(_spk_count_mnist_sq, axis=0, dtype='uint16')
    spk_count_mnist_c = np.sum(_spk_count_mnist_c, axis=0, dtype='uint16')

    # ---------------------------------------------------------------
    # Compute distances
    # ---------------------------------------------------------------
    distances_train_per_class, _ = distance_to_clusters_averages(
        spk_count_train_thr, preds_train_thr, agg_counts_per_class_cluster, len(class_names)
    )
    distances_test_per_class, closest_clusters_test_per_class = distance_to_clusters_averages(
        spk_count_test, preds_test, agg_counts_per_class_cluster, len(class_names)
    )
    distances_mnist_sq_per_class, closest_clusters_mnist_sq_per_class = distance_to_clusters_averages(
        spk_count_mnist_sq, preds_mnist_sq, agg_counts_per_class_cluster, len(class_names)
    )
    distances_mnist_c_per_class, closest_clusters_mnist_c_per_class = distance_to_clusters_averages(
        spk_count_mnist_c, preds_mnist_c, agg_counts_per_class_cluster, len(class_names)
    )

    # ---------------------------------------------------------------
    # Create reconstructions
    # ---------------------------------------------------------------
    if args.model == 'ConvNet':
        weights = model.snn.state_dict()['fc1.weight'].cpu().numpy()
    elif args.model == 'Fully_connected':
        weights = model.state_dict()['snn.fc1.weight'].cpu().numpy()
    else:
        raise NameError

    reconst_test = np.dot(spk_count_test, weights)
    reconst_mnist_sq = np.dot(spk_count_mnist_sq, weights)
    reconst_mnist_c = np.dot(spk_count_mnist_c, weights)

    # Extract negative part
    reconst_test_neg = np.where(reconst_test <= 0, -reconst_test, 0)
    reconst_mnist_sq_neg = np.where(reconst_mnist_sq <= 0, -reconst_mnist_sq, 0)
    reconst_mnist_c_neg = np.where(reconst_mnist_c <= 0, -reconst_mnist_c, 0)

    # Extract positive part
    reconst_test = np.where(reconst_test >= 0, reconst_test, 0)
    reconst_mnist_sq = np.where(reconst_mnist_sq >= 0, reconst_mnist_sq, 0)
    reconst_mnist_c = np.where(reconst_mnist_c >= 0, reconst_mnist_c, 0)

    # Backprop the hidden counts of the median
    agg_counts_reconstructed_per_class_and_cluster = compute_reconstruction_per_class(agg_counts_per_class_cluster,
                                                                                      weights)
    # Extract the positive part
    avg_frecs_reconst_per_cl_clu_positive = extract_positive_part_per_class(
        agg_counts_reconstructed_per_class_and_cluster)

    # Rearrange ftmaps
    if args.model == 'ConvNet':
        reconst_test = rearrange_to_ftmaps(reconst_test, ftmaps_shape=(50, 11, 11))
        reconst_mnist_sq = rearrange_to_ftmaps(reconst_mnist_sq, ftmaps_shape=(50, 11, 11))
        reconst_mnist_c = rearrange_to_ftmaps(reconst_mnist_c, ftmaps_shape=(50, 11, 11))
        avg_frecs_reconst_per_cl_clu_positive = rearrange_to_ftmaps_per_class(
            avg_frecs_reconst_per_cl_clu_positive, (50, 11, 11)
        )

    # AUROC and AUPR and extrac distance threshodls
    thresholds = auroc_aupr(
        len(class_names), distances_train_per_class, distances_test_per_class, distances_mnist_sq_per_class
    )

    # ---------------------------------------------------------------
    # Define the plot range
    # ---------------------------------------------------------------
    # To define the plot range we take 100 examples of test and process them to
    # obtain the attribution. Then, we take the quantile 80 of all the pixel values
    # in the 100 examples taken.
    array_histogram = np.zeros((100, 28, 28), dtype='float32')
    for ind, i in enumerate(np.random.randint(0, 10000, 100)):
        prediction = preds_test[i]
        # Obtain the index of the represented number in the per class order
        # to extract the closest cluster (as it is in per class order)
        i_in_per_cls_order = np.where(find_idx_of_class(prediction, preds_test) == np.array(i))[0][0]
        if args.model == 'ConvNet':
            array_histogram[ind] = resize(np.sum(np.abs(
                reconst_test[i] - avg_frecs_reconst_per_cl_clu_positive[prediction][
                    closest_clusters_test_per_class[prediction][i_in_per_cls_order]]), axis=0), (28, 28))
        elif args.model == 'Fully_connected':
            array_histogram[ind] = np.abs(reconst_test[i] - avg_frecs_reconst_per_cl_clu_positive[prediction][
                closest_clusters_test_per_class[prediction][i_in_per_cls_order]]).reshape(28, 28)
        else:
            raise NameError
    plot_range = [int(np.quantile(array_histogram.flatten(), 0.80)), int(np.max(array_histogram.flatten()))]
    print("Plot range for heatmaps:", plot_range)

    # array_histrogram = np.zeros((100, 28, 28), dtype='float32')
    # for ind, i in enumerate(np.random.randint(0, 10000, 100)):
    #     prediction = preds_test[i]
    #     # Obtain the index of the represented number in the per class order
    #     to extract the closest cluster (as it is in per class order)
    #     i_in_per_cls_order = np.where(find_idx_of_class(prediction, preds_test) == np.array(i))[0][0]
    #     if args.model == 'ConvNet':
    #         array_histrogram[ind] = resize(np.sum(reconst_test[i], axis=0), (28, 28))
    #     elif args.model == 'Fully_connected':
    #         array_histrogram[ind] = reconst_test[i].reshape(28, 28)
    #     else:
    #         raise NameError
    # plot_range_inputs = [int(np.quantile(array_histrogram.flatten(), 0.80)), int(np.max(array_histrogram.flatten()))]
    # max_plot_value_inputs = plot_range_inputs[1]
    # print("Plot max value for inputs:", max_plot_value_inputs)

    # ---------------------------------------------------------------
    # Make the plots
    # ---------------------------------------------------------------
    for plt_idx in range(args.number_of_plots):
        # Indices selection
        idxs = np.random.randint(0, 10000, 3)
        print(idxs)

        nrows = len(idxs)
        ncols = 3
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
        for i, ax_row in enumerate(axes):

            for j, ax in enumerate(ax_row):

                if j == 0:
                    image = test_loader.dataset[idxs[i]][0].squeeze()
                    prediction = preds_test[idxs[i]]
                    # Obtain the index of the represented number in the per class order
                    # to extract the closest cluster (as it is in per class order)
                    i_in_per_cls_order = np.where(find_idx_of_class(prediction, preds_test) == np.array(idxs[i]))[0][
                        0]
                    diff = np.abs(reconst_test[idxs[i]] - avg_frecs_reconst_per_cl_clu_positive[prediction][
                        closest_clusters_test_per_class[prediction][i_in_per_cls_order]])

                    dist = distances_test_per_class[prediction][i_in_per_cls_order]

                elif j == 1:
                    image = mnist_sq_loader.dataset[idxs[i]][0].squeeze()
                    prediction = preds_mnist_sq[idxs[i]]
                    i_in_per_cls_order = np.where(find_idx_of_class(prediction, preds_mnist_sq) == np.array(idxs[i]))[0][0]
                    diff = np.abs(reconst_mnist_sq[idxs[i]] - avg_frecs_reconst_per_cl_clu_positive[prediction][
                        closest_clusters_mnist_sq_per_class[prediction][i_in_per_cls_order]])

                    dist = distances_mnist_sq_per_class[prediction][i_in_per_cls_order]

                elif j == 2:
                    image = mnist_c_loader.dataset[idxs[i]][0].squeeze()
                    prediction = preds_mnist_c[idxs[i]]
                    i_in_per_cls_order = np.where(find_idx_of_class(prediction, preds_mnist_c) == np.array(idxs[i]))[0][0]
                    diff = np.abs(reconst_mnist_c[idxs[i]] - avg_frecs_reconst_per_cl_clu_positive[prediction][
                        closest_clusters_mnist_c_per_class[prediction][i_in_per_cls_order]])
                    dist = distances_mnist_c_per_class[prediction][i_in_per_cls_order]

                else:
                    raise IndexError

                if args.model == 'ConvNet':
                    diff = resize(np.sum(diff, axis=0), (28, 28))
                plot_ax(ax, img=image, plt_range=[0, 1], cmap='binary')

                which_fpr = 80  # To select the FPR (80 or 95)
                if dist - thresholds[prediction, which_fpr] >= 0:
                    distance_to_centroid = '+ ' + str(dist - thresholds[prediction, which_fpr])
                else:
                    distance_to_centroid = '\N{MINUS SIGN} ' + str(abs(dist - thresholds[prediction, which_fpr]))

                if args.model == 'ConvNet':
                    plot_ax(ax, img=diff.reshape(28, 28), plt_range=plot_range, cmap='Reds', alpha=0.65,
                            title=f'{distance_to_centroid}', xlabel=f'Prediction = {prediction}', fontsize=18)
                elif args.model == 'Fully_connected':
                    plot_ax(ax, img=diff.reshape(28, 28), plt_range=plot_range, cmap='Reds', alpha=0.65,
                            title=f'{distance_to_centroid}', xlabel=f'Prediction = {prediction}', fontsize=18)
                else:
                    raise NameError

                    # plot_ax(ax, img=diff.reshape(28,28), plt_range=[200,1000], cmap='Reds',
                    # alpha=0.65, title=f'{distance_to_centroid}',xlabel=f'Prediction = {prediction}', fontsize=18)
                # plot_ax(ax, img=diff.reshape(28,28), plt_range=[200,1000], cmap='Reds',
                # alpha=0.65, title=f'{prediction} | {dist}', fontsize=18)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.45)
        fig.savefig(figures_path / f'{args.model}_reconstruction_{plt_idx}.pdf', bbox_inches='tight')
        # fig.savefig(figures_path / f'{selected_model}_reconstruction.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
