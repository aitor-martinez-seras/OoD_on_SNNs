from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import numpy as np
from torchvision.utils import make_grid
from tqdm import tqdm
from Orange.evaluation import compute_CD, graph_ranks
from baycomp import SignedRankTest, SignTest


# ----------------------
# Bayesian Analysis and CD Graph
# ----------------------
def cd_graph(score_per_method: OrderedDict, fig_path: Path):
    # TeX must be installed to use this statement
    plt.rcParams.update({"text.usetex": True, "font.family": "Latin Modern Sans"})
    ood_method_names = list(score_per_method.keys())
    score_per_method = np.transpose(np.array(list(score_per_method.values())))

    number_of_ood_datasets = np.shape(score_per_method)[0]
    order = np.argsort(-score_per_method, axis=1)
    ranks = np.argsort(order, axis=1) + 1
    avgranks = np.mean(ranks, axis=0)

    for i, name in enumerate(ood_method_names):
        ood_method_names[i] = _change_ensemble_name(name)

    CD = compute_CD(avgranks, number_of_ood_datasets, test='nemenyi')
    graph_ranks(avgranks, ood_method_names, cd=CD, width=5, textspace=0.8)
    plt.savefig(fr'{fig_path.as_posix()}CD_Graph.pdf', bbox_inches='tight')  # bbox_inches=Bbox([[0.2, 0], [0.8, 1]]))
    plt.close()


def _change_ensemble_name(name):
    if name == 'Ensemble-Odin-SCP':
        name = 'E-ODIN-SCP'
    elif name == 'Ensemble-Odin-Energy':
        name = 'E-ODIN-Energy'
    elif name == 'Ensemble-Energy-SCP':
        name = 'E-Energy-SCP'
    return name


def bayesian_test(scores_dict: OrderedDict, option: str, fig_path: Path, rope: float, use_bbox: bool):
    # TeX must be installed to use this statement
    plt.rcParams.update({"text.usetex": True, "font.family": "Latin Modern Sans"})
    if option == 'signrank':
        bayesian_test_obj = SignedRankTest
    elif option == 'signtest':
        bayesian_test_obj = SignTest
    else:
        raise NameError('Wrong option selected')

    method_left, method_right = scores_dict.keys()

    name_method_left = _change_ensemble_name(method_left)
    name_method_right = _change_ensemble_name(method_right)

    names = (name_method_left, name_method_right)

    # print(bayesian_test_obj.probs(scores_dict[method_left], scores_dict[method_right], rope=rope))
    bayesian_test_obj.plot(scores_dict[method_left], scores_dict[method_right], rope=rope, names=names, nsamples=5000)
    if use_bbox:
        from matplotlib.transforms import Bbox
        bbox = Bbox([[1.25, 0.75], [5.5, 3.55]])
        plt.savefig(fr'{fig_path.as_posix()}_{option}.pdf', bbox_inches=bbox)
    else:
        plt.savefig(fr'{fig_path.as_posix()}_{option}.pdf', bbox_inches='tight')
    plt.close()

# ----------------------
# General plots and auxiliary functions
# ----------------------


def plot_ax(ax, img, plt_range, cmap, alpha=1, title=None, fontsize=8, xlabel=None):
    ax.set_xticks([])
    ax.set_yticks([])
    im = ax.imshow(img, vmin=plt_range[0], vmax=plt_range[1], cmap=cmap, alpha=alpha)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    return im


def plot_grid(images, size=8):
    grid = make_grid(images, nrow=size)
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()


def plot_image(image, height=6, width=6):
    plt.subplots(1, 1, figsize=(width, height))
    plt.imshow(image)
    plt.show()


def show_img_from_dataloader(data_loader, img_pos=0, number_of_iterations=1):
    images, targets = _iterate_dataloader(data_loader, number_of_iterations)
    print('Max:', images[img_pos].max(), 'Min:', images[img_pos].min())
    plt.imshow(images[img_pos].permute(1, 2, 0))
    plt.show()


def show_grid_from_dataloader(data_loader, number_of_iterations=1):
    images, targets = _iterate_dataloader(data_loader, number_of_iterations)
    print('Targets:', targets)
    grid = make_grid(images, nrow=8)
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()


def _iterate_dataloader(data_loader, number_of_iterations):
    assert number_of_iterations >= 1, 'number_of_iterations must be greater or equal than 1'
    for _ in range(number_of_iterations):
        images, targets = next(iter(data_loader))
    return images, targets


def plot_loss_history(training_losses, test_losses, fpath=""):
    plt.figure(facecolor="w", figsize=(10, 5))
    plt.plot(training_losses)
    plt.plot(test_losses)
    plt.title("Loss Curves")
    plt.legend(["Train Loss", "Test Loss"])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()
    if fpath:
        plt.savefig(fpath)
        plt.close()


# ----------------------
# Cluster plots
# ----------------------

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def plot_dendogram_per_class(class_names, clusters_per_class, name, save=True):
    """
    Plot the top three levels of the dendrogram for each class
    """
    n_classes = len(class_names)
    fig, axes = plt.subplots(2, int(n_classes / 2), figsize=(6 * n_classes / 2, 12))
    fig.suptitle('Hierarchical Clustering Dendrogram', fontsize=22, y=0.94)
    # fig.supxlabel('X axis: Number of points in node
    # (index of the number if not in parenthesis)',fontsize = h + w*0.1,y=0.065)

    for class_index, ax in tqdm(enumerate(axes.flat), desc='Create the clusters with the selected distance thresholds'):
        plot_dendrogram(clusters_per_class[class_index], truncate_mode='level', p=3, ax=ax)
        ax.set_title('Class {}'.format(class_names[class_index]), fontsize=22)
        # ax[i,j].set_xlabel("Number of points in node",fontsize=h)
    if save:
        fig.savefig(f'{name}_DendrogramPerClass.pdf')
        plt.close(fig)
    else:
        fig.show()


def plot_clusters_performance(
        class_names,
        cluster_performance_for_all_possible_thresholds_per_class, possible_distance_thrs,
        clustering_performance_scores_for_selected_thresholds_per_class, selected_distance_thrs_per_class,
        name, performance_measuring_method, save=True
):
    n_classes = len(class_names)
    fig, axes = plt.subplots(2, int(n_classes / 2), figsize=(6 * n_classes / 2, 12))

    for class_index, ax in enumerate(axes.flat):
        ax.plot(possible_distance_thrs,
                cluster_performance_for_all_possible_thresholds_per_class[class_index], color='blue')
        ax.plot(selected_distance_thrs_per_class[class_index],
                clustering_performance_scores_for_selected_thresholds_per_class[class_index], 'ro')
        ax.set_title(class_names[class_index])

    if save is True:
        fig.savefig(f'{name}_{performance_measuring_method}.pdf')
        plt.close(fig)
    else:
        fig.show()


# ----------------------
# OOD Metric plots
# ----------------------

def plot_auroc(fpr, tpr, save=''):
    # AUC
    auc = np.trapz(tpr, fpr)
    # Plot
    plt.figure(figsize=(15, 12))
    # Plot lines
    plt.plot(fpr, tpr, label='ROC curve', lw=3)
    plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), 'k--', label='Random ROC curve')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('FPR', fontsize=20)
    plt.ylabel('TPR', fontsize=20)
    plt.title('ROC curve, AUC = %.3f' % auc, fontsize=25, pad=10)

    plt.fill_between(fpr, tpr, alpha=0.3)

    # Create empty plot with blank marker containing the extra label
    plt.plot([], [], ' ', label=f'FPR at 95% TPR = {round(fpr[95] * 100, 2)}%')
    plt.plot([], [], ' ', label=f'FPR at 80% TPR = {round(fpr[80] * 100, 2)}%')
    plt.legend(fontsize=20, loc='upper left')

    if save:
        plt.savefig(f'{save}_AUPR.png', dpi=200)
        plt.close()


def plot_aupr(precision, tpr, save=''):
    # AUPR
    auc = np.trapz(precision, tpr)
    # Plot
    plt.figure(figsize=(15, 12))
    plt.plot(tpr, precision, label='ROC curve', lw=3)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('FPR', fontsize=20)
    plt.ylabel('Precision', fontsize=20)
    plt.title('PR curve, AUC = %.3f' % auc, fontsize=25, pad=10)
    plt.fill_between(tpr, precision, alpha=0.3)
    plt.legend(fontsize=20, loc='upper left')
    if save:
        plt.savefig(f'{save}_AUPR.png', dpi=200)
        plt.close()


def plot_histogram(train, test, ood):
    plt.figure(figsize=(10, 5), tight_layout=True)
    plt.hist(train, bins=50, color='blue', alpha=0.6, density=True, label='Train')
    plt.hist(test, bins=50, color='green', alpha=0.6, density=True, label='Test')
    plt.hist(ood, bins=50, color='darkorange', alpha=0.6, density=True, label='ood')
    # plt.ylim([0,10])
    plt.legend(fontsize=18)
    plt.show()
