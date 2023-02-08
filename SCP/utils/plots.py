import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import numpy as np
from torchvision.utils import make_grid


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


def plot_clusters_performance(class_names, cluster_performance_for_all_possible_thresholds_per_class,
                              possible_distance_thrs, selected_distance_thrs_per_class,
                              clustering_performance_scores_for_selected_thresholds_per_class,
                              name, performance_measuring_method, save=True):
    print('Selected distance thresholds:\n', selected_distance_thrs_per_class)
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
