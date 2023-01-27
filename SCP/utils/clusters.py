import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances

from .common import find_idx_of_class
from .plots import plot_dendrogram


def print_created_clusters_per_class(clusters_per_class, class_names):
    print('')
    print('Created clusters:')
    print('-' * 75)
    # Printing the cluster composition for each class
    for cl_ind, clusters_one_class in enumerate(clusters_per_class):
        unique, counts = np.unique(clusters_one_class.labels_, return_counts=True)
        print('Clase', class_names[cl_ind].ljust(15), '\t', dict(zip(unique, counts)))
        print('-' * 75)


def average_per_class_and_cluster(spike_frecs, preds, clusters_per_class, n_classes,
                                  option='median', n_samples=None):
    """
    Function that receives the counts and the predictions of a subset and the
    clusterization objects per class of that subset and outputs a list of arrays
    with the mean or median hidden frecuency vector of each cluster of each class
    """
    # For every class
    avg_per_class = []
    for class_index in range(n_classes):
        # Calculation of the array of frecuencies that corresponds to a class
        if n_samples is None:
            indices = find_idx_of_class(class_index, preds)
        else:
            indices = find_idx_of_class(class_index, preds, n_samples)
        spikesFrecsOneClass = spike_frecs[indices]
        # For every cluster of the class, compute the median
        avgPerCluster = []
        for cluster_index in np.unique(clusters_per_class[class_index].labels_):
            # We can compute the mean or the median of the neuron frecuencies for that cluster
            if option == 'median':
                avgPerCluster.append(np.median(
                    spikesFrecsOneClass[np.where(clusters_per_class[class_index].labels_ == cluster_index)[0]],
                    axis=0))
            elif option == 'mean':
                avgPerCluster.append(np.mean(
                    spikesFrecsOneClass[np.where(clusters_per_class[class_index].labels_ == cluster_index)[0]],
                    axis=0))
        avgPerCluster = np.array(avgPerCluster)
        # Save all the cluster averages to the list of the classes
        avg_per_class.append(avgPerCluster)
    return avg_per_class


def distance_to_clusters_averages(spike_frecs, predictions, avg_per_class, n_classes):
    """
    Function that computes the distance of the introduced array to the cluster's averages of the predicted class
    Takes the frecuencies and the predictions of each sample of a subset (not ordered by class) and
    the averages of each cluster and class
    :returns distance of each sample to the cluster average for each class
    # TODO: Describe the dimensions of the inputs
    """
    # Order array by predicted class
    spike_frecs_per_class = []
    for class_index in range(n_classes):
        spike_frecs_per_class.append(spike_frecs[find_idx_of_class(class_index, predictions)])
    # Compute the pairwise distances per predicted class
    distances_per_class = []
    closest_clusters_per_class = []
    for class_index, frecs_one_class in enumerate(spike_frecs_per_class):
        if frecs_one_class.size == 0:
            distances_per_class.append([])
            closest_clusters_per_class.append([])
        else:
            parwise_dist = pairwise_distances(frecs_one_class, avg_per_class[class_index], metric='manhattan')
            distances_per_class.append(np.min(parwise_dist, axis=1))
            closest_clusters_per_class.append(np.argmin(parwise_dist, axis=1))
    return distances_per_class, closest_clusters_per_class


def create_clusters(preds_train_clusters, spk_count_train_clusters, class_names, size=1000,
                    distance_for_clustering=None, verbose=2):
    """
    Verbose = 0 -> No prints and plots neither loggin info
    verbose = 1 -> Returns loggin info only
    Verbose = 2 -> Prints and plots
    """
    # Define de number of classes
    n_classes = len(class_names)

    # Select a distance threshold for each class
    if distance_for_clustering is None:
        distance_for_clustering = (800, 3000)
    opt_dist_thr_per_class = []
    opt_silh_score_values_per_class = []
    dist_thrs = np.linspace(distance_for_clustering[0],
                            distance_for_clustering[1], 50)
    silhScoresPerClass = []
    clusterLabels = []

    for class_index in tqdm(range(n_classes), desc='Computing silhuette score for various distance thresholds'):
        dunnIndexes = []
        silh_scores = []
        for dist in dist_thrs:
            indices = find_idx_of_class(class_index, preds_train_clusters, size)
            cluster_model = AgglomerativeClustering(n_clusters=None, metric='manhattan', linkage='average',
                                                    distance_threshold=dist)
            try:  # Handle the case that one class has no representation in the training samples
                cluster_model.fit(spk_count_train_clusters[indices])
                clusterLabels.append(cluster_model.labels_)
            except ValueError as e:
                print('Error probably caused by the lack of training samples for one specific class')
                raise (e)
            try:
                silh_scores.append(
                    silhouette_score(spk_count_train_clusters[indices], cluster_model.labels_, metric='manhattan'))
            except ValueError:
                silh_scores.append(0)
        silhScoresPerClass.append(silh_scores)

        # Iterate the inverted to catch the smallest distance value with the
        # greatest silhouette score
        max_score = 0
        max_index = 0
        for idx, current_score in enumerate(silh_scores):
            # Store the greatest value we encounter traveling the curve
            # Only update the value if it is greater, not if it equal
            if current_score > max_score:
                max_index = idx
                max_score = current_score
        # We append the distance treshold to a list where they are going to be
        # stored, one for each class
        opt_dist_thr_per_class.append(dist_thrs[max_index])
        opt_silh_score_values_per_class.append(silh_scores[max_index])

    # Plot the silhouette score for every distance threshold
    if verbose == 2:
        # Plot to see the silhouette scores
        print('Selected distance thresholds:\n', opt_dist_thr_per_class)
        # TODO: Make figure dependant on the number of classes by a formula
        if n_classes == 10:
            fig, axes = plt.subplots(2, 5, figsize=(6 * n_classes / 2, 12))
        elif n_classes == 26:
            fig, axes = plt.subplots(2, 13, figsize=(6 * n_classes / 2, 12))
        else:
            raise NameError(f'The number of classes {n_classes} is not implemented for the plots')

        for class_index, ax in enumerate(axes.flat):
            ax.plot(dist_thrs, silhScoresPerClass[class_index], color='blue')
            ax.plot(opt_dist_thr_per_class[class_index], opt_silh_score_values_per_class[class_index], 'ro')
            ax.set_title(class_names[class_index])
        plt.savefig('silhouetteScores.pdf')

    # Create the clusters by extracting the labels for every sample
    clusters_per_class = []
    for class_index in range(n_classes):
        indices = find_idx_of_class(class_index, preds_train_clusters, 1000)
        if isinstance(opt_dist_thr_per_class, list):
            cluster_model = AgglomerativeClustering(n_clusters=None, metric='manhattan', linkage='complete',
                                                    distance_threshold=opt_dist_thr_per_class[class_index])
        else:
            cluster_model = AgglomerativeClustering(n_clusters=None, metric='manhattan', linkage='complete',
                                                    distance_threshold=opt_dist_thr_per_class)

        cluster_model.fit(spk_count_train_clusters[indices])
        # Save the cluster models
        clusters_per_class.append(cluster_model)

    if verbose == 2:
        # TODO: Make figure dependant on the number of classes by a formula
        # Plot the top three levels of the dendrogram
        if n_classes == 10:
            fig, axes = plt.subplots(2, 5, figsize=(6 * n_classes / 2, 12))
        elif n_classes == 26:
            fig, axes = plt.subplots(2, 13, figsize=(6 * n_classes / 2, 12))
        else:
            raise NameError(f'The number of classes {n_classes} is not implemented for the plots')
        fig.suptitle('Hierarchical Clustering Dendrogram', fontsize=22, y=0.94)
        # fig.supxlabel('X axis: Number of points in node (index of the number if not in parenthesis)',fontsize = h + w*0.1,y=0.065)

        for class_index, ax in tqdm(enumerate(axes.flat),
                                    desc='Create the clusters with the selected distance thresholds'):
            plot_dendrogram(cluster_model, truncate_mode='level', p=3, ax=ax)
            ax.set_title('Class {}'.format(class_names[class_index]),
                         fontsize=22)
            # ax[i,j].set_xlabel("Number of points in node",fontsize=h)

        plt.savefig(f'DendrogramPerClass.pdf')
        fig.show()

        print_created_clusters_per_class(clusters_per_class, class_names)

    if verbose == 1:
        string_for_logger = 'Created clusters:\n' + '-' * 75 + '\n'
        for class_index in range(n_classes):
            unique, counts = np.unique(clusters_per_class[class_index].labels_, return_counts=True)
            string_for_logger += f'Clase {class_names[class_index].ljust(15)} \t {dict(zip(unique, counts))}\n' + '-' * 75 + '\n'

        return clusters_per_class, string_for_logger

    return clusters_per_class
