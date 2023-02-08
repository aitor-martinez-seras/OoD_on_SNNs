import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as skmetrics

from .common import find_idx_of_class
from .plots import plot_dendrogram, plot_clusters_performance, plot_dendogram_per_class


def create_string_for_logger(clusters_per_class, class_names) -> str:
    string_for_logger = 'Created clusters:\n' + '-' * 75 + '\n'
    for class_index in range(len(class_names)):
        unique, counts = np.unique(clusters_per_class[class_index].labels_, return_counts=True)
        string_for_logger += f'Clase {class_names[class_index].ljust(15)} \t {dict(zip(unique, counts))}\n' + '-' * 75 + '\n'
    return string_for_logger


def print_created_clusters_per_class(clusters_per_class, class_names):
    print('')
    print('Created clusters:')
    print('-' * 75)
    # Printing the cluster composition for each class
    for cl_ind, clusters_one_class in enumerate(clusters_per_class):
        unique, counts = np.unique(clusters_one_class.labels_, return_counts=True)
        print('Clase', class_names[cl_ind].ljust(15), '\t', dict(zip(unique, counts)))
        print('-' * 75)


def bic_score(X, labels):
    """
    BIC score for the goodness of fit of clusters.
    This Python function is directly translated from the GoLang code made by the author of the paper.
    The original code is available here:
    https://github.com/bobhancock/goxmeans/blob/a78e909e374c6f97ddd04a239658c7c5b7365e5c/km.go#L778
    """
    import math
    n_points = len(labels)
    n_clusters = len(set(labels))
    n_dimensions = X.shape[1]

    n_parameters = (n_clusters - 1) + (n_dimensions * n_clusters) + 1

    loglikelihood = 0
    for label_name in set(labels):
        X_cluster = X[labels == label_name]
        n_points_cluster = len(X_cluster)
        centroid = np.mean(X_cluster, axis=0)
        variance = np.sum((X_cluster - centroid) ** 2) / (len(X_cluster) - 1)
        loglikelihood += \
            n_points_cluster * np.log(n_points_cluster) \
            - n_points_cluster * np.log(n_points) \
            - n_points_cluster * n_dimensions / 2 * np.log(2 * math.pi * variance) \
            - (n_points_cluster - 1) / 2

    bic = loglikelihood - (n_parameters / 2) * np.log(n_points)
    return bic


def aggregation_per_class_and_cluster(spike_counts, preds, clusters_per_class, n_classes,
                                      option='median', n_samples=None):
    """
    Function that receives the counts and the predictions of a subset and the
    cluster objects per class of that subset and outputs a list of arrays
    with the mean or median hidden count vector of each cluster of each class.
    That is, creates the centroids.
    """
    # For every class
    aggregation_per_class = []
    for class_index in range(n_classes):
        # Calculation of the array of counts that corresponds to a class
        if n_samples is None:
            indices = find_idx_of_class(class_index, preds)
        else:
            indices = find_idx_of_class(class_index, preds, n_samples)
        spike_counts_one_class = spike_counts[indices]
        # For every cluster of the class, compute the median
        aggregation_per_cluster = []
        for cluster_index in np.unique(clusters_per_class[class_index].labels_):
            # We can compute the mean or the median of the neuron counts for that cluster
            if option == 'median':
                aggregation_per_cluster.append(
                    np.median(
                        spike_counts_one_class[np.where(clusters_per_class[class_index].labels_ == cluster_index)[0]],
                        axis=0
                    )
                )
            elif option == 'mean':
                aggregation_per_cluster.append(
                    np.mean(
                        spike_counts_one_class[np.where(clusters_per_class[class_index].labels_ == cluster_index)[0]],
                        axis=0
                    )
                )
        aggregation_per_cluster = np.array(aggregation_per_cluster)
        # Save all the cluster averages to the list of the classes
        aggregation_per_class.append(aggregation_per_cluster)
    return aggregation_per_class


def distance_to_clusters_averages(spike_counts, predictions, aggregation_per_class, n_classes):
    """
    Function that computes the distance of the introduced array to the cluster's averages of the predicted class
    Takes the counts and the predictions of each sample of a subset (not ordered by class) and
    the averages of each cluster and class
    :returns distance of each sample to the cluster average for each class
    """
    # Order array by predicted class
    spike_counts_per_class = []
    for class_index in range(n_classes):
        spike_counts_per_class.append(spike_counts[find_idx_of_class(class_index, predictions)])
    # Compute the pairwise distances per predicted class
    distances_per_class = []
    closest_clusters_per_class = []
    for class_index, frecs_one_class in enumerate(spike_counts_per_class):
        if frecs_one_class.size == 0:
            distances_per_class.append([])
            closest_clusters_per_class.append([])
        else:
            pairwise_dist = skmetrics.pairwise_distances(
                frecs_one_class, aggregation_per_class[class_index], metric='manhattan'
            )
            distances_per_class.append(np.min(pairwise_dist, axis=1))
            closest_clusters_per_class.append(np.argmin(pairwise_dist, axis=1))
    return distances_per_class, closest_clusters_per_class


def select_distance_threshold_for_one_class(clustering_performance_scores) -> int:
    """
    Return the index of the greatest clustering performance distance threshold
    """
    # Iterate the inverted array to catch the smallest distance value with the greatest silhouette score
    max_score = 0
    max_index = 0
    for idx, current_score in enumerate(clustering_performance_scores):
        # Store the greatest value we encounter traveling the curve
        # Only update the value if it is greater, not if it equal
        if current_score > max_score:
            max_index = idx
            max_score = current_score
    return max_index


def select_best_distance_threshold_for_each_class(
        class_names, possible_distance_thrs, size, preds_train, spk_count_train, performance_measuring_method,
        name, verbose
):
    n_classes = len(class_names)
    clustering_performance_scores_for_all_possible_thresholds_per_class = []
    clustering_performance_scores_for_selected_thresholds_per_class = []
    selected_distance_thrs_per_class = []

    # Select performance measuring method
    if performance_measuring_method == 'silhouette':
        cluster_performance_measuring_function = skmetrics.silhouette_score

    elif performance_measuring_method == 'bic':
        cluster_performance_measuring_function = bic_score

    elif performance_measuring_method == 'calinski':
        cluster_performance_measuring_function = skmetrics.calinski_harabasz_score
        performance_measuring_method = 'calinski-harabasz'
    else:
        raise NameError(f'Wrong option selected for measuring performance of the clustering. '
                        f'Selected {performance_measuring_method}')

    for class_index in tqdm(range(n_classes), desc=f'Computing {performance_measuring_method}'
                                                   f' score for various distance thresholds'):
        clustering_performance_scores = []
        for threshold in possible_distance_thrs:
            indices = find_idx_of_class(class_index, preds_train, size)
            cluster_model = AgglomerativeClustering(
                n_clusters=None, metric='manhattan', linkage='average', distance_threshold=threshold
            )
            try:
                cluster_model.fit(spk_count_train[indices])
                # cluster_labels.append(cluster_model.labels_)
            except ValueError as e:   # Handle the case that one class has no representation in the training samples
                print(f'Error probably caused by the lack of training samples for class index {class_index}')
                raise e
            try:  # TODO: We may need to modify this call in case another function is used
                clustering_performance_scores.append(
                    cluster_performance_measuring_function(
                        spk_count_train[indices], cluster_model.labels_, metric='manhattan'
                    )
                )
            except ValueError:
                clustering_performance_scores.append(0)

        clustering_performance_scores_for_all_possible_thresholds_per_class.append(clustering_performance_scores)

        # Determine the best method selecting the index where the max performance is obtained
        max_index = select_distance_threshold_for_one_class(clustering_performance_scores)

        # Append the distance threshold to a list where they are going to be stored, one for each class
        selected_distance_thrs_per_class.append(possible_distance_thrs[max_index])
        clustering_performance_scores_for_selected_thresholds_per_class.append(
            clustering_performance_scores[max_index]
        )

    # Plot the performance score for every distance threshold
    if verbose == 2:
        print('Selected distance thresholds:\n', [round(i, 3) for i in selected_distance_thrs_per_class])
        plot_clusters_performance(
            class_names,
            clustering_performance_scores_for_all_possible_thresholds_per_class,
            possible_distance_thrs,
            clustering_performance_scores_for_selected_thresholds_per_class,
            selected_distance_thrs_per_class,
            name, performance_measuring_method, save=True
        )

    return selected_distance_thrs_per_class, clustering_performance_scores_for_selected_thresholds_per_class


def create_clusters_per_class_based_on_distance_threshold(
        class_names, preds_train, spk_count_train, selected_distance_thrs_per_class
):
    n_classes = len(class_names)
    # Create the clusters by extracting the labels for every sample
    clusters_per_class = []
    for class_index in range(n_classes):
        indices = find_idx_of_class(class_index, preds_train, 1000)
        if isinstance(selected_distance_thrs_per_class, list):
            cluster_model = AgglomerativeClustering(n_clusters=None, metric='manhattan', linkage='complete',
                                                    distance_threshold=selected_distance_thrs_per_class[class_index])
        else:
            cluster_model = AgglomerativeClustering(n_clusters=None, metric='manhattan', linkage='complete',
                                                    distance_threshold=selected_distance_thrs_per_class)

        cluster_model.fit(spk_count_train[indices])
        clusters_per_class.append(cluster_model)
    return clusters_per_class


# Possible refactorization: enable multiprocessing, as each class is independent
def create_clusters(preds_train, spk_count_train, class_names, size=1000,
                    distance_for_clustering=None, verbose=2, name='', performance_measuring_method='silhouette'):
    """
    Verbose = 0 -> No prints and plots neither loggin info
    verbose = 1 -> Returns loggin info only
    Verbose = 2 -> Logging info and plots
    """
    # Select a distance threshold for each class
    if distance_for_clustering is None:  # in case it is not defined
        distance_for_clustering = (800, 3000)
    possible_distance_thrs = np.linspace(distance_for_clustering[0], distance_for_clustering[1], 50)

    # Select the best performing cluster configuration for each class based on a performance metric
    selected_distance_thrs_per_class, clustering_performance_scores_for_selected_thresholds_per_class = select_best_distance_threshold_for_each_class(
        class_names, possible_distance_thrs, size, preds_train, spk_count_train, performance_measuring_method,
        name, verbose
    )

    # Create the clusters for every class
    clusters_per_class = create_clusters_per_class_based_on_distance_threshold(
        class_names, preds_train, spk_count_train, selected_distance_thrs_per_class
    )

    if verbose == 2:
        plot_dendogram_per_class(class_names, clusters_per_class, name, save=True)

    if verbose == 0:
        return clusters_per_class
    else:
        string_for_logger = create_string_for_logger(clusters_per_class, class_names)
        return clusters_per_class, string_for_logger


def silhouette_score_log(clusters_per_class, preds_train, spk_count_train, size):
    scores = []
    for class_index, cluster_model in enumerate(clusters_per_class):
        indices = find_idx_of_class(class_index, preds_train, 1000)
        scores.append(round(
            skmetrics.silhouette_score(spk_count_train[indices], cluster_model.labels_, metric='manhattan'), 3
        ))
    return scores
