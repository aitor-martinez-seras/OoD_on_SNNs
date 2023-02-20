import numpy as np


def tp_fn_fp_tn_computation(in_or_out_distribution_per_tpr):
    '''
    Function that creates an array with the number of values of tp and fp or fn and tn, depending on if the
    passed array is InD or OD.
    :in_or_out_distribution_per_tpr: array with True if predicted InD and False if predicted OD, for each TPR
    ::return: array with shape (tpr, 2) with the 2 dimensions being tp,fn if passed array is InD, and fp and tn if the passed array is OD
    '''
    tp_fn_fp_tn = np.zeros((len(in_or_out_distribution_per_tpr), 2), dtype='uint16')
    length_array = in_or_out_distribution_per_tpr.shape[1]
    for index, element in enumerate(in_or_out_distribution_per_tpr):
        n_true = int(len(element.nonzero()[0]))
        tp_fn_fp_tn[index, 0] = n_true
        tp_fn_fp_tn[index, 1] = length_array - n_true
    return tp_fn_fp_tn


# ---------------------------------------------
# Metrics for Distances per class approach #
# ---------------------------------------------
def thresholds_per_class_for_each_TPR(n_classes, dist_per_class):
    # Creation of the array with the thresholds for each TPR (class, dist_per_TPR)
    sorted_distances_per_class = [np.sort(x) for x in dist_per_class]
    tpr_range = np.arange(0, 1, 0.01)
    tpr_range[-1] = 0.99999999  # For selecting the last item correctly
    distance_thresholds_test = np.zeros((n_classes, len(tpr_range)))
    for class_index in range(n_classes):
        for index, tpr in enumerate(tpr_range):
            distance_thresholds_test[class_index, index] = sorted_distances_per_class[class_index][
                int(len(sorted_distances_per_class[class_index]) * tpr)
            ]
    return distance_thresholds_test


def compare_distances_per_class_to_distance_thr_per_class(distances_list_per_class, thr_distances_array):
    '''
    Function that creates an array of shape (tpr, InD_or_OD), where tpr has the lenght of the number of steps of the TPR list
    and second dimensions has the total lenght of the distances_list_per_class, and cotains True if its InD and False if is OD
    :distances_list_per_class: list with each element being an array with the distances to avg clusters of one class [array(.), array(.)]
    :thr_distances_array: array of shape (class, dist_for_each_tpr), where first dimension is the class and the second is the distance for the TPR
     corresponding to that position. For example, the TPR = 0.85 corresponds to the 85th position.
    '''
    in_or_out_distribution_per_tpr = np.zeros(
        (len(np.transpose(thr_distances_array)), len(np.concatenate(distances_list_per_class))), dtype=bool)
    for tpr_index, thr_distances_per_class in enumerate(np.transpose(thr_distances_array)):
        in_or_out_distribution_per_tpr[tpr_index] = np.concatenate(
            [dist_one_class < thr_distances_per_class[cls_index] for cls_index, dist_one_class in
             enumerate(distances_list_per_class)])

    return in_or_out_distribution_per_tpr


def computation_distances_per_class_in_or_out_distribution_per_tpr(
        dist_test_per_class, dist_ood_per_class, dist_thresholds
):
    # Creation of the array with True if predicted InD (True) or OD (False)
    in_or_out_distribution_per_tpr_test = compare_distances_per_class_to_distance_thr_per_class(dist_test_per_class,
                                                                                                dist_thresholds)
    in_or_out_distribution_per_tpr_test[0] = np.zeros((in_or_out_distribution_per_tpr_test.shape[1]),
                                                      dtype=bool)  # To fix that one element is True when TPR is 0
    in_or_out_distribution_per_tpr_test[-1] = np.ones((in_or_out_distribution_per_tpr_test.shape[1]),
                                                      dtype=bool)  # To fix that last element is True when TPR is 1
    in_or_out_distribution_per_tpr_ood = compare_distances_per_class_to_distance_thr_per_class(dist_ood_per_class,
                                                                                               dist_thresholds)
    return in_or_out_distribution_per_tpr_test, in_or_out_distribution_per_tpr_ood


def compute_precision_tpr_fpr_for_test_and_ood(dist_test_per_class, dist_ood_per_class, dist_thresholds):
    # Creation of the array with True if predicted InD (True) or OD (False)
    in_or_out_distribution_per_tpr_test = compare_distances_per_class_to_distance_thr_per_class(dist_test_per_class,
                                                                                                dist_thresholds)
    in_or_out_distribution_per_tpr_test[0] = np.zeros((in_or_out_distribution_per_tpr_test.shape[1]),
                                                      dtype=bool)  # To fix that one element is True when TPR is 0
    in_or_out_distribution_per_tpr_test[-1] = np.ones((in_or_out_distribution_per_tpr_test.shape[1]),
                                                      dtype=bool)  # To fix that last element is True when TPR is 1
    in_or_out_distribution_per_tpr_ood = compare_distances_per_class_to_distance_thr_per_class(dist_ood_per_class,
                                                                                               dist_thresholds)

    # Creation of arrays with TP, FN and FP, TN
    tp_fn_test = tp_fn_fp_tn_computation(in_or_out_distribution_per_tpr_test)
    fp_tn_ood = tp_fn_fp_tn_computation(in_or_out_distribution_per_tpr_ood)

    # Computing TPR, FPR and Precision
    tpr_values = tp_fn_test[:, 0] / (tp_fn_test[:, 0] + tp_fn_test[:, 1])
    fpr_values = fp_tn_ood[:, 0] / (fp_tn_ood[:, 0] + fp_tn_ood[:, 1])
    precision = tp_fn_test[:, 0] / (tp_fn_test[:, 0] + fp_tn_ood[:, 0])

    # Eliminating NaN value at TPR = 1
    precision[0] = 1
    return precision, tpr_values, fpr_values


# ---------------------------------------------
# Metrics for Distances all classes at the same time approach #
# ---------------------------------------------
def thresholds_for_each_TPR_distances(distance):
    """
    Creation of the array with the thresholds for each TPR
    """
    sorted_distance = np.sort(distance)
    # Inverse the order to get it correctly (greater the threshold, lower the TPR)
    tpr_range = np.arange(0, 1, 0.01)
    tpr_range[0] = 0.99999999  # For selecting the first item correctly
    distance_thresholds = np.zeros(len(tpr_range))
    for index, tpr in enumerate(tpr_range):
        distance_thresholds[index] = sorted_distance[int(len(sorted_distance) * tpr)]
    return distance_thresholds


def compare_distances_to_distance_thr_one_for_all_classes(distances_evaluating, thr_distances_array):
    '''
    Function that creates an array of shape (tpr, InD_or_OD), where tpr has
    the lenght of the number of steps of the TPR list and second dimensions
    has the total lenght of the distances_evaluating, and cotains True if its InD and False if is ood
    Parameters
    ----------
    distances_evaluating: List of length equal to number of classes, each position being an array,
        Each element of the list are the distances to avg clusters of one class
    thr_distances_array: numpy array
        Array containing the distance for the TPR corresponding to that position.
        For example, the TPR = 0.85 corresponds to the 85th position.

    Returns
    ----------
    in_or_out_distribution_per_tpr: List of arrays
    '''
    in_or_out_distribution_per_tpr = np.zeros((len(thr_distances_array), len(distances_evaluating)), dtype=bool)
    for tpr_index, thr_for_one_tpr in enumerate(thr_distances_array):
        in_or_out_distribution_per_tpr[tpr_index] = np.where(distances_evaluating < thr_for_one_tpr, True, False)

    return in_or_out_distribution_per_tpr


def distance_method_compute_precision_tpr_fpr_for_test_and_ood(dist_test, dist_ood, dist_thresholds):
    # Creation of the array with True if predicted InD (True) or OD (False)
    in_or_out_distribution_per_tpr_test = compare_distances_to_distance_thr_one_for_all_classes(dist_test,
                                                                                                dist_thresholds)
    in_or_out_distribution_per_tpr_test[0] = np.zeros((in_or_out_distribution_per_tpr_test.shape[1]),
                                                      dtype=bool)  # To fix that one element is True when TPR is 0
    in_or_out_distribution_per_tpr_test[-1] = np.ones((in_or_out_distribution_per_tpr_test.shape[1]),
                                                      dtype=bool)  # To fix that last element is True when TPR is 1
    in_or_out_distribution_per_tpr_ood = compare_distances_to_distance_thr_one_for_all_classes(dist_ood,
                                                                                               dist_thresholds)

    # Creation of arrays with TP, FN and FP, TN
    tp_fn_test = tp_fn_fp_tn_computation(in_or_out_distribution_per_tpr_test)
    fp_tn_ood = tp_fn_fp_tn_computation(in_or_out_distribution_per_tpr_ood)

    # Computing TPR, FPR and Precision
    tpr_values = tp_fn_test[:, 0] / (tp_fn_test[:, 0] + tp_fn_test[:, 1])
    fpr_values = fp_tn_ood[:, 0] / (fp_tn_ood[:, 0] + fp_tn_ood[:, 1])
    precision = tp_fn_test[:, 0] / (tp_fn_test[:, 0] + fp_tn_ood[:, 0])

    # Eliminating NaN value at TPR = 1
    precision[0] = 1
    return precision, tpr_values, fpr_values


# ---------------------------------------------
# Metrics for Likelihood all classes at the same time approach #
# ---------------------------------------------
def thresholds_for_each_TPR_likelihood(likelihood):
    """
    Creation of the array with the thresholds for each TPR
    """
    sorted_likelihood = np.sort(likelihood)
    # Inverse the order to get it correctly (greater the threshold, lower the TPR)
    tpr_range = np.arange(0, 1, 0.01)[::-1]
    tpr_range[0] = 0.99999999  # For selecting the first item correctly
    likelihood_thresholds = np.zeros(len(tpr_range))
    for index, tpr in enumerate(tpr_range):
        likelihood_thresholds[index] = sorted_likelihood[int(len(sorted_likelihood) * tpr)]
    return likelihood_thresholds


def compare_likelihood_to_likelihood_thr_one_for_all_classes(likelihoods_evaluating, thr_likelihoods_array):
    '''
    Function that creates an array of shape (tpr, InD_or_OD), where tpr has the lenght of the number of steps of the TPR list
    and second dimensions has the total lenght of the likelihoods_evaluating, and cotains True if its InD and False if is ood
    :likelihoods_evaluating: list with each element being an array with the distances to avg clusters of one class [array(.), array(.)]
    :thr_likelihoods_array: array containing the distance for the TPR
     corresponding to that position. For example, the TPR = 0.85 corresponds to the 85th position.
    '''
    in_or_out_distribution_per_tpr = np.zeros((len(thr_likelihoods_array), len(likelihoods_evaluating)), dtype=bool)
    for tpr_index, thr_for_one_tpr in enumerate(thr_likelihoods_array):
        in_or_out_distribution_per_tpr[tpr_index] = np.where(likelihoods_evaluating > thr_for_one_tpr, True, False)

    return in_or_out_distribution_per_tpr


def computation_likelihood_in_or_out_distribution_per_tpr(likelihood_test, likelihood_ood, likelihood_thresholds):
    # Creation of the array with True if predicted InD (True) or OD (False)
    in_or_out_distribution_per_tpr_test = compare_likelihood_to_likelihood_thr_one_for_all_classes(
        likelihood_test, likelihood_thresholds
    )
    # To fix that one element is True when TPR is 0
    in_or_out_distribution_per_tpr_test[0] = np.zeros(
        (in_or_out_distribution_per_tpr_test.shape[1]), dtype=bool
    )
    # To fix that last element is True when TPR is 1
    in_or_out_distribution_per_tpr_test[-1] = np.ones(
        (in_or_out_distribution_per_tpr_test.shape[1]), dtype=bool
    )
    in_or_out_distribution_per_tpr_ood = compare_likelihood_to_likelihood_thr_one_for_all_classes(
        likelihood_ood, likelihood_thresholds
    )
    return in_or_out_distribution_per_tpr_test, in_or_out_distribution_per_tpr_ood


def likelihood_method_compute_precision_tpr_fpr_for_test_and_ood(likelihood_test, likelihood_ood,
                                                                 likelihood_thresholds):
    # Creation of the array with True if predicted InD (True) or OD (False)
    in_or_out_distribution_per_tpr_test = compare_likelihood_to_likelihood_thr_one_for_all_classes(likelihood_test,
                                                                                                   likelihood_thresholds)
    in_or_out_distribution_per_tpr_test[0] = np.zeros((in_or_out_distribution_per_tpr_test.shape[1]),
                                                      dtype=bool)  # To fix that one element is True when TPR is 0
    in_or_out_distribution_per_tpr_test[-1] = np.ones((in_or_out_distribution_per_tpr_test.shape[1]),
                                                      dtype=bool)  # To fix that last element is True when TPR is 1
    in_or_out_distribution_per_tpr_ood = compare_likelihood_to_likelihood_thr_one_for_all_classes(likelihood_ood,
                                                                                                  likelihood_thresholds)

    # Creation of arrays with TP, FN and FP, TN
    tp_fn_test = tp_fn_fp_tn_computation(in_or_out_distribution_per_tpr_test)
    fp_tn_ood = tp_fn_fp_tn_computation(in_or_out_distribution_per_tpr_ood)

    # Computing TPR, FPR and Precision
    tpr_values = tp_fn_test[:, 0] / (tp_fn_test[:, 0] + tp_fn_test[:, 1])
    fpr_values = fp_tn_ood[:, 0] / (fp_tn_ood[:, 0] + fp_tn_ood[:, 1])
    precision = tp_fn_test[:, 0] / (tp_fn_test[:, 0] + fp_tn_ood[:, 0])

    # Eliminating NaN value at TPR = 1 and other NaN values that may appear due to
    # precision = TP / (TP + FN) = 0 / (0 + 0)
    precision[0] = 1
    # np.nan_to_num(precision, nan=1, copy=False)
    return precision, tpr_values, fpr_values
