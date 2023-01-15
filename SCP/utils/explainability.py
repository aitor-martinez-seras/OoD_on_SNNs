import numpy as np
import matplotlib as plt


def order_array_of_samples_to_per_class_list(array, preds, n_samples=0):
    list_per_class = []
    for cl_index in range(len(train_data.classes)):
        indices = searchIndicesOfClass(cl_index, preds, n_samples)
        list_per_class.append(array[indices])
    return list_per_class


def compute_reconstruction(spk_counts, weights):
    if isinstance(weights, list):
        for l, w_l in enumerate(weights):
            if l == 0:
                reconst = np.matmul(spk_counts, w_l)
            else:
                reconst = np.matmul(reconst, w_l)
    else:
        reconst = np.matmul(spk_counts, weights)
    return reconst


def compute_reconstruction_per_class(spk_frec_per_class, weights):
    reconst_per_class = []
    for spikes_one_class in spk_frec_per_class:
        reconst = compute_reconstruction_n_layers(spikes_one_class, weights)
        reconst_per_class.append(reconst)
    return reconst_per_class


def compute_reconstruction_n_layers(spk_counts, weights):
    if isinstance(weights, list):
        for l, w_l in enumerate(weights):
            if l == 0:
                reconst = np.matmul(spk_counts, w_l)
            else:
                reconst = np.matmul(reconst, w_l)
    else:
        reconst = np.matmul(spk_counts, weights)

    return reconst


def extract_positive_part_per_class(list_per_class):
    positive_part_list = []
    for array_one_class in list_per_class:
        positive_part_list.append(np.where(array_one_class > 0, array_one_class, 0))
    return positive_part_list


def rearrange_to_ftmaps(spk_count, ftmaps_shape=(50, 8, 8)):
    ch, h, w = ftmaps_shape
    unflatten = torch.nn.Unflatten(-1, (ch, h, w))
    return unflatten(torch.from_numpy(spk_count)).numpy()


def rearrange_to_ftmaps_per_class(spk_count_per_class, ftmaps_shape=(50, 8, 8)):
    ftmaps_per_class = []
    for spk_one_class in spk_count_per_class:
        ftmaps_per_class.append(rearrange_to_ftmaps(spk_one_class, ftmaps_shape))
    return ftmaps_per_class


def auroc_aupr(d_train_per_class, d_test_per_class, d_ood_per_class):
    # Creation of the array with the thresholds for each TPR (class, dist_per_TPR)
    d_thresholds_train = thresholds_per_class_for_each_TPR(d_train_per_class)
    # Conmputing precision, tpr and fpr
    precision, tpr_values, fpr_values = compute_precision_tpr_fpr_for_test_and_ood(
        d_test_per_class, d_ood_per_class, d_thresholds_train
    )
    # Appending that when FPR = 1 the TPR is also 1:
    tpr_val_auroc = np.append(tpr_values, 1)
    fpr_val_auroc = np.append(fpr_values, 1)

    # AUROC
    auroc = np.trapz(tpr_val_auroc, fpr_val_auroc)

    # AUPR
    aupr = np.trapz(precision, tpr_values)

    print('-' * 30)
    print(f'Results')
    print('-' * 30)
    print(f'AUROC = {auroc * 100:.3f} %')
    print(f'AUPR  = {aupr * 100:.3f} %')
    print('-' * 30)
    print(f'FPR at 95% TPR: {round(fpr_val_auroc[95] * 100, 2)}%')
    print(
        f'Threshold mean and std for all classes at 95% TPR: {d_thresholds_train[:, 95].mean():.2f}, {d_thresholds_train[:, 95].std():.2f}')
    print('-' * 30, '\n')

    return d_thresholds_train


def create_spk_count(model, device, ood_dict, od_dataset, mnist_c_opt='zigzag', conv_spikes=False):
    if od_dataset == 'MNIST-C':
        test_loader_ood = ood_dict[od_dataset](
            BATCH_SIZE,
            test_only=True,
            option=mnist_c_opt
        )
    else:
        test_loader_ood = ood_dict[od_dataset](BATCH_SIZE, test_only=True)

    # Extract logits and hidden spikes
    if selected_model == 'ConvNet':
        if conv_spikes is True:
            accuracy_ood, preds_ood, logits_ood, _spk_count_ood, _conv_spk_ood = test(model, device, test_loader_ood,
                                                                                      return_logits=True,
                                                                                      return_conv_spikes=True)
            print(f'Accuracy for the ood dataset {od_dataset} is {accuracy_ood:.3f} %')
            return test_loader_ood, preds_ood, np.sum(_spk_count_ood, axis=0, dtype='uint16'), np.sum(_conv_spk_ood,
                                                                                                      axis=0,
                                                                                                      dtype='uint16')
        else:
            accuracy_ood, preds_ood, logits_ood, _spk_count_ood = test(model, device, test_loader_ood,
                                                                       return_logits=True, return_conv_spikes=False)
    else:
        accuracy_ood, preds_ood, logits_ood, _spk_count_ood = test(model, device, test_loader_ood, return_logits=True,
                                                                   return_conv_spikes=False)
    print(f'Accuracy for the ood dataset {od_dataset} is {accuracy_ood:.3f} %')

    return test_loader_ood, preds_ood, np.sum(_spk_count_ood, axis=0, dtype='uint16')