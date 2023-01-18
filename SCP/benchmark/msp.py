import numpy as np
import scipy.special as scysp

from SCP.utils.metrics import thresholds_for_each_TPR_likelihood, \
    likelihood_method_compute_precision_tpr_fpr_for_test_and_ood
from SCP.benchmark._base import _OODMethod


class MSP(_OODMethod):

    def __init__(self):
        super().__init__()

    def __call__(self, logits_train, logits_test, logits_ood):
        # Softmax
        softmax_train_winners = np.max(scysp.softmax(logits_train, axis=1), axis=1)
        softmax_test_winners = np.max(scysp.softmax(logits_test, axis=1), axis=1)
        softmax_ood_winners = np.max(scysp.softmax(logits_ood, axis=1), axis=1)

        # Creation of the array with the thresholds for each TPR (class, dist_per_TPR)
        likelihood_thresholds_train = thresholds_for_each_TPR_likelihood(softmax_train_winners)

        # Computing precision, tpr and fpr
        self.precision, self.tpr_values, self.fpr_values = likelihood_method_compute_precision_tpr_fpr_for_test_and_ood(
            softmax_test_winners, softmax_ood_winners, likelihood_thresholds_train
        )
        return super().compute_metrics()


def baseline_method(logits_train, logits_test, logits_ood):

    # Softmax
    softmax_train_winners = np.max(scysp.softmax(logits_train, axis=1), axis=1)
    softmax_test_winners = np.max(scysp.softmax(logits_test, axis=1), axis=1)
    softmax_ood_winners = np.max(scysp.softmax(logits_ood, axis=1), axis=1)

    # Creation of the array with the thresholds for each TPR (class, dist_per_TPR)
    likelihood_thresholds_train = thresholds_for_each_TPR_likelihood(softmax_train_winners)

    # Computing precision, tpr and fpr
    precision, tpr_values, fpr_values = likelihood_method_compute_precision_tpr_fpr_for_test_and_ood(
        softmax_test_winners, softmax_ood_winners, likelihood_thresholds_train
    )

    # Appending that when FPR = 1 the TPR is also 1:
    tpr_values_auroc = np.append(tpr_values, 1)
    fpr_values_auroc = np.append(fpr_values, 1)

    # Metrics
    auroc = round(np.trapz(tpr_values_auroc, fpr_values_auroc) * 100, 2)
    aupr = round(np.trapz(precision, tpr_values) * 100, 2)
    fpr95 = round(fpr_values_auroc[95] * 100, 2)
    fpr80 = round(fpr_values_auroc[80] * 100, 2)

    return auroc, aupr, fpr95, fpr80
