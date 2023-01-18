import numpy as np
import scipy.special as scysp

from SCP.utils.metrics import thresholds_for_each_TPR_likelihood, \
    likelihood_method_compute_precision_tpr_fpr_for_test_and_ood
from SCP.benchmark._base import _OODMethod


class ODIN(_OODMethod):

    def __init__(self):
        super().__init__()

    def __call__(self, logits_train, logits_test, logits_ood):
        prelim_results = []
        for temp in [1, 10, 100, 1000]:

            # Temperature scaling the softmax
            temp_softmax_train = scysp.softmax(logits_train / temp, axis=1)
            temp_softmax_test = scysp.softmax(logits_test / temp, axis=1)
            temp_softmax_ood = scysp.softmax(logits_ood / temp, axis=1)

            # Getting only the winners
            temp_softmax_train_winners = np.max(temp_softmax_train, axis=1)
            temp_softmax_test_winners = np.max(temp_softmax_test, axis=1)
            temp_softmax_ood_winners = np.max(temp_softmax_ood, axis=1)

            # Creation of the array with the thresholds for each TPR (class, dist_per_TPR)
            likelihood_thresholds_train = thresholds_for_each_TPR_likelihood(temp_softmax_train_winners)

            # Conmputing precision, tpr and fpr
            self.precision, self.tpr_values, self.fpr_values = likelihood_method_compute_precision_tpr_fpr_for_test_and_ood(
                temp_softmax_test_winners, temp_softmax_ood_winners, likelihood_thresholds_train)

            auroc, aupr, fpr95, fpr80 = super().compute_metrics()
            prelim_results.append([auroc, aupr, fpr95, fpr80, temp])

        # Extrac the best result for different temperatures
        prelim_results = np.array(prelim_results)
        index_max = np.argmax(prelim_results[:, 0])
        auroc, aupr, fpr95, fpr80, temp = prelim_results[index_max]

        return auroc, aupr, fpr95, fpr80, temp
