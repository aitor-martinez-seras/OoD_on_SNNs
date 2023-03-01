import numpy as np
import scipy.special as scysp

from SCP.utils.metrics import thresholds_for_each_TPR_likelihood, \
    likelihood_method_compute_precision_tpr_fpr_for_test_and_ood
from SCP.detection._base import _OODMethod


class MSP(_OODMethod):

    def __init__(self):
        super().__init__()

    def __call__(self, logits_train, logits_test, logits_ood, save_histogram=False, name='', *args, **kwargs):
        # Softmax
        softmax_train_winners = np.max(scysp.softmax(logits_train, axis=1), axis=1)
        softmax_test_winners = np.max(scysp.softmax(logits_test, axis=1), axis=1)
        softmax_ood_winners = np.max(scysp.softmax(logits_ood, axis=1), axis=1)

        if save_histogram:
            super().save_histogram_fig(
                softmax_train_winners, softmax_test_winners, softmax_ood_winners,
                name=f'{name}_baseline'
            )

        # Creation of the array with the thresholds for each TPR (class, dist_per_TPR)
        likelihood_thresholds_train = thresholds_for_each_TPR_likelihood(softmax_train_winners)
        
        # Computing precision, tpr and fpr
        self.precision, self.tpr_values, self.fpr_values = likelihood_method_compute_precision_tpr_fpr_for_test_and_ood(
            softmax_test_winners, softmax_ood_winners, likelihood_thresholds_train
        )
        return super().compute_metrics()

