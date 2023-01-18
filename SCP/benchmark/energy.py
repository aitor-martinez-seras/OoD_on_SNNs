import numpy as np
import torch


from SCP.utils.metrics import thresholds_for_each_TPR_likelihood, \
    likelihood_method_compute_precision_tpr_fpr_for_test_and_ood
from SCP.benchmark._base import _OODMethod


class EnergyOOD(_OODMethod):

    def __init__(self):
        super().__init__()

    def __call__(self, logits_train, logits_test, logits_ood):
        prelim_results = []
        for temp in [1, 10, 100, 1000, 10000, 100000]:
            # Compute the energies
            energy_train = -(-temp * torch.logsumexp(torch.Tensor(logits_train) / temp, dim=1)).numpy()
            energy_test = -(-temp * torch.logsumexp(torch.Tensor(logits_test) / temp, dim=1)).numpy()
            energy_ood = -(-temp * torch.logsumexp(torch.Tensor(logits_ood) / temp, dim=1)).numpy()
            # Creation of the array with the thresholds for each TPR (class, dist_per_TPR)
            distance_thresholds_train = thresholds_for_each_TPR_likelihood(energy_train)
            # Conmputing precision, tpr and fpr
            precision, tpr_values, fpr_values = likelihood_method_compute_precision_tpr_fpr_for_test_and_ood(
                energy_test, energy_ood, distance_thresholds_train)
            # Appending that when FPR = 1 the TPR is also 1:
            tpr_values_auroc = np.append(tpr_values, 1)
            fpr_values_auroc = np.append(fpr_values, 1)
            # Metrics
            auroc = round(np.trapz(tpr_values_auroc, fpr_values_auroc) * 100, 2)
            aupr = round(np.trapz(precision, tpr_values) * 100, 2)
            fpr95 = round(fpr_values_auroc[95] * 100, 2)
            fpr80 = round(fpr_values_auroc[80] * 100, 2)
            prelim_results.append([auroc, aupr, fpr95, fpr80, temp])

        # Extrac the best result for different temperatures
        prelim_results = np.array(prelim_results)
        index_max = np.argmax(prelim_results[:, 0])
        auroc, aupr, fpr95, fpr80, temp = prelim_results[index_max]
        return auroc, aupr, fpr95, fpr80, temp
