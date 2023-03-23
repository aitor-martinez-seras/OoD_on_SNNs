import torch


from SCP.utils.metrics import thresholds_for_each_TPR_likelihood, \
    likelihood_method_compute_precision_tpr_fpr_for_test_and_ood
from SCP.detection._base import _OODMethod


class EnergyOOD(_OODMethod):

    def __init__(self):
        super().__init__()

    def __call__(self, logits_train, logits_test, logits_ood, save_histogram=False, name='', *args, **kwargs):
        temp = 1
        # Compute the energies
        energy_train = -(-temp * torch.logsumexp(torch.Tensor(logits_train) / temp, dim=1)).numpy()
        energy_test = -(-temp * torch.logsumexp(torch.Tensor(logits_test) / temp, dim=1)).numpy()
        energy_ood = -(-temp * torch.logsumexp(torch.Tensor(logits_ood) / temp, dim=1)).numpy()

        if save_histogram:
            super().save_histogram_fig(energy_train, energy_test, energy_ood, name=f'{name}_Energy_temp{temp}')

        # Creation of the array with the thresholds for each TPR (class, dist_per_TPR)
        likelihood_thresholds_train = thresholds_for_each_TPR_likelihood(energy_train)

        # Computing precision, tpr and fpr
        self.precision, self.tpr_values, self.fpr_values = likelihood_method_compute_precision_tpr_fpr_for_test_and_ood(
            energy_test, energy_ood, likelihood_thresholds_train)

        auroc, aupr, fpr95, fpr80 = super().compute_metrics()

        return auroc, aupr, fpr95, fpr80, temp
