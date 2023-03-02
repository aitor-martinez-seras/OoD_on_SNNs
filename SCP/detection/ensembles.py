import matplotlib.pyplot as plt
import scipy.special as scysp
import numpy as np
import torch

from SCP.detection._base import _OODMethod
from SCP.utils.metrics import thresholds_per_class_for_each_TPR, compute_precision_tpr_fpr_for_test_and_ood, \
    computation_likelihood_in_or_out_distribution_per_tpr, \
    computation_distances_per_class_in_or_out_distribution_per_tpr, \
    thresholds_for_each_TPR_likelihood, likelihood_method_compute_precision_tpr_fpr_for_test_and_ood, \
    tp_fn_fp_tn_computation

from SCP.utils.common import len_of_list_per_class


class EnsembleOdinSCP(_OODMethod):

    def __init__(self):
        super().__init__()

    def __call__(self, distances_train_per_class, distances_test_per_class, distances_ood_per_class,
                 logits_train, logits_test, logits_ood,
                 save_histogram=False, name='', class_names=None, preds_ood=None, *args, **kwargs):
        # ----- Odin -----
        # Get the best temperature
        prelim_results = []
        possible_temps = [1, 10, 100, 1000]
        for temp in possible_temps:

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
            # prec_tpr_fpr.append((self.precision, self.tpr_values, self.fpr_values))

        # Extrac the best AUROC result between the different temperatures
        prelim_results = np.array(prelim_results)
        index_max = np.argmax(prelim_results[:, 0])

        temp = possible_temps[index_max]

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

        odin_in_or_ood_per_tpr_test, odin_in_or_ood_per_tpr_ood = computation_likelihood_in_or_out_distribution_per_tpr(
            temp_softmax_test_winners, temp_softmax_ood_winners, likelihood_thresholds_train
        )

        # ----- SCP -----
        # Creation of the array with the thresholds for each TPR (class, dist_per_TPR)
        distance_thresholds_train = thresholds_per_class_for_each_TPR(
            len(class_names), distances_train_per_class
        )

        scp_in_or_ood_per_tpr_test, scp_in_or_ood_per_tpr_ood = computation_distances_per_class_in_or_out_distribution_per_tpr(
            distances_test_per_class, distances_ood_per_class,  distance_thresholds_train
        )

        # ----- Ensemble -----
        sum_test = scp_in_or_ood_per_tpr_test + odin_in_or_ood_per_tpr_test
        sum_ood = scp_in_or_ood_per_tpr_ood + odin_in_or_ood_per_tpr_ood
        in_or_ood_per_tpr_test = np.where(sum_test >= 1, 1, 0)
        in_or_ood_per_tpr_ood = np.where(sum_ood >= 1, 1, 0)

        # Metrics
        # Creation of arrays with TP, FN and FP, TN
        tp_fn_test = tp_fn_fp_tn_computation(in_or_ood_per_tpr_test)
        fp_tn_ood = tp_fn_fp_tn_computation(in_or_ood_per_tpr_ood)

        # Computing TPR, FPR and Precision
        self.tpr_values = tp_fn_test[:, 0] / (tp_fn_test[:, 0] + tp_fn_test[:, 1])
        self.fpr_values = fp_tn_ood[:, 0] / (fp_tn_ood[:, 0] + fp_tn_ood[:, 1])
        self.precision = tp_fn_test[:, 0] / (tp_fn_test[:, 0] + fp_tn_ood[:, 0])
        # Eliminating NaN value at TPR = 1
        self.precision[0] = 1

        if save_histogram:
            self.plot_in_or_ood(
                odin_in_or_ood_per_tpr_test, odin_in_or_ood_per_tpr_ood,
                scp_in_or_ood_per_tpr_test, scp_in_or_ood_per_tpr_ood,
                name=f'{name}_Ensemble_OdinSCP',
            )

        auroc, aupr, fpr95, fpr80 = super().compute_metrics()

        return auroc, aupr, fpr95, fpr80, temp

    def plot_in_or_ood(self, odin_in_or_ood_per_tpr_test, odin_in_or_ood_per_tpr_ood,
                       scp_in_or_ood_per_tpr_test, scp_in_or_ood_per_tpr_ood, name):
        fig, axes = plt.subplots(2, 2, figsize=(12, 4), tight_layout=True)

        axes[0, 0].matshow(odin_in_or_ood_per_tpr_test)
        axes[0, 0].set_title('Odin test')

        axes[0, 1].matshow(odin_in_or_ood_per_tpr_ood)
        axes[0, 1].set_title('Odin ood')

        axes[1, 0].matshow(scp_in_or_ood_per_tpr_test)
        axes[1, 0].set_title('SCP test')

        axes[1, 1].matshow(scp_in_or_ood_per_tpr_ood)
        axes[1, 1].set_title('SCP ood')

        fig.savefig(f'{name}.pdf')
        plt.close(fig)


class EnsembleEnergySCP(_OODMethod):

    def __init__(self):
        super().__init__()

    def __call__(self, distances_train_per_class, distances_test_per_class, distances_ood_per_class,
                 logits_train, logits_test, logits_ood,
                 save_histogram=False, name='', class_names=None, preds_ood=None, *args, **kwargs):
        # ----- Odin -----
        # Get the best temperature
        # ----- Energy -----
        temp_energy = 1
        # Compute the energies
        energy_train = -(-temp_energy * torch.logsumexp(torch.Tensor(logits_train) / temp_energy, dim=1)).numpy()
        energy_test = -(-temp_energy * torch.logsumexp(torch.Tensor(logits_test) / temp_energy, dim=1)).numpy()
        energy_ood = -(-temp_energy * torch.logsumexp(torch.Tensor(logits_ood) / temp_energy, dim=1)).numpy()

        # Creation of the array with the thresholds for each TPR (class, dist_per_TPR)
        likelihood_thresholds_train = thresholds_for_each_TPR_likelihood(energy_train)

        # Creation of the array with True if predicted InD (True) or OD (False)
        energy_in_or_ood_per_tpr_test, energy_in_or_ood_per_tpr_ood = computation_likelihood_in_or_out_distribution_per_tpr(
            energy_test, energy_ood, likelihood_thresholds_train
        )

        # ----- SCP -----
        # Creation of the array with the thresholds for each TPR (class, dist_per_TPR)
        distance_thresholds_train = thresholds_per_class_for_each_TPR(
            len(class_names), distances_train_per_class
        )

        scp_in_or_ood_per_tpr_test, scp_in_or_ood_per_tpr_ood = computation_distances_per_class_in_or_out_distribution_per_tpr(
            distances_test_per_class, distances_ood_per_class,  distance_thresholds_train
        )

        # ----- Ensemble -----
        sum_test = scp_in_or_ood_per_tpr_test + energy_in_or_ood_per_tpr_test
        sum_ood = scp_in_or_ood_per_tpr_ood + energy_in_or_ood_per_tpr_ood
        in_or_ood_per_tpr_test = np.where(sum_test >= 1, 1, 0)
        in_or_ood_per_tpr_ood = np.where(sum_ood >= 1, 1, 0)

        # Metrics
        # Creation of arrays with TP, FN and FP, TN
        tp_fn_test = tp_fn_fp_tn_computation(in_or_ood_per_tpr_test)
        fp_tn_ood = tp_fn_fp_tn_computation(in_or_ood_per_tpr_ood)

        # Computing TPR, FPR and Precision
        self.tpr_values = tp_fn_test[:, 0] / (tp_fn_test[:, 0] + tp_fn_test[:, 1])
        self.fpr_values = fp_tn_ood[:, 0] / (fp_tn_ood[:, 0] + fp_tn_ood[:, 1])
        self.precision = tp_fn_test[:, 0] / (tp_fn_test[:, 0] + fp_tn_ood[:, 0])
        # Eliminating NaN value at TPR = 1
        self.precision[0] = 1

        if save_histogram:
            self.plot_in_or_ood(
                energy_in_or_ood_per_tpr_test, energy_in_or_ood_per_tpr_ood,
                scp_in_or_ood_per_tpr_test, scp_in_or_ood_per_tpr_ood,
                name=f'{name}_Ensemble_OdinSCP',
            )

        auroc, aupr, fpr95, fpr80 = super().compute_metrics()

        return auroc, aupr, fpr95, fpr80, temp_energy


class EnsembleOdinEnergy(_OODMethod):

    def __init__(self):
        super().__init__()

    def __call__(self, logits_train, logits_test, logits_ood,
                 save_histogram=False, name='', class_names=None, preds_ood=None, *args, **kwargs):
        # ----- Odin -----
        # Get the best temperature
        prelim_results = []
        possible_temps_odin = [1, 10, 100, 1000]
        for temp_odin in possible_temps_odin:

            # Temperature scaling the softmax
            temp_softmax_train = scysp.softmax(logits_train / temp_odin, axis=1)
            temp_softmax_test = scysp.softmax(logits_test / temp_odin, axis=1)
            temp_softmax_ood = scysp.softmax(logits_ood / temp_odin, axis=1)

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
            prelim_results.append([auroc, aupr, fpr95, fpr80, temp_odin])
            # prec_tpr_fpr.append((self.precision, self.tpr_values, self.fpr_values))

        # Extrac the best AUROC result between the different temperatures
        prelim_results = np.array(prelim_results)
        index_max = np.argmax(prelim_results[:, 0])

        temp_odin = possible_temps_odin[index_max]

        # Temperature scaling the softmax
        temp_softmax_train = scysp.softmax(logits_train / temp_odin, axis=1)
        temp_softmax_test = scysp.softmax(logits_test / temp_odin, axis=1)
        temp_softmax_ood = scysp.softmax(logits_ood / temp_odin, axis=1)

        # Getting only the winners
        temp_softmax_train_winners = np.max(temp_softmax_train, axis=1)
        temp_softmax_test_winners = np.max(temp_softmax_test, axis=1)
        temp_softmax_ood_winners = np.max(temp_softmax_ood, axis=1)

        # Creation of the array with the thresholds for each TPR (class, dist_per_TPR)
        likelihood_thresholds_train = thresholds_for_each_TPR_likelihood(temp_softmax_train_winners)

        odin_in_or_ood_per_tpr_test, odin_in_or_ood_per_tpr_ood = computation_likelihood_in_or_out_distribution_per_tpr(
            temp_softmax_test_winners, temp_softmax_ood_winners, likelihood_thresholds_train
        )

        # ----- Energy -----
        temp_energy = 1
        # Compute the energies
        energy_train = -(-temp_energy * torch.logsumexp(torch.Tensor(logits_train) / temp_energy, dim=1)).numpy()
        energy_test = -(-temp_energy * torch.logsumexp(torch.Tensor(logits_test) / temp_energy, dim=1)).numpy()
        energy_ood = -(-temp_energy * torch.logsumexp(torch.Tensor(logits_ood) / temp_energy, dim=1)).numpy()

        # Creation of the array with the thresholds for each TPR (class, dist_per_TPR)
        likelihood_thresholds_train = thresholds_for_each_TPR_likelihood(energy_train)

        # Creation of the array with True if predicted InD (True) or OD (False)
        energy_in_or_ood_per_tpr_test, energy_in_or_ood_per_tpr_ood = computation_likelihood_in_or_out_distribution_per_tpr(
            energy_test, energy_ood, likelihood_thresholds_train
        )

        # ----- Ensemble -----
        sum_test = energy_in_or_ood_per_tpr_test + odin_in_or_ood_per_tpr_test
        sum_ood = energy_in_or_ood_per_tpr_ood + odin_in_or_ood_per_tpr_ood
        in_or_ood_per_tpr_test = np.where(sum_test >= 1, 1, 0)
        in_or_ood_per_tpr_ood = np.where(sum_ood >= 1, 1, 0)

        # Metrics
        # Creation of arrays with TP, FN and FP, TN
        tp_fn_test = tp_fn_fp_tn_computation(in_or_ood_per_tpr_test)
        fp_tn_ood = tp_fn_fp_tn_computation(in_or_ood_per_tpr_ood)

        # Computing TPR, FPR and Precision
        self.tpr_values = tp_fn_test[:, 0] / (tp_fn_test[:, 0] + tp_fn_test[:, 1])
        self.fpr_values = fp_tn_ood[:, 0] / (fp_tn_ood[:, 0] + fp_tn_ood[:, 1])
        self.precision = tp_fn_test[:, 0] / (tp_fn_test[:, 0] + fp_tn_ood[:, 0])
        # Eliminating NaN value at TPR = 1
        self.precision[0] = 1

        if save_histogram:
            self.plot_in_or_ood(
                odin_in_or_ood_per_tpr_test, odin_in_or_ood_per_tpr_ood,
                energy_in_or_ood_per_tpr_test, energy_in_or_ood_per_tpr_ood,
                name=f'{name}_Ensemble_OdinSCP',
            )

        auroc, aupr, fpr95, fpr80 = super().compute_metrics()

        return auroc, aupr, fpr95, fpr80, temp_odin

    def plot_in_or_ood(self, odin_in_or_ood_per_tpr_test, odin_in_or_ood_per_tpr_ood,
                       energy_in_or_ood_per_tpr_test, energy_in_or_ood_per_tpr_ood, name):
        fig, axes = plt.subplots(2, 2, figsize=(12, 4), tight_layout=True)

        axes[0, 0].matshow(odin_in_or_ood_per_tpr_test)
        axes[0, 0].set_title('Odin test')

        axes[0, 1].matshow(odin_in_or_ood_per_tpr_ood)
        axes[0, 1].set_title('Odin ood')

        axes[1, 0].matshow(energy_in_or_ood_per_tpr_test)
        axes[1, 0].set_title('Energy test')

        axes[1, 1].matshow(energy_in_or_ood_per_tpr_ood)
        axes[1, 1].set_title('Energy ood')

        fig.savefig(f'{name}.pdf')
        plt.close(fig)


if __name__ == "__main__":
    ensemble_odin_scp = EnsembleOdinSCP()
