import matplotlib.pyplot as plt
import scipy.special as scysp
import numpy as np

from SCP.detection._base import _OODMethod
from SCP.utils.common import find_idx_of_class
from SCP.utils.metrics import thresholds_per_class_for_each_TPR, compute_precision_tpr_fpr_for_test_and_ood, \
    computation_likelihood_in_or_out_distribution_per_tpr, \
    computation_distances_per_class_in_or_out_distribution_per_tpr, \
    thresholds_for_each_TPR_likelihood, likelihood_method_compute_precision_tpr_fpr_for_test_and_ood, \
    tp_fn_fp_tn_computation


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

            if save_histogram:
                super().save_histogram_fig(
                    temp_softmax_train_winners, temp_softmax_test_winners, temp_softmax_ood_winners,
                    name=f'{name}_ODIN_temp{temp}'
                )

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

        return super().compute_metrics()

    def plot_in_or_ood(self, odin_in_or_ood_per_tpr_test, odin_in_or_ood_per_tpr_ood,
                       scp_in_or_ood_per_tpr_test, scp_in_or_ood_per_tpr_ood, name):
        fig, axes = plt.subplots(2, 2, figsize=(12, 12), tight_layout=True)

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


if __name__ == "__main__":
    ensemble_odin_scp = EnsembleOdinSCP()
