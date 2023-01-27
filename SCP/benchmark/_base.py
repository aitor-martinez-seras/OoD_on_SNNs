import numpy as np
import matplotlib.pyplot as plt


class _OODMethod:

    def __init__(self):
        self.precision = None
        self.tpr_values = None
        self.fpr_values = None

    def compute_metrics(self):
        assert self.precision is not None, "Precision values are not computed"
        assert self.tpr_values is not None, "FPR values are not computed"
        assert self.fpr_values is not None, "TPR values are not computed"
        # Appending that when FPR = 1 the TPR is also 1
        tpr_values_auroc = np.append(self.tpr_values, 1)
        fpr_values_auroc = np.append(self.fpr_values, 1)

        auroc = round(np.trapz(tpr_values_auroc, fpr_values_auroc) * 100, 2)
        aupr = round(np.trapz(self.precision, self.tpr_values) * 100, 2)
        fpr95 = round(fpr_values_auroc[95] * 100, 2)
        fpr80 = round(fpr_values_auroc[80] * 100, 2)
        return auroc, aupr, fpr95, fpr80

    def __call__(self, score_train, score_test, score_ood, save_histogram=False, name='', *args, **kwargs):
        # This class must be overridden by the child
        pass

    def save_histogram_fig(self, score_train, score_test, score_ood, name, *args, **kwargs):
        plt.figure(figsize=(10, 5), tight_layout=True)
        plt.hist(score_train, bins=50, color='blue', alpha=0.6, density=True, label='Train')
        plt.hist(score_test, bins=50, color='green', alpha=0.6, density=True, label='Test')
        plt.hist(score_ood, bins=50, color='darkorange', alpha=0.6, density=True, label='ood')
        # plt.ylim([0,10])
        plt.legend(fontsize=18)
        plt.savefig(f'{name}.png', dpi=200)
        plt.close()
