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

    def save_auroc_fig(self, save=''):
        # Appending that when FPR = 1 the TPR is also 1
        tpr_values_auroc = np.append(self.tpr_values, 1)
        fpr_values_auroc = np.append(self.fpr_values, 1)
        # AUC
        auc = np.trapz(tpr_values_auroc, fpr_values_auroc)
        # Plot
        plt.figure(figsize=(15, 12))
        # Plot lines
        plt.plot(fpr_values_auroc, tpr_values_auroc, label='ROC curve', lw=3)
        plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), 'k--', label='Random ROC curve')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel('FPR', fontsize=20)
        plt.ylabel('TPR', fontsize=20)
        plt.title('ROC curve, AUC = %.3f' % auc, fontsize=25, pad=10)
        plt.fill_between(fpr_values_auroc, tpr_values_auroc, alpha=0.3)
        # Create empty plot with blank marker containing the extra label
        plt.plot([], [], ' ', label=f'FPR at 95% TPR = {round(fpr_values_auroc[95] * 100, 2)}%')
        plt.plot([], [], ' ', label=f'FPR at 80% TPR = {round(fpr_values_auroc[80] * 100, 2)}%')
        plt.legend(fontsize=20, loc='upper left')
        plt.savefig(f'{save}_AUPR.png', dpi=200)
        plt.close()

    def save_aupr_fig(self, save=''):
        # AUPR
        auc = np.trapz(self.precision, self.tpr_values)
        # Plot
        plt.figure(figsize=(15, 12))
        plt.plot(self.tpr_values, self.precision, label='ROC curve', lw=3)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel('FPR', fontsize=20)
        plt.ylabel('Precision', fontsize=20)
        plt.title('PR curve, AUC = %.3f' % auc, fontsize=25, pad=10)
        plt.fill_between(self.tpr_values, self.precision, alpha=0.3)
        plt.legend(fontsize=20, loc='upper left')
        plt.savefig(f'{save}_AUPR.png', dpi=200)
        plt.close()

