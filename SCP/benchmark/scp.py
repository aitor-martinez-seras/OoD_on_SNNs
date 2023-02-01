import matplotlib.pyplot as plt

from SCP.benchmark._base import _OODMethod
from SCP.utils.common import find_idx_of_class
from SCP.utils.metrics import thresholds_per_class_for_each_TPR, compute_precision_tpr_fpr_for_test_and_ood


class SCP(_OODMethod):

    def __init__(self):
        super().__init__()

    def __call__(self, distances_train_per_class, distances_test_per_class, distances_ood_per_class,
                 save_histogram=False, name='', class_names=None, preds_ood=None, *args, **kwargs):

        # Creation of the array with the thresholds for each TPR (class, dist_per_TPR)
        distance_thresholds_train = thresholds_per_class_for_each_TPR(
            len(class_names), distances_train_per_class
        )

        # Computing precision, tpr and fpr
        self.precision, self.tpr_values, self.fpr_values = compute_precision_tpr_fpr_for_test_and_ood(
            distances_test_per_class, distances_ood_per_class, distance_thresholds_train
        )
        if save_histogram:
            self.plot_classwise_distances_to_cluster_averages(
                distances_train_per_class, distances_test_per_class, distances_ood_per_class,
                name=f'{name}_SCP', class_names=class_names, preds_ood=preds_ood
            )

        return super().compute_metrics()

    def plot_classwise_distances_to_cluster_averages(self, dist_train, dist_test, dist_ood,
                                                     class_names, name, preds_ood):
        n_classes = len(class_names)
        if n_classes % 4 == 0:
            rows = 4
            cols = n_classes // 4
        elif n_classes % 2 == 0:
            rows = 2
            cols = n_classes // 2
        else:
            raise ValueError('Plot not implemented for that number of classes')

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows),
                                 tight_layout=True)
        for class_index, ax in enumerate(axes.flat):
            n_samples = len(dist_ood[class_index])
            ood_bins = n_samples + 1 if n_samples < 25 else 25
            ax.hist(dist_train[class_index], bins=25, color='dodgerblue', density=True, label='Train')
            ax.hist(dist_test[class_index], bins=25, color='green', alpha=0.6, density=True, label='Test')
            ax.hist(dist_ood[class_index], bins=ood_bins, color='darkorange', alpha=0.6, density=True, label='ood')
            ax.set_title(
                f'{class_names[class_index]} // Number of samples ood = {len(preds_ood[find_idx_of_class(class_index, preds_ood)])}')
            ax.legend()
        fig.savefig(f'{name}.pdf')
        plt.close(fig)


if __name__ == "__main__":
    scp = SCP()
    scp.save_auroc_fig()
