import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from SCP.utils.metrics import thresholds_for_each_TPR_likelihood, likelihood_method_compute_precision_tpr_fpr_for_test_and_ood, \
    thresholds_for_each_TPR_distances, distance_method_compute_precision_tpr_fpr_for_test_and_ood

from SCP.detection._base import _OODMethod

def iterate_data_gradnorm(model, data_loader, temperature, num_classes):
    confs = []
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    print('Extracting Gradnorm')
    for b, (x, y) in enumerate(data_loader):
        if b % 10 == 0:
            print(f'{b} batches processed')
        for img in x:
            img = img.unsqueeze(0)
            inputs = Variable(img.cuda(), requires_grad=True)

            model.zero_grad()
            outputs = model(inputs)
            targets = torch.ones((inputs.shape[0], num_classes)).cuda()
            outputs = outputs / temperature
            loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1))

            loss.backward()

            layer_grad = model.snn.fc_out.weight.grad.data


            layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()

            confs.append(layer_grad_norm)

    confs = np.array(confs)

    print(f'Average of the extracted gradients = {confs.mean()}')

    return confs


class GradNorm(_OODMethod):

    def __init__(self):
        super().__init__()

    def __call__(self, scores_train, scores_test, scores_ood, save_histogram=False, name='', *args, **kwargs):

        if save_histogram:
            super().save_histogram_fig(
                scores_train, scores_test, scores_ood, name=f'{name}_gradnorm'
            )

        # Creation of the array with the thresholds for each TPR (class, dist_per_TPR)
        likelihood_thresholds_train = thresholds_for_each_TPR_distances(scores_train)

        # Computing precision, tpr and fpr
        self.precision, self.tpr_values, self.fpr_values = distance_method_compute_precision_tpr_fpr_for_test_and_ood(
            scores_test, scores_ood, likelihood_thresholds_train
        )
        
        return super().compute_metrics()