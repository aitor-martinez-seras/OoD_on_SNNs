import torch
import numpy as np


def validate_one_epoch(model, device, test_loader, return_logits=False):
    # To accumulate all the spikes across different batches
    preds = []
    all_logits = []
    hidden_spikes = []
    losses = []
    correct = 0

    model.eval()

    with torch.no_grad():
        for data, target in test_loader:
            # n = n + 1
            data, target = data.to(device), target.to(device)

            if return_logits is True:
                logits, hdd_spk = model(data, flag="hidden_spikes_and_logits")
                output = torch.nn.functional.log_softmax(logits, dim=1)
                all_logits.append(logits.cpu().numpy())
                hidden_spikes.append(hdd_spk.detach().cpu().numpy())

            else:
                output = model(data)
                # Compute and sum the loss
                test_loss = torch.nn.functional.nll_loss(output, target)
                losses.append(test_loss.item())

            # Get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            # Extract the labels, predictions and the hidden layer spikes
            preds.append(pred.cpu().numpy())

    accuracy = 100.0 * correct / len(test_loader.dataset)
    if return_logits is True:
        return accuracy, np.concatenate(preds).squeeze(), np.concatenate(all_logits), np.hstack(hidden_spikes)
    else:
        # Concatenate is used to attach each batch to the previous one
        # in the same dimensions, to obtain the full test split of spikes
        # in a single array. Squeeze() is to eliminate one extra dimension.
        return np.mean(losses), accuracy, np.concatenate(preds).squeeze()
