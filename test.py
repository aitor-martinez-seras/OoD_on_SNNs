from pathlib import Path
import argparse
from typing import Dict

import torch
import numpy as np

from SCP.datasets import in_distribution_datasets_loader
from SCP.models.model import load_model
from SCP.utils.common import load_config, load_paths_config


def get_args_parser():
    parser = argparse.ArgumentParser(description="Training SNN", add_help=True)

    parser.add_argument("--dataset", default="", type=str, help="dataset to train on")
    parser.add_argument("--conf", default="config", type=str, help="name of the configuration in config folder")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", dest='batch_size', default=16, type=int, help="batch size")
    parser.add_argument("--model", default="", type=str, help="name of the model")
    parser.add_argument("--encoder", default="poisson", type=str,
                        help="encoder to use. Options 'poisson' and 'constant'")
    parser.add_argument("--n-time-steps", default=24, type=int, dest='n_time_steps',
                        help="number of timesteps for the simulation")
    parser.add_argument("--f-max", default=100, type=int, dest='f_max',
                        help="max frecuency of the input neurons per second")
    parser.add_argument("--n-hidden-layers", default=1, type=int,
                        dest="n_hidden_layers", help="number of hidden layers of the models")
    parser.add_argument("--penultimate-layer-neurons", default=200, type=int, dest="penultimate_layer_neurons",
                        help="number of neurons in the second to last layer of the model")
    parser.add_argument("--load-weights", type=str, default=False, dest='load_weights',
                        help="load weights for a model")
    parser.add_argument("--epochs", default=1, type=int)

    return parser


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


def main(args):
    print('****************** Starting testing script ******************')
    print(args)

    # Device for computation
    device = args.device if torch.cuda.is_available() else torch.device('cpu')

    # Paths
    config_pth = load_paths_config()
    logs_path = Path(config_pth["paths"]["logs"])
    weights_path = Path(config_pth["paths"]["weights"])
    datasets_path = Path(config_pth["paths"]["datasets"])

    # Load dataset and its config and create the data loaders
    dat_conf = load_config('datasets')
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context  # To enable the correct download of datasets
    if args.dataset in load_config('datasets').keys():
        dat_conf = dat_conf[args.dataset]
    else:
        raise NotImplementedError(f'Dataset with name {args.dataset} is not implemented')
    print(f'Loading {args.dataset}...')
    train_data, train_loader, test_loader = in_distribution_datasets_loader[args.dataset](
        args.batch_size, datasets_path,
    )
    print(f'Load completed!')

    # Load model
    model = load_model(
        model_arch=args.model,
        input_size=dat_conf['input_size'],
        hidden_neurons=args.penultimate_layer_neurons,
        output_neurons=dat_conf['classes'],
        n_hidden_layers=args.n_hidden_layers,
        encoder=args.encoder,
        n_time_steps=args.n_time_steps,
        f_max=args.f_max
    )

    if args.load_weights:


    model = model.to(device)

    print('Testing...')
    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}:')
        # Val
        mean_test_loss, accuracy, _ = validate_one_epoch(model, device, test_loader)

        # Accumulate losses
        print(f"\nThe mean loss of the model for epoch {epoch + 1} is {mean_test_loss}%")
        print(f"\nThe accuracy of the model for epoch {epoch + 1} is {accuracy}%")


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
