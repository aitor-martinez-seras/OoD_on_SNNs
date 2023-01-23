from pathlib import Path
import argparse

from tqdm import tqdm, trange
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import numpy as np

from SCP.datasets import in_distribution_datasets_loader
from SCP.utils.common import load_paths_config, load_config
from SCP.utils.plots import plot_loss_history
from SCP.models.model import load_model
from test import validate_one_epoch
from SCP.models.model import save_checkpoint


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
    # parser.add_argument("--load-model", action="store_true", default=False,
    #                     help="Can only be set if no SNN is used, and in that case the pretrained weights for"
    #                          "RPN and Detector will be used")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--opt", default="AdamW", type=str, help="optimizer. Options: AdamW and SGD")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd", "--weight-decay", default=0.0001, type=float, metavar="W",
        help="weight decay (default: 1e-4)", dest="weight_decay",
    )
    parser.add_argument("--lr-decay-milestones", default=[], type=int, nargs='+',
                        dest="lr_decay_milestones", help="lr decay milestones")
    parser.add_argument("--lr-decay-step", default=0, type=int, dest="lr_decay_step", help="lr decay step")
    parser.add_argument("--lr-decay-rate", default=0, type=float, dest="lr_decay_rate", help="lr decay rate")

    return parser


def train_one_epoch(model, device, train_loader, optimizer, epoch):

    model.train()
    losses = []

    for data, target in tqdm(train_loader, leave=False, desc=f'Progress of epoch {epoch + 1}'):

        # Process data
        data, target = data.to(device), target.to(device)
        output = model(data)

        # Negative loglikelihoog loss, for classification problems.
        # The input must contain log probabilities of each class
        loss = torch.nn.functional.nll_loss(output, target)

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Store the losses of every minibatch
        losses.append(loss.item())

    mean_loss = np.mean(losses)
    return losses, mean_loss


def train(model, device, train_loader: DataLoader, test_loader: DataLoader,
          epochs: int, optimizer: Optimizer, lr_scheduler):
    training_losses = []
    test_losses = []
    accuracies = []
    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}:', end="\n\t")
        # Train
        _, mean_training_loss = train_one_epoch(model, device, train_loader, optimizer, epoch)

        # Val
        mean_test_loss, accuracy, _ = validate_one_epoch(model, device, test_loader)

        # Accumulate losses
        training_losses.append(mean_training_loss)
        test_losses.append(mean_test_loss)
        accuracies.append(accuracy)
        print(f"Training loss:\t{mean_training_loss}%", end="\n\t")
        print(f"Test loss:\t {mean_test_loss}%", end="\n\t")
        print(f"Accuracy test:\t{accuracies[-1]}%", end="\n\t")

        # Update the learning rate
        if lr_scheduler:
            lr_scheduler.step()

    return training_losses, test_losses


def main(args):
    print('****************** Starting training script ******************')
    print(args)
    # Device for computation
    device = args.device if torch.cuda.is_available() else torch.device('cpu')

    # Paths
    config_pth = load_paths_config()
    logs_path = Path(config_pth["paths"]["logs"])
    figures_path = Path(config_pth["paths"]["figures"])
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
    print(f'Load of {args.dataset} completed!')

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

    # Optimizer and LR scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD and AdamW are supported.")

    # Learning rate scheduler
    lr_scheduler = None
    if args.lr_decay_milestones:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=args.lr_decay_milestones,
            gamma=args.lr_decay_rate,
            last_epoch=-1,
            verbose=True
        )
    elif args.lr_decay_step:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate, verbose=True
        )
    else:
        print('No LR scheduler used')

    # Move model to device after resuming it
    model = model.to(device)

    # Train the model
    train_losses, test_losses = train(
        model,
        device,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    print('Saving model...')
    fname = f'{args.dataset}_{args.model}_{args.penultimate_layer_neurons}_{dat_conf["classes"]}_{args.n_hidden_layers}'
    save_checkpoint(
        fpath=weights_path / f'state_dict_{fname}_layers.pth',
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        args=args,
        epoch=args.epochs,
    )
    print('Model saved!')
    plot_loss_history(train_losses, test_losses, fpath=figures_path / f'history_{fname}.jpg')


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)

