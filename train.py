import time
from pathlib import Path
import argparse

from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import numpy as np

from SCP.datasets import datasets_loader
from SCP.datasets.utils import load_dataloader
from SCP.utils.common import load_config, my_custom_logger
from SCP.utils.plots import plot_loss_history
from SCP.models.model import load_model, load_weights, load_checkpoint
from test import validate_one_epoch
from SCP.models.model import save_checkpoint


def get_args_parser():
    parser = argparse.ArgumentParser(description="Training SNN", add_help=True)

    parser.add_argument("--dataset", default="", type=str, help="dataset to train on")
    parser.add_argument("--save-every", default=25, type=int, dest='save_every')
    parser.add_argument("--resume", type=str, default='', help="path to the checkpoint to resume training")
    parser.add_argument("--load-weights", type=str, default=False, dest='load_weights',
                        help="load weights for a model")
    parser.add_argument("--conf", default="config", type=str, help="name of the configuration in config folder")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", dest='batch_size', default=16, type=int, help="batch size")
    parser.add_argument("--model", default="", type=str, help="name of the model",
                        choices=['Fully_connected', 'ConvNet'])
    parser.add_argument("--encoder", default="poisson", type=str,
                        help="encoder to use. Options 'poisson' and 'constant'")
    parser.add_argument("--n-time-steps", default=24, type=int, dest='n_time_steps',
                        help="number of timesteps for the simulation")
    parser.add_argument("--f-max", default=100, type=int, dest='f_max',
                        help="max frequency of the input neurons per second")
    parser.add_argument("--n-hidden-layers", default=1, type=int,
                        dest="n_hidden_layers", help="number of hidden layers of the models")
    parser.add_argument("--penultimate-layer-neurons", default=200, type=int, dest="penultimate_layer_neurons",
                        help="number of neurons in the second to last layer of the model")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--opt", default="AdamW", type=str, help="optimizer. Options: Adam, AdamW and SGD")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd", "--weight-decay", default=0.00001, type=float, metavar="W",
        help="weight decay (default: 1e-5)", dest="weight_decay",
    )
    parser.add_argument("--lr-decay-milestones", default=[], type=int, nargs='+',
                        dest="lr_decay_milestones", help="lr decay milestones")
    parser.add_argument("--lr-decay-step", default=0, type=int, dest="lr_decay_step", help="lr decay step")
    parser.add_argument("--lr-decay-rate", default=0, type=float, dest="lr_decay_rate", help="lr decay rate")
    parser.add_argument("--constant-lr-scheduler", default=0, type=float, dest="constant_lr_scheduler",
                        help="Use ConstantLR to decrease the LR the first epoch by the factor specified")
    parser.add_argument("--train-seed", default=6, type=int, dest='train_seed', help="seed for the train set")
    parser.add_argument("--test-seed", default=7, type=int, dest='test_seed', help="seed for the test set")

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


def train(model, device, train_loader: DataLoader, test_loader: DataLoader, epochs: int, start_epoch: int,
          optimizer: Optimizer, lr_scheduler, logger, save_every_n_epochs=0, weights_pth=Path('.'), file_name='',
          args=None):

    training_losses = []
    test_losses = []
    accuracies = []
    assert save_every_n_epochs >= 0 and isinstance(save_every_n_epochs, int), f'save_every must be ' \
                                                            f'an integer greater than 0, not {save_every_n_epochs}'
    if save_every_n_epochs > 0:
        assert weights_pth != '.' and file_name != '' and args is not None, 'datasets_path, file_path ' \
                                                                             'and args must be passed to the function'

    for epoch in range(start_epoch, epochs):
        logger.info(f'Epoch {epoch + 1}:')
        t = time.perf_counter()
        # Train
        _, mean_training_loss = train_one_epoch(model, device, train_loader, optimizer, epoch)

        # Val
        mean_test_loss, accuracy, _ = validate_one_epoch(model, device, test_loader)

        # Accumulate losses
        training_losses.append(mean_training_loss)
        test_losses.append(mean_test_loss)
        accuracies.append(accuracy)
        logger.info(f"\tTraining loss:\t{mean_training_loss}")
        logger.info(f"\tTest loss:\t {mean_test_loss}")
        logger.info(f"\tAccuracy test:\t{accuracies[-1]}%")
        logger.info(f"\tComputation time:\t{(time.perf_counter() - t)/60:.2f} minutes")

        # Update the learning rate
        if lr_scheduler:
            if isinstance(lr_scheduler, list):
                for sched in lr_scheduler:
                    sched.step()
            else:
                lr_scheduler.step()

        if save_every_n_epochs:
            if (epoch + 1) % save_every_n_epochs == 0:
                file_path = weights_pth / f'{file_name}_checkpoint{epoch+1}.pth'
                save_checkpoint(file_path, model, optimizer, args, epoch, lr_scheduler)
                logger.info(' ---------------------------------')
                logger.info(f'  - Checkpoint saved for epoch {epoch+1} -')
                logger.info(' ---------------------------------')

    return training_losses, test_losses


def main(args):
    print('****************** Starting training script ******************')
    # Device for computation
    device = args.device if torch.cuda.is_available() else torch.device('cpu')

    # Paths
    config_pth = load_config('paths')
    logs_path = Path(config_pth["paths"]["logs"])
    figures_path = Path(config_pth["paths"]["figures"])
    weights_path = Path(config_pth["paths"]["weights"])
    datasets_path = Path(config_pth["paths"]["datasets"])

    # Load dataset and its config and create the data loaders
    all_datasets_conf = load_config('datasets')
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context  # To enable the correct download of datasets
    if args.dataset in all_datasets_conf.keys():
        dataset_conf = all_datasets_conf[args.dataset]
    else:
        raise NotImplementedError(f'Dataset with name {args.dataset} is not implemented')

    print(f'Loading {args.dataset}...')
    in_dataset_data_loader = datasets_loader[args.dataset](datasets_path)
    train_data = in_dataset_data_loader.load_data(
        split='train', transformation_option='train', output_shape=dataset_conf['input_size'][1:]
    )
    test_data = in_dataset_data_loader.load_data(
        split='test', transformation_option='test', output_shape=dataset_conf['input_size'][1:]
    )
    # Define loaders. Use a seeds
    g_train = torch.Generator()
    g_train.manual_seed(args.train_seed)
    g_test = torch.Generator()
    g_test.manual_seed(args.test_seed)
    train_loader = load_dataloader(train_data, args.batch_size, shuffle=True, generator=g_train)
    test_loader = load_dataloader(test_data, args.batch_size, shuffle=True, generator=g_test)
    print(f'Load of {args.dataset} completed!')

    # Set logger
    fname = f'{args.dataset}_{args.model}_{args.penultimate_layer_neurons}' \
            f'_{dataset_conf["classes"]}_{args.n_hidden_layers}_layers'
    logger = my_custom_logger(logger_name=f'train_{fname}.txt', logs_pth=logs_path)
    logger.info(args)

    # Load model
    model = load_model(
        model_arch=args.model,
        input_size=dataset_conf['input_size'],
        hidden_neurons=args.penultimate_layer_neurons,
        output_neurons=dataset_conf['classes'],
        n_hidden_layers=args.n_hidden_layers,
        encoder=args.encoder,
        n_time_steps=args.n_time_steps,
        f_max=args.f_max
    )
    model = model.to(device)

    if args.load_weights:
        load_weights(model, args.load_weights)

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
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif opt_name == "adagrad":
        optimizer = torch.optim.Adagrad(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD and AdamW are supported.")

    # Learning rate scheduler
    lr_scheduler = None
    if args.lr_decay_milestones:
        logger.info('Using MultiStepLR')
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
        logger.info('No LR scheduler used')

    if args.constant_lr_scheduler:
        logger.info('Using the ConstantLR to adjust by a factor the first epoch of the training')
        lr_scheduler = [
            lr_scheduler, torch.optim.lr_scheduler.ConstantLR(
                optimizer=optimizer,
                factor=args.constant_lr_scheduler,
                total_iters=1,
            )
        ]

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, weights_path=args.resume, optimizer=optimizer, lr_scheduler=lr_scheduler)

    logger.info('* - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
    logger.info(model)
    logger.info('* - - - - - - - - - - - - - - - - - - - - - - - - - - - -')

    # Train the model
    train_losses, test_losses = train(
        model,
        device,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        start_epoch=start_epoch,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        logger=logger,
        save_every_n_epochs=args.save_every,
        weights_pth=weights_path,
        file_name=fname,
        args=args
    )

    logger.info('Saving model...')
    save_checkpoint(
        fpath=weights_path / f'state_dict_{fname}.pth',
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        args=args,
        epoch=args.epochs,
    )
    logger.info('Model saved!')
    plot_loss_history(train_losses, test_losses, fpath=figures_path / f'history_{fname}.jpg')


if __name__ == "__main__":
    main(get_args_parser().parse_args())

