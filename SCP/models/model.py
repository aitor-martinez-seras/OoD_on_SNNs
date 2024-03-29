from collections import OrderedDict
import math

import torch
from norse.torch import PoissonEncoder
from norse.torch import ConstantCurrentLIFEncoder
from norse.torch import LIFParameters

from SCP.models.fc import FCSNN1, FCSNN2
from SCP.models.conv import ConvSNN1, ConvSNN2, ConvSNN11


def save_checkpoint(fpath, model, optimizer, args, epoch, lr_scheduler):

    if lr_scheduler:
        if isinstance(lr_scheduler, list):
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler[0].state_dict(),  # Only the milestones
                "args": args,
                "epoch": epoch,
            }
        else:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "args": args,
                "epoch": epoch,
            }
    else:
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": None,
            "args": args,
            "epoch": epoch,
        }
    torch.save(checkpoint, fpath)


def _load(fpath):
    try:
        file = torch.load(fpath, map_location="cpu")
    except NotImplementedError:
        print('')
        print('WARNING: Loading weights saved on Linux into a Windows machine, overriding pathlib.PosixPath'
              ' with pathlib.WindowsPath to enable the load')
        print('')
        import pathlib
        pathlib.PosixPath = pathlib.WindowsPath
        file = torch.load(fpath, map_location="cpu")
    return file


def load_weights(model, weights_path):
    print('----------------------------------')
    print('       LOADING WEIGHTS')
    weights = _load(weights_path)
    if isinstance(weights, OrderedDict):
        print(model.load_state_dict(weights, strict=False))
    elif type(weights) == dict:
        print(model.load_state_dict(weights["model"], strict=False))
    else:
        raise TypeError(f"Loaded file has wrong type: {type(weights)}")
    print("Loading weights finished!")
    print('----------------------------------')


def load_checkpoint(model, weights_path, optimizer, lr_scheduler):
    print('----------------------------------')
    print('         RESUMING TRAINING        ')
    checkpoint = _load(weights_path)
    if type(checkpoint) == dict:
        print(model.load_state_dict(checkpoint["model"]))
        optimizer.load_state_dict(checkpoint["optimizer"])
        if lr_scheduler:
            if checkpoint["lr_scheduler"]:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            else:
                lr_scheduler = None
                print('WARNING: LR SCHEDULER not loaded as it was not present in the checkpoint file')
        start_epoch = checkpoint["epoch"]
    else:
        raise TypeError(f"Loaded file has wrong type: {type(checkpoint)}")
    print("Loading checkpoint finished!")
    print('----------------------------------')
    return start_epoch


# TODO: Maybe optimize the encoder for less memory utilization in the future
#   by using the internals of PoissonEncoder
class Model(torch.nn.Module):
    def __init__(self, encoder, snn, decoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.snn = snn
        self.decoder = decoder

    def forward(self, x, flag=None):
        x = self.encoder(x)
        if flag is None:
            x = self.snn(x)
            x = self.decoder(x)
            return x

        elif flag == "hidden_spikes_and_logits":
            x, hdd_spks = self.snn(x, flag)
            return x, hdd_spks


# For the case where the input is already in spiking form
def no_encoder(x):
    return x


def decode(x):
    # Then compute the logsoftmax across the
    log_p_y = torch.nn.functional.log_softmax(x, dim=1)
    return log_p_y


def load_model(model_type: str, input_size: list, hidden_neurons=None, output_neurons=10, arch_selector=1,
               encoder='poisson', n_time_steps=24, f_max=100):

    # Select the encoder
    if encoder.lower() == 'poisson':
        encoder = PoissonEncoder(seq_length=n_time_steps, f_max=f_max)
    elif encoder.lower() == 'constant':
        encoder = ConstantCurrentLIFEncoder(seq_length=n_time_steps, p=LIFParameters(v_th=torch.tensor(0.25)))
    elif encoder.lower() == 'neuromorphic':
        encoder = no_encoder
    else:
        raise NotImplementedError(f'Encoder {encoder} not implemented')

    if model_type == 'Fully_connected':

        # To obtain input size flattened from the different dimensions of the image
        input_size = math.prod(input_size)

        if hidden_neurons is None:
            hidden_neurons = 200

        if arch_selector == 1:

            model = Model(
                encoder=encoder,
                snn=FCSNN1(input_features=input_size,
                           hidden_features=hidden_neurons,
                           output_features=output_neurons),
                decoder=decode
            )

        elif arch_selector == 2:
            model = Model(
                encoder=encoder,
                snn=FCSNN2(input_features=input_size,
                           hidden_features=hidden_neurons,
                           output_features=output_neurons),
                decoder=decode
            )

        else:
            raise NameError('Wrong number of layers')

    elif model_type == 'ConvNet':

        if hidden_neurons is None:
            hidden_neurons = 300

        if arch_selector == 1:
            model = Model(
                encoder=encoder,
                snn=ConvSNN1(
                    input_size=input_size,
                    hidden_neurons=hidden_neurons,
                    output_neurons=output_neurons,
                    alpha=80
                ),
                decoder=decode
            )

        elif arch_selector == 2:
            model = Model(
                encoder=encoder,
                snn=ConvSNN2(
                    input_size=input_size,
                    hidden_neurons=hidden_neurons,
                    output_neurons=output_neurons,
                    alpha=80
                ),
                decoder=decode
            )

        elif arch_selector == 11:
            model = Model(
                encoder=encoder,
                snn=ConvSNN11(
                    input_size=input_size,
                    hidden_neurons=hidden_neurons,
                    output_neurons=output_neurons,
                    alpha=100
                ),
                decoder=decode
            )

        else:
            raise NameError('Wrong architecture selected')
    else:
        raise ValueError('Wrong model type introduced')

    return model
