import torch
from norse.torch import PoissonEncoder
from norse.torch import ConstantCurrentLIFEncoder
from norse.torch import LIFParameters

from SCP.models.fc import FCSNN1, FCSNN2
from SCP.models.conv import ConvSNN1, ConvSNN2


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
            x, _ = torch.max(x, 0)
            return x, hdd_spks


def decode(x):
    # First take the max across all time steps, the first dimension
    # [time_step, batch_size, output_neurons ]
    x, _ = torch.max(x, 0)
    # Then compute the logsoftmax across the
    log_p_y = torch.nn.functional.log_softmax(x, dim=1)
    return log_p_y


def load_model(model_arch: str, device, input_features: int,
               hidden_neurons=None, output_neurons=10, encoder='poisson',
               n_time_steps=24, f_max=100, n_hidden_layers=1):
    if encoder == 'poisson':
        encoder = PoissonEncoder(seq_length=n_time_steps, f_max=100)
    elif encoder == 'constant current':
        encoder = ConstantCurrentLIFEncoder(seq_length=n_time_steps, p=LIFParameters(v_th=torch.tensor(0.25)))
    else:
        raise NotImplementedError(f'Encoder {encoder} not implemented')

    if model_arch == 'Fully_connected':
        if hidden_neurons is None:
            hidden_neurons = 200

        if n_hidden_layers == 1:

            model = Model(
                encoder=encoder,
                snn=FCSNN1(input_features=input_features,
                           hidden_features=hidden_neurons,
                           output_features=output_neurons),
                decoder=decode
            ).to(device)

        elif n_hidden_layers == 2:
            model = Model(
                encoder=encoder,
                snn=FCSNN2(input_features=input_features,
                           hidden_features=hidden_neurons,
                           output_features=output_neurons),
                decoder=decode
            ).to(device)

        else:
            raise NameError('Wrong number of layers')

    elif model_arch == 'ConvNet':

        if hidden_neurons is None:
            hidden_neurons = 300

        if n_hidden_layers == 1:
            model = Model(
                encoder=encoder,
                snn=ConvSNN1(hidden_neurons=hidden_neurons,
                             output_neurons=output_neurons,
                             alpha=80),
                decoder=decode
            ).to(device)

        elif n_hidden_layers == 2:
            model = Model(
                encoder=PoissonEncoder(n_time_steps),
                snn=ConvSNN2(hidden_neurons=hidden_neurons,
                             output_neurons=output_neurons,
                             alpha=80),
                decoder=decode
            ).to(device)

        else:
            raise NameError('Wrong number of layers')
    else:
        raise ValueError('Wrong model architecture introduced')

    return model
