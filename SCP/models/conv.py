import torch
from norse.torch import LIFParameters
from norse.torch.module.lif import LIFCell
from norse.torch import LICell


class ConvSNN1(torch.nn.Module):
    def __init__(self, hidden_neurons, output_neurons, num_channels=1,
                 feature_size=28, alpha=100, input_size=None):
        # super(ConvNet, self).__init__()
        super().__init__()

        self.features = int(((feature_size - 2) / 2) - 2)

        self.ftmaps_h = int(((input_size[1] - 2) / 2) - 2)
        self.ftmaps_v = int(((input_size[2] - 2) / 2) - 2)

        self.conv1 = torch.nn.Conv2d(input_size[0], 20, 3, 1, bias=False)
        self.conv2 = torch.nn.Conv2d(20, 50, 3, 1, bias=False)
        # self.fc1 = torch.nn.Linear(self.features * self.features * 50, hidden_neurons, bias=False)
        self.fc1 = torch.nn.Linear(self.ftmaps_h * self.ftmaps_v * 50, hidden_neurons, bias=False)
        self.fc2 = torch.nn.Linear(hidden_neurons, output_neurons, bias=False)  # Out fc
        self.lif0 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.1), alpha=alpha))
        self.lif1 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.05), alpha=alpha))
        self.lif2 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.05), alpha=alpha))
        self.out = LICell()

        # TODO: Abstraer los thrs the aqui (0.2, 0.2, 0.1)

        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons

    def forward(self, x, flag=None):
        seq_length = x.shape[0]
        batch_size = x.shape[1]

        # specify the initial states
        s0 = s1 = s2 = so = None
        voltages = torch.zeros(
            seq_length, batch_size, self.output_neurons, device=x.device, dtype=x.dtype
        )
        if flag is None:
            for ts in range(seq_length):
                # First convolution
                z = self.conv1(x[ts, :])
                print(f'Encoder: {(z.count_nonzero() / z.nelement()) * 100:.3f}%')
                z, s0 = self.lif0(z, s0)
                z = torch.nn.functional.avg_pool2d(z, 2)  # (batch_size, filters (20), (H-4)/2, (W-4)/2)
                print(f'After conv1: {(z.count_nonzero() / z.nelement()) * 100:.3f}%')

                # Second convolution
                z = self.conv2(z)
                z, s1 = self.lif1(z, s1)
                # z = torch.nn.functional.avg_pool2d(z, 2)
                print(f'After conv2: {(z.count_nonzero() / z.nelement()) * 100:.3f}%')

                # Fully connected part
                z = z.flatten(start_dim=1)
                # z = z.view(-1, self.features ** 2 * 50)  # Flatten -Z (batch_size, 800)

                # First linear connection
                z = self.fc1(z)  # (batch_size, 500)
                z, s2 = self.lif2(z, s2)  # The neuron is the activation function

                # Second linear connection
                z = self.fc2(z)  # (batch_size, 10)
                v, so = self.out(z, so)
                voltages[ts, :, :] = v

            return voltages

        elif flag == "hidden_spikes_and_logits":
            hdn_spk_last_layer = torch.zeros(
                seq_length, batch_size, self.hidden_neurons, device=x.device, dtype=torch.int8
            )
            for ts in range(seq_length):  # A la derecha pongo la salida del modelo
                # First convolution
                z = self.conv1(x[ts, :])  # (batch_size, filters (20), H-2, W-2)
                z, s0 = self.lif0(z, s0)  # (batch_size, filters (20), H-2, W-2)
                z = torch.nn.functional.avg_pool2d(z, 2)  # (batch_size, filters (20), (H-4)/2, (W-4)/2)

                # Second convolution
                z = self.conv2(z)
                z, s1 = self.lif1(z, s1)

                # Fully connected part
                z = z.view(-1, self.features ** 2 * 50)  # Flatten -Z (batch_size, 800)

                # First linear connection
                z = self.fc1(z)  # (batch_size, 500)
                z, s2 = self.lif2(z, s2)  # The neuron is the activation function
                hdn_spk_last_layer[ts, :, :] = z  # To save the spikes (ts, batch_size, 500)

                # Second linear connection
                z = self.fc2(z)  # (batch_size, 10)
                v, so = self.out(z, so)
                voltages[ts, :, :] = v

            return voltages, hdn_spk_last_layer

        elif flag == "hidden_and_conv_spikes_and_logits":
            hdn_spk_last_layer = torch.zeros(
                seq_length, batch_size, self.hidden_neurons, device=x.device, dtype=torch.int8
            )
            conv_spikes = torch.zeros(seq_length, batch_size, 50, 11, 11,
                                      device=x.device, dtype=torch.bool)
            for ts in range(seq_length):  # A la derecha pongo la salida del modelo
                # First convolution
                z = self.conv1(x[ts, :])  # (batch_size, filters (20), H-2, W-2)
                z, s0 = self.lif0(z, s0)  # (batch_size, filters (20), H-2, W-2)
                z = torch.nn.functional.avg_pool2d(z, 2)  # (batch_size, filters (20), (H-4)/2, (W-4)/2)

                # Second convolution
                z = self.conv2(z)  # Multiplica por 10 para que el valor de
                z, s1 = self.lif1(z, s1)
                conv_spikes[ts] = z

                # Fully connected part
                z = z.view(-1, self.features ** 2 * 50)  # Flatten -Z (batch_size, 800)

                # First linear connection
                z = self.fc1(z)  # (batch_size, 500)
                z, s2 = self.lif2(z, s2)  # The neuron is the activation function
                hdn_spk_last_layer[ts, :, :] = z  # To save the spikes (ts, batch_size, 500)

                # Second linear connection
                z = self.fc2(z)  # (batch_size, 10)
                v, so = self.out(z, so)
                voltages[ts, :, :] = v

            return voltages, hdn_spk_last_layer, conv_spikes


class ConvSNN2(torch.nn.Module):
    def __init__(self, hidden_neurons, output_neurons, num_channels=1,
                 feature_size=28, alpha=100):
        # super(ConvNet, self).__init__()
        super().__init__()

        self.features = int(((feature_size - 2) / 2) - 2)

        self.conv1 = torch.nn.Conv2d(num_channels, 20, 3, 1, bias=False)
        self.conv2 = torch.nn.Conv2d(20, 50, 3, 1, bias=False)
        self.fc1 = torch.nn.Linear(self.features * self.features * 50, 500, bias=False)
        self.fc2 = torch.nn.Linear(500, hidden_neurons, bias=False)
        self.fc_out = torch.nn.Linear(hidden_neurons, output_neurons, bias=False)  # Out fc
        self.lif0 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.2), alpha=alpha))
        self.lif1 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.2), alpha=alpha))
        self.lif2 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.1), alpha=alpha))
        self.lif3 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.05), alpha=alpha))
        self.out = LICell()

        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons

    def forward(self, x, flag=None):
        seq_length = x.shape[0]
        batch_size = x.shape[1]

        # specify the initial states
        s0 = s1 = s2 = s3 = so = None
        voltages = torch.zeros(
            seq_length, batch_size, self.output_neurons, device=x.device, dtype=x.dtype
        )
        if flag is None:
            for ts in range(seq_length):  # A la derecha pongo la salida del modelo
                # First convolution
                z = self.conv1(x[ts, :])
                z, s0 = self.lif0(z, s0)
                z = torch.nn.functional.avg_pool2d(z, 2)  # (batch_size, filters (20), (H-4)/2, (W-4)/2)

                # Second convolution
                z = self.conv2(z)
                z, s1 = self.lif1(z, s1)
                # z = torch.nn.functional.avg_pool2d(z, 2)

                # Fully connected part
                z = z.view(-1, self.features ** 2 * 50)  # Flatten -Z (batch_size, 800)

                # First linear connection
                z = self.fc1(z)  # (batch_size, 500)
                z, s2 = self.lif2(z, s2)  # The neuron is the activation function

                # Second linear connection
                z = self.fc2(z)
                z, s3 = self.lif3(z, s3)  # The neuron is the activation function

                # Final linear connection
                z = self.fc_out(z)  # (batch_size, 10)
                v, so = self.out(z, so)
                voltages[ts, :, :] = v

            return voltages

        elif flag == "hidden_spikes_and_logits":
            hdn_spk_last_layer = torch.zeros(
                seq_length, batch_size, self.hidden_neurons, device=x.device, dtype=torch.int8
            )
            for ts in range(seq_length):  # A la derecha pongo la salida del modelo
                # First convolution
                z = self.conv1(x[ts, :])  # (batch_size, filters (20), H-2, W-2)
                z, s0 = self.lif0(z, s0)  # (batch_size, filters (20), H-2, W-2)
                z = torch.nn.functional.avg_pool2d(z, 2)  # (batch_size, filters (20), (H-4)/2, (W-4)/2)

                # Second convolution
                z = self.conv2(z)  # Multiplica por 10 para que el valor de
                z, s1 = self.lif1(z, s1)

                # Fully connected part
                z = z.view(-1, self.features ** 2 * 50)  # Flatten -Z (batch_size, 800)

                # First linear connection
                z = self.fc1(z)  # (batch_size, 500)
                z, s2 = self.lif2(z, s2)  # The neuron is the activation function
                # hdn_spk_last_layer[ts, :, :] = z # To save the spikes (ts, batch_size, 500)

                # Second linear connection
                z = self.fc2(z)
                z, s3 = self.lif3(z, s3)  # The neuron is the activation function
                hdn_spk_last_layer[ts, :, :] = z  # To save the spikes (ts, batch_size, 500)

                # Second linear connection
                z = self.fc_out(z)  # (batch_size, 10)
                v, so = self.out(z, so)
                voltages[ts, :, :] = v

            return voltages, hdn_spk_last_layer
