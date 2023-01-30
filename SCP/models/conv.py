import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from norse.torch import LIFParameters
from norse.torch.module.lif import LIFCell
from norse.torch import LICell


class ConvSNN1(nn.Module):
    def __init__(self, hidden_neurons, output_neurons, num_channels=1,
                 feature_size=28, alpha=100, input_size=None):
        # super(ConvNet, self).__init__()
        super().__init__()

        self.features = int(((feature_size - 2) / 2) - 2)

        self.ftmaps_h = int(((input_size[1] - 2) / 2) - 2)
        self.ftmaps_v = int(((input_size[2] - 2) / 2) - 2)

        self.conv1 = nn.Conv2d(input_size[0], 20, 3, 1, bias=False)
        self.conv2 = nn.Conv2d(20, 50, 3, 1, bias=False)
        # self.fc1 = nn.Linear(self.features * self.features * 50, hidden_neurons, bias=False)
        self.fc1 = nn.Linear(self.ftmaps_h * self.ftmaps_v * 50, hidden_neurons, bias=False)
        self.fc2 = nn.Linear(hidden_neurons, output_neurons, bias=False)  # Out fc
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
                # print(f'Encoder: {(x[ts, :].count_nonzero() / x[ts, :].nelement()) * 100:.3f}%')
                z = self.conv1(x[ts, :])
                z, s0 = self.lif0(z, s0)
                z = nn.functional.avg_pool2d(z, 2)  # (batch_size, filters (20), (H-4)/2, (W-4)/2)
                # print(f'After conv1: {(z.count_nonzero() / z.nelement()) * 100:.3f}%')

                # Second convolution
                z = self.conv2(z)
                z, s1 = self.lif1(z, s1)
                # z = nn.functional.avg_pool2d(z, 2)
                # print(f'After conv2: {(z.count_nonzero() / z.nelement()) * 100:.3f}%')

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
            # The max across all time steps is the logit, the first dimension
            # [time_step, batch_size, output_neurons]
            voltages, _ = torch.max(voltages, 0)
            return voltages

        elif flag == "hidden_spikes_and_logits":
            hdn_spk_last_layer = torch.zeros(
                seq_length, batch_size, self.hidden_neurons, device=x.device, dtype=torch.int8
            )
            for ts in range(seq_length):  # A la derecha pongo la salida del modelo
                # First convolution
                z = self.conv1(x[ts, :])  # (batch_size, filters (20), H-2, W-2)
                z, s0 = self.lif0(z, s0)  # (batch_size, filters (20), H-2, W-2)
                z = nn.functional.avg_pool2d(z, 2)  # (batch_size, filters (20), (H-4)/2, (W-4)/2)

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
            # The max across all time steps is the logit, the first dimension
            # [time_step, batch_size, output_neurons]
            voltages, _ = torch.max(voltages, 0)

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
                z = nn.functional.avg_pool2d(z, 2)  # (batch_size, filters (20), (H-4)/2, (W-4)/2)

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


class ConvSNN2(nn.Module):
    def __init__(self, hidden_neurons, output_neurons, alpha=100, input_size=None):
        super().__init__()

        self.ftmaps_h = int(((input_size[1] - 2) / 2) - 2)
        self.ftmaps_v = int(((input_size[2] - 2) / 2) - 2)

        self.conv1 = nn.Conv2d(input_size[0], 20, 3, 1, bias=False)
        self.conv2 = nn.Conv2d(20, 50, 3, 1, bias=False)
        self.fc1 = nn.Linear(self.ftmaps_h * self.ftmaps_v * 50, 500, bias=False)
        self.fc2 = nn.Linear(500, hidden_neurons, bias=False)
        self.fc_out = nn.Linear(hidden_neurons, output_neurons, bias=False)  # Out fc
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
            for ts in range(seq_length):
                # First convolution
                z = self.conv1(x[ts, :])
                z, s0 = self.lif0(z, s0)
                z = nn.functional.avg_pool2d(z, 2)  # (batch_size, filters (20), (H-4)/2, (W-4)/2)

                # Second convolution
                z = self.conv2(z)
                z, s1 = self.lif1(z, s1)
                # z = nn.functional.avg_pool2d(z, 2)

                # Fully connected part
                z = z.flatten(start_dim=1)
                # z = z.view(-1, self.features ** 2 * 50)  # Flatten -Z (batch_size, 800)

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
            # The max across all time steps is the logit, the first dimension
            # [time_step, batch_size, output_neurons]
            voltages, _ = torch.max(voltages, 0)

            return voltages

        elif flag == "hidden_spikes_and_logits":
            hdn_spk_last_layer = torch.zeros(
                seq_length, batch_size, self.hidden_neurons, device=x.device, dtype=torch.int8
            )
            for ts in range(seq_length):  # A la derecha pongo la salida del modelo
                # First convolution
                z = self.conv1(x[ts, :])  # (batch_size, filters (20), H-2, W-2)
                z, s0 = self.lif0(z, s0)  # (batch_size, filters (20), H-2, W-2)
                z = nn.functional.avg_pool2d(z, 2)  # (batch_size, filters (20), (H-4)/2, (W-4)/2)

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
            # The max across all time steps is the logit, the first dimension
            # [time_step, batch_size, output_neurons]
            voltages, _ = torch.max(voltages, 0)

            return voltages, hdn_spk_last_layer


class ConvSNN3(nn.Module):
    """
    3 Convolutions with only one avg pool
    """
    def __init__(self, input_size, hidden_neurons, output_neurons, alpha=100):
        # super(ConvNet, self).__init__()
        super().__init__()

        self.ftmaps_h = int(((input_size[1] - 2) / 2) - 2 - 2)
        self.ftmaps_v = int(((input_size[2] - 2) / 2) - 2 - 2)

        # Convolutions
        self.conv1 = nn.Conv2d(input_size[0], 32, 3, 1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, bias=False)

        # Linear part
        self.fc1 = nn.Linear(self.ftmaps_h * self.ftmaps_v * 128, hidden_neurons, bias=False)
        self.fc_out = nn.Linear(hidden_neurons, output_neurons, bias=False)  # Out fc

        # LIF cells
        self.lif_conv1 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.1), alpha=alpha))
        self.lif_conv2 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.05), alpha=alpha))
        self.lif_conv3 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.05), alpha=alpha))
        self.lif_fc1 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.05), alpha=alpha))
        self.out = LICell()

        # TODO: Abstraer los thrs the aqui (0.2, 0.2, 0.1)

        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons

    def forward(self, x, flag=None):
        seq_length = x.shape[0]
        batch_size = x.shape[1]

        # specify the initial states
        sconv1 = sconv2 = sconv3 = sfc1 = so = None
        voltages = torch.zeros(
            seq_length, batch_size, self.output_neurons, device=x.device, dtype=x.dtype
        )
        if flag is None:
            for ts in range(seq_length):
                # First convolution
                # print(f'Encoder: {(x[ts, :].count_nonzero() / x[ts, :].nelement()) * 100:.3f}%')
                z = self.conv1(x[ts, :])
                z, sconv1 = self.lif_conv1(z, sconv1)
                z = nn.functional.avg_pool2d(z, 2)  # (batch_size, filters (20), (H-4)/2, (W-4)/2)
                # print(f'After conv1: {(z.count_nonzero() / z.nelement()) * 100:.3f}%')

                # Second convolution
                z = self.conv2(z)
                z, sconv2 = self.lif_conv2(z, sconv2)
                # z = nn.functional.avg_pool2d(z, 2)
                # print(f'After conv2: {(z.count_nonzero() / z.nelement()) * 100:.3f}%')

                # Second convolution
                z = self.conv3(z)
                z, sconv3 = self.lif_conv2(z, sconv3)
                # print(f'After conv3: {(z.count_nonzero() / z.nelement()) * 100:.3f}%')

                # Fully connected part
                z = z.flatten(start_dim=1)

                # First FC
                z = self.fc1(z)
                z, sfc1 = self.lif_fc1(z, sfc1)

                # Fc out
                z = self.fc_out(z)
                v, so = self.out(z, so)
                voltages[ts, :, :] = v

            # The max across all time steps is the logit, the first dimension
            # [time_step, batch_size, output_neurons]
            voltages, _ = torch.max(voltages, 0)

            return voltages

        elif flag == "hidden_spikes_and_logits":
            hdn_spk_last_layer = torch.zeros(
                seq_length, batch_size, self.hidden_neurons, device=x.device, dtype=torch.int8
            )
            for ts in range(seq_length):  # A la derecha pongo la salida del modelo
                # First convolution
                z = self.conv1(x[ts, :])  # (batch_size, filters (20), H-2, W-2)
                z, s0 = self.lif0(z, s0)  # (batch_size, filters (20), H-2, W-2)
                z = nn.functional.avg_pool2d(z, 2)  # (batch_size, filters (20), (H-4)/2, (W-4)/2)

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

            # The max across all time steps is the logit, the first dimension
            # [time_step, batch_size, output_neurons]
            voltages, _ = torch.max(voltages, 0)

            return voltages, hdn_spk_last_layer


class SpikingNN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).type(torch.cuda.FloatTensor)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= 0.0] = 0
        return grad_input


def Pooling_sNeuron(membrane_potential, threshold, i):
    # check exceed membrane potential and reset
    ex_membrane = nn.functional.threshold(membrane_potential, threshold, 0)
    membrane_potential = membrane_potential - ex_membrane
    # generate spike
    out = SpikingNN.apply(ex_membrane)
    return membrane_potential, out


class ConvSNN4(nn.Module):
    def __init__(self, input_size, hidden_neurons, output_neurons, alpha=100):
        # super(ConvNet, self).__init__()
        super().__init__()

        self.input_size = input_size

        self.ftmaps_h = int(((input_size[1] - 2) / 2) - 2 - 2)
        self.ftmaps_v = int(((input_size[2] - 2) / 2) - 2 - 2)

        self.conv11 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)

        self.conv21 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv22 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)

        self.conv31 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv32 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv33 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.avgpool3 = nn.AvgPool2d(kernel_size=2)

        self.fc0 = nn.Linear(4*4*256, 1024, bias=False)
        self.fc1 = nn.Linear(1024, 100, bias=False)
        self.fc_out = nn.Linear(100, 10, bias=False)

        # LIF cells
        self.lif_conv11 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.1), alpha=alpha))
        self.lif_conv12 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.1), alpha=alpha))

        self.lif_conv21 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.05), alpha=alpha))
        self.lif_conv22 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.05), alpha=alpha))

        self.lif_conv31 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.05), alpha=alpha))
        self.lif_conv32 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.05), alpha=alpha))
        self.lif_conv33 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.05), alpha=alpha))

        self.lif_fc1 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.05), alpha=alpha))
        self.lif_fc2 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.05), alpha=alpha))
        self.out = LICell()

        # TODO: Abstraer los thrs the aqui (0.2, 0.2, 0.1)

        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons

    def forward(self, x, flag=None):
        seq_length = x.shape[0]
        batch_size = x.shape[1]

        # Dropout
        drop = nn.Dropout(p=0.2, inplace=True)

        mask_11 = Variable(torch.ones(batch_size, 64, 32, 32).cuda(), requires_grad=False)
        mask_11 = drop(mask_11)
        mask_12 = Variable(torch.ones(batch_size, 64, 32, 32).cuda(), requires_grad=False)
        mask_12 = drop(mask_12)
        mask_21 = Variable(torch.ones(batch_size, 128, 16, 16).cuda(), requires_grad=False)
        mask_21 = drop(mask_21)
        mask_22 = Variable(torch.ones(batch_size, 128, 16, 16).cuda(), requires_grad=False)
        mask_22 = drop(mask_22)
        mask_31 = Variable(torch.ones(batch_size, 256, 8, 8).cuda(), requires_grad=False)
        mask_31 = drop(mask_31)
        mask_32 = Variable(torch.ones(batch_size, 256, 8, 8).cuda(), requires_grad=False)
        mask_32 = drop(mask_32)
        mask_33 = Variable(torch.ones(batch_size, 256, 8, 8).cuda(), requires_grad=False)
        mask_33 = drop(mask_33)

        mask_f0 = Variable(torch.ones(batch_size, 1024).cuda(), requires_grad=False)
        mask_f0 = drop(mask_f0)

        mem_1s = Variable(torch.zeros(batch_size, 64, 16, 16).cuda(), requires_grad=False)
        mem_2s = Variable(torch.zeros(batch_size, 128, 8, 8).cuda(), requires_grad=False)
        mem_3s = Variable(torch.zeros(batch_size, 256, 4, 4).cuda(), requires_grad=False)

        # specify the initial states
        sconv11 = sconv12 = sconv21 = sconv22 = sconv31 = sconv32 = sconv33 = sfc0 = sfc1 = so = None
        voltages = torch.zeros(
            seq_length, batch_size, self.output_neurons, device=x.device, dtype=x.dtype
        )
        if flag is None:
            for ts in range(seq_length):
                # First convolution
                # print(f'Encoder: {(x[ts, :].count_nonzero() / x[ts, :].nelement()) * 100:.3f}%')
                z = self.conv11(x[ts, :])
                z, sconv11 = self.lif_conv11(z, sconv11)
                z = torch.mul(z, mask_11)
                z = self.conv12(z)
                z, sconv12 = self.lif_conv12(z, sconv12)
                z = torch.mul(z, mask_12)
                # pooling Layer
                mem_1s = mem_1s + self.avgpool1(z)
                mem_1s, z = Pooling_sNeuron(mem_1s, 0.75, ts)
                # z = nn.functional.avg_pool2d(z, 2)
                # print(f'After conv1: {(z.count_nonzero() / z.nelement()) * 100:.3f}%')

                # Second convolution
                z = self.conv21(z)
                z, sconv21 = self.lif_conv21(z, sconv21)
                z = torch.mul(z, mask_21)
                z = self.conv22(z)
                z, sconv22 = self.lif_conv22(z, sconv22)
                z = torch.mul(z, mask_22)
                # z = nn.functional.avg_pool2d(z, 2)
                # print(f'After conv2: {(z.count_nonzero() / z.nelement()) * 100:.3f}%')
                # pooling Layer
                mem_2s = mem_2s + self.avgpool2(z)
                mem_2s, z = Pooling_sNeuron(mem_2s, 0.75, ts)

                # Third convolution
                z = self.conv31(z)
                z, sconv31 = self.lif_conv31(z, sconv31)
                z = torch.mul(z, mask_31)
                z = self.conv32(z)
                z, sconv32 = self.lif_conv32(z, sconv32)
                z = torch.mul(z, mask_32)
                z = self.conv33(z)
                z, sconv33 = self.lif_conv32(z, sconv33)
                z = torch.mul(z, mask_33)
                # print(f'After conv3: {(z.count_nonzero() / z.nelement()) * 100:.3f}%')
                # pooling Layer
                mem_3s = mem_3s + self.avgpool3(z)
                mem_3s, z = Pooling_sNeuron(mem_3s, 0.75, ts)

                # Fully connected part
                z = z.flatten(start_dim=1)

                # First FC
                z = self.fc0(z)
                z, sfc0 = self.lif_fc1(z, sfc0)

                # Second FC
                z = self.fc1(z)
                z, sfc1 = self.lif_fc1(z, sfc1)

                # Fc out
                z = self.fc_out(z)
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
                z = nn.functional.avg_pool2d(z, 2)  # (batch_size, filters (20), (H-4)/2, (W-4)/2)

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


class ConvSNN5(nn.Module):
    """
    As CONVSNN11 but with decode method of MAX VOLTAGE
    """
    def __init__(self, input_size, hidden_neurons, output_neurons, alpha=100):
        # super(ConvNet, self).__init__()
        super().__init__()

        self.ftmaps_h = int(((input_size[1] / 2) / 2) - 2 - 2)
        self.ftmaps_v = int(((input_size[1] / 2) / 2) - 2 - 2)
        # self.ftmaps_h = int(((input_size[1] - 2 - 2) - 2 - 2) - 2 - 2)
        # self.ftmaps_v = int(((input_size[2] - 2 - 2) - 2 - 2) - 2 - 2)

        # Convolutions
        self.conv1 = nn.Conv2d(input_size[0], 32, 3, 1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1, bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=2)

        # Linear part
        self.fc1 = nn.Linear(4 * 4 * 128, hidden_neurons, bias=False)
        self.fc_out = nn.Linear(hidden_neurons, output_neurons, bias=False)  # Out fc

        # LIF cells
        self.lif_conv1 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.25), alpha=alpha))
        self.lif_conv2 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.2), alpha=alpha))
        self.lif_conv3 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.1), alpha=alpha))

        self.lif_fc1 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.1), alpha=alpha))
        # self.lif_fc2 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.1), alpha=alpha))
        self.out = LICell()

        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons

        # for m in self.modules():
        #     import math
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        #         variance1 = math.sqrt(2.0 / n)
        #         m.weight.data.normal_(0, variance1)
        #         # define threshold
        #         # m.threshold = 1
        #
        #     elif isinstance(m, nn.Linear):
        #         size = m.weight.size()
        #         fan_in = size[1]  # number of columns
        #         variance2 = math.sqrt(2.0 / fan_in)
        #         m.weight.data.normal_(0.0, variance2)
        #         # define threshold
        #         # m.threshold = 1

    def forward(self, x, flag=None):
        seq_length = x.shape[0]
        batch_size = x.shape[1]

        # # Dropout
        # drop = nn.Dropout(p=0.5, inplace=True)
        # mask_f1 = Variable(torch.ones(batch_size, self.hidden_neurons).cuda(), requires_grad=False)
        # mask_f1 = drop(mask_f1)

        # specify the initial states
        sconv1 = sconv2 = sconv3 = sfc1 = so = None
        voltages = torch.zeros(
            seq_length, batch_size, self.output_neurons, device=x.device, dtype=x.dtype
        )
        if flag is None:
            for ts in range(seq_length):
                # print(f'Encoder: {(x[ts, :].count_nonzero() / x[ts, :].nelement()) * 100:.3f}%')

                # First convolution
                z = self.conv1(x[ts, :])
                z, sconv1 = self.lif_conv1(z, sconv1)
                z = self.avgpool(z)

                # Second convolution
                z = self.conv2(z)
                z, sconv2 = self.lif_conv2(z, sconv2)
                z = self.avgpool(z)
                # print(f'After conv1: {(z.count_nonzero() / z.nelement()) * 100:.3f}%')

                # Third convolution
                z = self.conv3(z)
                z, sconv3 = self.lif_conv3(z, sconv3)
                z = self.avgpool(z)
                # print(f'After conv2: {(z.count_nonzero() / z.nelement()) * 100:.3f}%')

                # Fully connected part
                z = z.flatten(start_dim=1)

                # First FC
                z = self.fc1(z)
                z, sfc1 = self.lif_fc1(z, sfc1)
                # z = torch.mul(z, mask_f1)

                # Fc out
                z = self.fc_out(z)
                v, so = self.out(z, so)
                voltages[ts] = v
            # The max across all time steps is the logit, the first dimension
            # [time_step, batch_size, output_neurons]
            voltages, _ = torch.max(voltages, 0)

            return voltages

        elif flag == "hidden_spikes_and_logits":
            hidden_spks = torch.zeros(
                seq_length, batch_size, self.hidden_neurons, device=x.device, dtype=torch.int8
            )
            for ts in range(seq_length):

                # z = self.extract_fatures(x[ts], sconv1, sconv2, sconv3)

                # First convolution
                z = self.conv1(x[ts])
                z, sconv1 = self.lif_conv1(z, sconv1)
                z = self.avgpool(z)

                # Second convolution
                z = self.conv2(z)
                z, sconv2 = self.lif_conv2(z, sconv2)
                z = self.avgpool(z)
                # print(f'After conv1: {(z.count_nonzero() / z.nelement()) * 100:.3f}%')

                # Third convolution
                z = self.conv3(z)
                z, sconv3 = self.lif_conv3(z, sconv3)
                z = self.avgpool(z)
                # print(f'After conv2: {(z.count_nonzero() / z.nelement()) * 100:.3f}%')

                # Fully connected part
                z = z.flatten(start_dim=1)

                # First FC
                z = self.fc1(z)
                z, sfc1 = self.lif_fc1(z, sfc1)
                # z = torch.mul(z, mask_f1)
                hidden_spks[ts, :, :] = z

                # Fc out
                z = self.fc_out(z)
                v, so = self.out(z, so)
                voltages[ts] = v
            # The max across all time steps is the logit, the first dimension
            # [time_step, batch_size, output_neurons]
            voltages, _ = torch.max(voltages, 0)

            return voltages, hidden_spks


class ConvSNN6(nn.Module):
    """
    Convolutional as in the task of cifar10 in Norse
    """
    def __init__(self, input_size, hidden_neurons, output_neurons, alpha=100):
        # super(ConvNet, self).__init__()
        super().__init__()

        self.ftmaps_h = int(((input_size[1] / 2) / 2) - 2 - 2)
        self.ftmaps_v = int(((input_size[1] / 2) / 2) - 2 - 2)
        # self.ftmaps_h = int(((input_size[1] - 2 - 2) - 2 - 2) - 2 - 2)
        # self.ftmaps_v = int(((input_size[2] - 2 - 2) - 2 - 2) - 2 - 2)

        c = 64
        c = [c, 2 * c, 4 * c, 4 * c]

        self.v_th = 0.2

        # Convolutions
        self.conv1 = nn.Conv2d(input_size[0],  c[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(c[0], c[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(c[1], c[2], kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(c[2], c[3], kernel_size=3, stride=1, padding=1, bias=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        # Linear part
        # self.fc1 = nn.Linear(4096, hidden_neurons, bias=False)
        self.fc_out = nn.Linear(4096, output_neurons, bias=False)  # Out fc

        # LIF cells
        self.lif_conv1 = LIFCell(p=LIFParameters(v_th=torch.tensor(self.v_th), alpha=alpha))
        self.lif_conv2 = LIFCell(p=LIFParameters(v_th=torch.tensor(self.v_th), alpha=alpha))
        self.lif_conv3 = LIFCell(p=LIFParameters(v_th=torch.tensor(self.v_th), alpha=alpha))
        self.lif_conv4 = LIFCell(p=LIFParameters(v_th=torch.tensor(self.v_th), alpha=alpha))

        # self.lif_fc1 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.1), alpha=alpha))
        self.out = LICell()

        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons

        # for m in self.modules():
        #     import math
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        #         variance1 = math.sqrt(2.0 / n)
        #         m.weight.data.normal_(0, variance1)
        #         # define threshold
        #         # m.threshold = 1
        #
        #     elif isinstance(m, nn.Linear):
        #         size = m.weight.size()
        #         fan_in = size[1]  # number of columns
        #         variance2 = math.sqrt(2.0 / fan_in)
        #         m.weight.data.normal_(0.0, variance2)
        #         # define threshold
        #         # m.threshold = 1

    def forward(self, x, flag=None):
        seq_length = x.shape[0]
        batch_size = x.shape[1]

        # specify the initial states
        sconv1 = sconv2 = sconv3 = sconv4 = so = None
        voltages = torch.zeros(
            seq_length, batch_size, self.output_neurons, device=x.device, dtype=x.dtype
        )
        if flag is None:
            for ts in range(seq_length):
                # First convolution
                # print(f'Encoder: {(x[ts, :].count_nonzero() / x[ts, :].nelement()) * 100:.3f}%')
                z = self.conv1(x[ts, :])
                z, sconv1 = self.lif_conv1(z, sconv1)

                # Second conv
                z = self.conv2(z)
                z, sconv2 = self.lif_conv2(z, sconv2)
                z = self.pool2(z)

                # Third convolution
                z = self.conv3(z)
                z, sconv3 = self.lif_conv3(z, sconv3)
                z = self.pool3(z)

                # Third convolution
                z = self.conv4(z)
                z, sconv4 = self.lif_conv4(z, sconv4)
                z = self.pool4(z)

                # Fully connected part
                z = z.flatten(start_dim=1)

                # Fc out
                z = self.fc_out(z)
                v, so = self.out(z, so)
                voltages[ts] = v
            # The max across all time steps is the logit, the first dimension
            # [time_step, batch_size, output_neurons]
            voltages, _ = torch.max(voltages, 0)
            return voltages

        elif flag == "hidden_spikes_and_logits":
            hdn_spk_last_layer = torch.zeros(
                seq_length, batch_size, self.hidden_neurons, device=x.device, dtype=torch.int8
            )
            for ts in range(seq_length):  # A la derecha pongo la salida del modelo
                # First convolution
                z = self.conv1(x[ts, :])  # (batch_size, filters (20), H-2, W-2)
                z, s0 = self.lif0(z, s0)  # (batch_size, filters (20), H-2, W-2)
                z = nn.functional.avg_pool2d(z, 2)  # (batch_size, filters (20), (H-4)/2, (W-4)/2)

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


class LIFConvNet(nn.Module):
    def __init__(
        self, seq_length, input_size, alpha=100
    ):
        from norse.torch.module import SequentialState
        super().__init__()
        self.seq_length = seq_length
        self.p = LIFParameters(v_th=torch.tensor(0.2), alpha=alpha)

        c = 64
        c = [c, 2 * c, 4 * c, 4 * c]

        self.features = SequentialState(
            # preparation
            nn.Conv2d(
                input_size[0], c[0], kernel_size=3, stride=1, padding=1, bias=False
            ),
            LIFCell(self.p),
            # block 1
            nn.Conv2d(c[0], c[1], kernel_size=3, stride=1, padding=1, bias=False),
            LIFCell(self.p),
            nn.MaxPool2d(2),
            # block 2
            nn.Conv2d(c[1], c[2], kernel_size=3, stride=1, padding=1, bias=False),
            LIFCell(self.p),
            nn.MaxPool2d(2),
            # block 3
            nn.Conv2d(c[2], c[3], kernel_size=3, stride=1, padding=1, bias=False),
            LIFCell(self.p),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )

        self.classification = SequentialState(
            # Classification
            nn.Linear(4096, 10, bias=False),
            LICell(),
        )

    def forward(self, x):
        voltages = torch.empty(
            self.seq_length, x.shape[1], 10, device=x.device, dtype=x.dtype
        )
        sf = None
        sc = None
        for ts in range(self.seq_length):
            out_f, sf = self.features(x[ts], sf)
            # print(out_f.shape)
            out_c, sc = self.classification(out_f, sc)
            print(f'Features: {(out_c.count_nonzero() / out_c.nelement()) * 100:.3f}%')
            voltages[ts, :, :] = out_c + 0.001 * torch.randn(
                x.shape[1], 10, device=x.device
            )

        y_hat, _ = torch.max(voltages, 0)
        return y_hat


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 10)
        # dropout
        self.dropout = nn.Dropout(p=.5)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flattening
        x = x.view(-1, 64 * 4 * 4)
        # fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = torch.nn.functional.log_softmax(self.fc3(x), dim=1)
        return x


class ConvSNN9(nn.Module):
    def __init__(self, input_size, hidden_neurons, output_neurons, alpha=100):
        # super(ConvNet, self).__init__()
        super().__init__()

        self.ftmaps_h = int(((input_size[1] / 2) / 2) - 2 - 2)
        self.ftmaps_v = int(((input_size[1] / 2) / 2) - 2 - 2)
        # self.ftmaps_h = int(((input_size[1] - 2 - 2) - 2 - 2) - 2 - 2)
        # self.ftmaps_v = int(((input_size[2] - 2 - 2) - 2 - 2) - 2 - 2)

        # Convolutions
        self.conv1 = nn.Conv2d(input_size[0], 32, 3, 1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1, bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        # Linear part
        self.fc1 = nn.Linear(4 * 4 * 128, 512, bias=False)
        self.fc2 = nn.Linear(512, hidden_neurons, bias=False)
        self.fc_out = nn.Linear(hidden_neurons, output_neurons, bias=False)  # Out fc

        # LIF cells
        self.lif_conv1 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.25), alpha=alpha))
        self.lif_conv2 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.2), alpha=alpha))
        self.lif_conv3 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.1), alpha=alpha))

        self.lif_fc1 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.1), alpha=alpha))
        self.lif_fc2 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.1), alpha=alpha))
        self.out = LICell()

        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons

        # for m in self.modules():
        #     import math
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        #         variance1 = math.sqrt(2.0 / n)
        #         m.weight.data.normal_(0, variance1)
        #         # define threshold
        #         # m.threshold = 1
        #
        #     elif isinstance(m, nn.Linear):
        #         size = m.weight.size()
        #         fan_in = size[1]  # number of columns
        #         variance2 = math.sqrt(2.0 / fan_in)
        #         m.weight.data.normal_(0.0, variance2)
        #         # define threshold
        #         # m.threshold = 1

    def forward(self, x, flag=None):
        seq_length = x.shape[0]
        batch_size = x.shape[1]

        # Dropout
        drop = nn.Dropout(p=0.5, inplace=True)

        # mask_11 = Variable(torch.ones(batch_size, 64, 32, 32).cuda(), requires_grad=False)
        # mask_11 = drop(mask_11)
        # mask_12 = Variable(torch.ones(batch_size, 64, 32, 32).cuda(), requires_grad=False)
        # mask_12 = drop(mask_12)
        # mask_21 = Variable(torch.ones(batch_size, 128, 16, 16).cuda(), requires_grad=False)
        # mask_21 = drop(mask_21)
        # mask_22 = Variable(torch.ones(batch_size, 128, 16, 16).cuda(), requires_grad=False)
        # mask_22 = drop(mask_22)
        # mask_31 = Variable(torch.ones(batch_size, 256, 8, 8).cuda(), requires_grad=False)
        # mask_31 = drop(mask_31)
        # mask_32 = Variable(torch.ones(batch_size, 256, 8, 8).cuda(), requires_grad=False)
        # mask_32 = drop(mask_32)
        # mask_33 = Variable(torch.ones(batch_size, 256, 8, 8).cuda(), requires_grad=False)
        # mask_33 = drop(mask_33)

        mask_f1 = Variable(torch.ones(batch_size, 512).cuda(), requires_grad=False)
        mask_f1 = drop(mask_f1)
        mask_f2 = Variable(torch.ones(batch_size, self.hidden_neurons).cuda(), requires_grad=False)
        mask_f2 = drop(mask_f2)

        # specify the initial states
        sconv1 = sconv2 = sconv3 = sfc1 = sfc2 = so = None
        # voltages = torch.zeros(
        #     seq_length, batch_size, self.output_neurons, device=x.device, dtype=x.dtype
        # )
        if flag is None:
            for ts in range(seq_length):
                # print(f'Encoder: {(x[ts, :].count_nonzero() / x[ts, :].nelement()) * 100:.3f}%')

                # First convolution
                z = self.conv1(x[ts, :])
                z, sconv1 = self.lif_conv1(z, sconv1)
                z = self.avgpool(z)

                # Second convolution
                z = self.conv2(z)
                z, sconv2 = self.lif_conv2(z, sconv2)
                z = self.avgpool(z)
                # print(f'After conv1: {(z.count_nonzero() / z.nelement()) * 100:.3f}%')

                # Third convolution
                z = self.conv3(z)
                z, sconv3 = self.lif_conv3(z, sconv3)
                z = self.avgpool(z)
                # print(f'After conv2: {(z.count_nonzero() / z.nelement()) * 100:.3f}%')

                # Fully connected part
                z = z.flatten(start_dim=1)

                # First FC
                z = self.fc1(z)
                z, sfc1 = self.lif_fc1(z, sfc1)
                z = torch.mul(z, mask_f1)

                # Second FC
                z = self.fc2(z)
                z, sfc2 = self.lif_fc1(z, sfc2)
                z = torch.mul(z, mask_f2)

                # Fc out
                z = self.fc_out(z)
                v, so = self.out(z, so)

            return v

        elif flag == "hidden_spikes_and_logits":
            hidden_spks = torch.zeros(
                seq_length, batch_size, self.hidden_neurons, device=x.device, dtype=torch.int8
            )
            for ts in range(seq_length):

                # z = self.extract_fatures(x[ts], sconv1, sconv2, sconv3)

                # First convolution
                z = self.conv1(x[ts])
                z, sconv1 = self.lif_conv1(z, sconv1)
                z = self.avgpool(z)

                # Second convolution
                z = self.conv2(z)
                z, sconv2 = self.lif_conv2(z, sconv2)
                z = self.avgpool(z)
                # print(f'After conv1: {(z.count_nonzero() / z.nelement()) * 100:.3f}%')

                # Third convolution
                z = self.conv3(z)
                z, sconv3 = self.lif_conv3(z, sconv3)
                z = self.avgpool(z)
                # print(f'After conv2: {(z.count_nonzero() / z.nelement()) * 100:.3f}%')

                # Fully connected part
                z = z.flatten(start_dim=1)

                # First FC
                z = self.fc1(z)
                z, sfc1 = self.lif_fc1(z, sfc1)
                z = torch.mul(z, mask_f1)

                # Second FC
                z = self.fc2(z)
                z, sfc2 = self.lif_fc2(z, sfc2)
                z = torch.mul(z, mask_f2)
                hidden_spks[ts, :, :] = z

                # Fc out
                z = self.fc_out(z)
                v, so = self.out(z, so)

            return v, hidden_spks


class ConvSNN10(nn.Module):
    def __init__(self, input_size, hidden_neurons, output_neurons, alpha=100):
        # super(ConvNet, self).__init__()
        super().__init__()

        self.ftmaps_h = int(((input_size[1] / 2) / 2) - 2 - 2)
        self.ftmaps_v = int(((input_size[1] / 2) / 2) - 2 - 2)
        # self.ftmaps_h = int(((input_size[1] - 2 - 2) - 2 - 2) - 2 - 2)
        # self.ftmaps_v = int(((input_size[2] - 2 - 2) - 2 - 2) - 2 - 2)

        # Convolutions
        self.conv1 = nn.Conv2d(input_size[0], 32, 3, 1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1, bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        # Linear part
        self.fc1 = nn.Linear(4 * 4 * 128, hidden_neurons, bias=False)
        self.fc_out = nn.Linear(hidden_neurons, output_neurons, bias=False)  # Out fc

        # LIF cells
        self.lif_conv1 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.25), alpha=alpha))
        self.lif_conv2 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.2), alpha=alpha))
        self.lif_conv3 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.1), alpha=alpha))

        self.lif_fc1 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.1), alpha=alpha))
        # self.lif_fc2 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.1), alpha=alpha))
        self.out = LICell()

        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons

        # for m in self.modules():
        #     import math
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        #         variance1 = math.sqrt(2.0 / n)
        #         m.weight.data.normal_(0, variance1)
        #         # define threshold
        #         # m.threshold = 1
        #
        #     elif isinstance(m, nn.Linear):
        #         size = m.weight.size()
        #         fan_in = size[1]  # number of columns
        #         variance2 = math.sqrt(2.0 / fan_in)
        #         m.weight.data.normal_(0.0, variance2)
        #         # define threshold
        #         # m.threshold = 1

    def forward(self, x, flag=None):
        seq_length = x.shape[0]
        batch_size = x.shape[1]

        # Dropout
        drop = nn.Dropout(p=0.5, inplace=True)
        mask_f1 = Variable(torch.ones(batch_size, self.hidden_neurons).cuda(), requires_grad=False)
        mask_f1 = drop(mask_f1)

        # specify the initial states
        sconv1 = sconv2 = sconv3 = sfc1 = so = None
        # voltages = torch.zeros(
        #     seq_length, batch_size, self.output_neurons, device=x.device, dtype=x.dtype
        # )
        if flag is None:
            for ts in range(seq_length):
                # print(f'Encoder: {(x[ts, :].count_nonzero() / x[ts, :].nelement()) * 100:.3f}%')

                # First convolution
                z = self.conv1(x[ts, :])
                z, sconv1 = self.lif_conv1(z, sconv1)
                z = self.avgpool(z)

                # Second convolution
                z = self.conv2(z)
                z, sconv2 = self.lif_conv2(z, sconv2)
                z = self.avgpool(z)
                # print(f'After conv1: {(z.count_nonzero() / z.nelement()) * 100:.3f}%')

                # Third convolution
                z = self.conv3(z)
                z, sconv3 = self.lif_conv3(z, sconv3)
                z = self.avgpool(z)
                # print(f'After conv2: {(z.count_nonzero() / z.nelement()) * 100:.3f}%')

                # Fully connected part
                z = z.flatten(start_dim=1)

                # First FC
                z = self.fc1(z)
                z, sfc1 = self.lif_fc1(z, sfc1)
                z = torch.mul(z, mask_f1)

                # Fc out
                z = self.fc_out(z)
                v, so = self.out(z, so)

            return v

        elif flag == "hidden_spikes_and_logits":
            hidden_spks = torch.zeros(
                seq_length, batch_size, self.hidden_neurons, device=x.device, dtype=torch.int8
            )
            for ts in range(seq_length):

                # z = self.extract_fatures(x[ts], sconv1, sconv2, sconv3)

                # First convolution
                z = self.conv1(x[ts])
                z, sconv1 = self.lif_conv1(z, sconv1)
                z = self.avgpool(z)

                # Second convolution
                z = self.conv2(z)
                z, sconv2 = self.lif_conv2(z, sconv2)
                z = self.avgpool(z)
                # print(f'After conv1: {(z.count_nonzero() / z.nelement()) * 100:.3f}%')

                # Third convolution
                z = self.conv3(z)
                z, sconv3 = self.lif_conv3(z, sconv3)
                z = self.avgpool(z)
                # print(f'After conv2: {(z.count_nonzero() / z.nelement()) * 100:.3f}%')

                # Fully connected part
                z = z.flatten(start_dim=1)

                # First FC
                z = self.fc1(z)
                z, sfc1 = self.lif_fc1(z, sfc1)
                z = torch.mul(z, mask_f1)
                hidden_spks[ts, :, :] = z

                # Fc out
                z = self.fc_out(z)
                v, so = self.out(z, so)

            return v, hidden_spks


class ConvSNN11_no_dropout(nn.Module):
    def __init__(self, input_size, hidden_neurons, output_neurons, alpha=100):
        # super(ConvNet, self).__init__()
        super().__init__()

        self.ftmaps_h = int(((input_size[1] / 2) / 2) - 2 - 2)
        self.ftmaps_v = int(((input_size[1] / 2) / 2) - 2 - 2)
        # self.ftmaps_h = int(((input_size[1] - 2 - 2) - 2 - 2) - 2 - 2)
        # self.ftmaps_v = int(((input_size[2] - 2 - 2) - 2 - 2) - 2 - 2)

        # Convolutions
        self.conv1 = nn.Conv2d(input_size[0], 32, 3, 1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1, bias=False)
        self.avgpool = nn.AvgPool2d(kernel_size=2)

        # Linear part
        self.fc1 = nn.Linear(4 * 4 * 128, hidden_neurons, bias=False)
        self.fc_out = nn.Linear(hidden_neurons, output_neurons, bias=False)  # Out fc

        # LIF cells
        self.lif_conv1 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.25), alpha=alpha))
        self.lif_conv2 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.2), alpha=alpha))
        self.lif_conv3 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.1), alpha=alpha))

        self.lif_fc1 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.1), alpha=alpha))
        # self.lif_fc2 = LIFCell(p=LIFParameters(v_th=torch.tensor(0.1), alpha=alpha))
        self.out = LICell()

        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons

        # for m in self.modules():
        #     import math
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        #         variance1 = math.sqrt(2.0 / n)
        #         m.weight.data.normal_(0, variance1)
        #         # define threshold
        #         # m.threshold = 1
        #
        #     elif isinstance(m, nn.Linear):
        #         size = m.weight.size()
        #         fan_in = size[1]  # number of columns
        #         variance2 = math.sqrt(2.0 / fan_in)
        #         m.weight.data.normal_(0.0, variance2)
        #         # define threshold
        #         # m.threshold = 1

    def forward(self, x, flag=None):
        seq_length = x.shape[0]
        batch_size = x.shape[1]

        # # Dropout
        # drop = nn.Dropout(p=0.5, inplace=True)
        # mask_f1 = Variable(torch.ones(batch_size, self.hidden_neurons).cuda(), requires_grad=False)
        # mask_f1 = drop(mask_f1)

        # specify the initial states
        sconv1 = sconv2 = sconv3 = sfc1 = so = None
        # voltages = torch.zeros(
        #     seq_length, batch_size, self.output_neurons, device=x.device, dtype=x.dtype
        # )
        if flag is None:
            for ts in range(seq_length):
                # print(f'Encoder: {(x[ts, :].count_nonzero() / x[ts, :].nelement()) * 100:.3f}%')

                # First convolution
                z = self.conv1(x[ts, :])
                z, sconv1 = self.lif_conv1(z, sconv1)
                z = self.avgpool(z)

                # Second convolution
                z = self.conv2(z)
                z, sconv2 = self.lif_conv2(z, sconv2)
                z = self.avgpool(z)
                # print(f'After conv1: {(z.count_nonzero() / z.nelement()) * 100:.3f}%')

                # Third convolution
                z = self.conv3(z)
                z, sconv3 = self.lif_conv3(z, sconv3)
                z = self.avgpool(z)
                # print(f'After conv2: {(z.count_nonzero() / z.nelement()) * 100:.3f}%')

                # Fully connected part
                z = z.flatten(start_dim=1)

                # First FC
                z = self.fc1(z)
                z, sfc1 = self.lif_fc1(z, sfc1)
                # z = torch.mul(z, mask_f1)

                # Fc out
                z = self.fc_out(z)
                v, so = self.out(z, so)

            return v

        elif flag == "hidden_spikes_and_logits":
            hidden_spks = torch.zeros(
                seq_length, batch_size, self.hidden_neurons, device=x.device, dtype=torch.int8
            )
            for ts in range(seq_length):

                # z = self.extract_fatures(x[ts], sconv1, sconv2, sconv3)

                # First convolution
                z = self.conv1(x[ts])
                z, sconv1 = self.lif_conv1(z, sconv1)
                z = self.avgpool(z)

                # Second convolution
                z = self.conv2(z)
                z, sconv2 = self.lif_conv2(z, sconv2)
                z = self.avgpool(z)
                # print(f'After conv1: {(z.count_nonzero() / z.nelement()) * 100:.3f}%')

                # Third convolution
                z = self.conv3(z)
                z, sconv3 = self.lif_conv3(z, sconv3)
                z = self.avgpool(z)
                # print(f'After conv2: {(z.count_nonzero() / z.nelement()) * 100:.3f}%')

                # Fully connected part
                z = z.flatten(start_dim=1)

                # First FC
                z = self.fc1(z)
                z, sfc1 = self.lif_fc1(z, sfc1)
                # z = torch.mul(z, mask_f1)
                hidden_spks[ts, :, :] = z

                # Fc out
                z = self.fc_out(z)
                v, so = self.out(z, so)

            return v, hidden_spks