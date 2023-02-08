import torch
from norse.torch import LIFParameters
from norse.torch.module.lif import LIFCell
from norse.torch import LICell


class FCSNN1(torch.nn.Module):
    def __init__(self, input_features, hidden_features, output_features, dt=0.001):
        super().__init__()

        # Hidden layer
        self.fc1 = torch.nn.Linear(input_features, hidden_features, bias=False)
        self.lif1 = LIFCell(
            p=LIFParameters(alpha=100, v_th=torch.tensor(0.5)),
            dt=dt
        )

        # Output layer
        self.fc_out = torch.nn.Linear(hidden_features, output_features, bias=False)
        self.out = LICell(dt=dt)

        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_neurons = output_features

    def forward(self, x, flag=None):
        # Input shape = [time_step,batch_size, ... , ... , ... ]
        seq_length, batch_size, _, _, _ = x.shape
        s1 = so = None
        # voltages = []

        voltages = torch.zeros(
            seq_length, batch_size, self.output_neurons,
            device=x.device, dtype=x.dtype
        )
        if flag is None:
            for ts in range(seq_length):
                # Flatten the input to [batch_size, input_features]
                z = x[ts, :, :, :].view(-1, self.input_features)

                # Hidden layer
                z = self.fc1(z)
                z, s1 = self.lif1(z, s1)

                # Output layer
                z = self.fc_out(z)
                vo, so = self.out(z, so)
                voltages[ts] = vo
            # The max across all time steps is the logit, the first dimension
            # [time_step, batch_size, output_neurons]
            voltages, _ = torch.max(voltages, 0)
            return voltages

        elif flag == "hidden_spikes_and_logits":
            hdn_spikes = torch.zeros(
                seq_length, batch_size, self.hidden_features,
                device=x.device, dtype=x.dtype
            )
            for ts in range(seq_length):
                # Flatten the input to [batch_size, input_features]
                z = x[ts, :, :, :].view(-1, self.input_features)

                # Hidden layer
                z = self.fc1(z)
                z, s1 = self.lif1(z, s1)
                hdn_spikes[ts] = z

                # Output layer
                z = self.fc_out(z)
                vo, so = self.out(z, so)
                voltages[ts] = vo

            # The max across all time steps is the logit, the first dimension
            # [time_step, batch_size, output_neurons]
            voltages, _ = torch.max(voltages, 0)
        else:
            raise NameError('Wrong flag')

        return voltages, hdn_spikes


class FCSNN2(torch.nn.Module):
    def __init__(self, input_features, hidden_features, output_features, dt=0.001):
        super().__init__()

        # Hidden layer
        self.fc1 = torch.nn.Linear(input_features, 300, bias=False)
        self.fc2 = torch.nn.Linear(300, hidden_features, bias=False)
        self.lif1 = LIFCell(
            p=LIFParameters(alpha=100, v_th=torch.tensor(0.25)),
            dt=dt
        )
        self.lif2 = LIFCell(
            p=LIFParameters(alpha=100, v_th=torch.tensor(0.25)),
            dt=dt
        )

        # Output layer
        self.fc_out = torch.nn.Linear(hidden_features, output_features, bias=False)
        self.out = LICell(dt=dt)

        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_neurons = output_features

    def forward(self, x, flag=None):
        # Input shape = [time_step,batch_size, ... , ... , ... ]
        seq_length, batch_size, _, _, _ = x.shape
        s1 = s2 = so = None
        # voltages = []

        voltages = torch.zeros(
            seq_length, batch_size, self.output_neurons,
            device=x.device, dtype=x.dtype
        )
        if flag is None:
            for ts in range(seq_length):
                # Flatten the input to [batch_size, input_features]
                z = x[ts, :, :, :].view(-1, self.input_features)

                # Hidden layer 1
                z = self.fc1(z)
                z, s1 = self.lif1(z, s1)

                # Hidden layer 2
                z = self.fc2(z)
                z, s2 = self.lif2(z, s2)

                # Output layer
                z = self.fc_out(z)
                vo, so = self.out(z, so)
                voltages[ts] = vo
            # The max across all time steps is the logit, the first dimension
            # [time_step, batch_size, output_neurons]
            voltages, _ = torch.max(voltages, 0)
            return voltages

        elif flag == "hidden_spikes_and_logits":
            hdn_spikes = torch.zeros(
                seq_length, batch_size, self.hidden_features,
                device=x.device, dtype=x.dtype
            )
            for ts in range(seq_length):
                # Flatten the input to [batch_size, input_features]
                z = x[ts, :, :, :].view(-1, self.input_features)

                # Hidden layer 1
                z = self.fc1(z)
                z, s1 = self.lif1(z, s1)
                # hdn_spikes[ts] = z

                # Hidden layer 2
                z = self.fc2(z)
                z, s2 = self.lif2(z, s2)
                hdn_spikes[ts] = z

                # Output layer
                z = self.fc_out(z)
                vo, so = self.out(z, so)
                voltages[ts] = vo
            # The max across all time steps is the logit, the first dimension
            # [time_step, batch_size, output_neurons]
            voltages, _ = torch.max(voltages, 0)
        else:
            raise NameError('Wrong flag')

        return voltages, hdn_spikes


class FCSNN3(torch.nn.Module):
    def __init__(self, input_features, hidden_features, output_features, dt=0.001):
        super().__init__()

        # Hidden layer
        self.fc1 = torch.nn.Linear(input_features, hidden_features, bias=False)
        self.lif1 = LIFCell(
            p=LIFParameters(alpha=100, v_th=torch.tensor(0.25)),
            dt=dt
        )

        # Output layer
        self.fc_out = torch.nn.Linear(hidden_features, output_features, bias=False)
        self.out = LICell(dt=dt)

        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_neurons = output_features

    def forward(self, x, flag=None):
        # Input shape = [time_step,batch_size, ... , ... , ... ]
        seq_length, batch_size, _ = x.shape
        s1 = s2 = so = None
        if flag is None:
            for ts in range(seq_length):
                # Flatten the input to [batch_size, input_features]
                z = x[ts]

                # Hidden layer 1
                z = self.fc1(z)
                z, s1 = self.lif1(z, s1)

                # Output layer
                z = self.fc_out(z)
                vo, so = self.out(z, so)
            return vo

        elif flag == "hidden_spikes_and_logits":
            hdn_spikes = torch.zeros(
                seq_length, batch_size, self.hidden_features,
                device=x.device, dtype=x.dtype
            )
            for ts in range(seq_length):
                # Flatten the input to [batch_size, input_features]
                z = x[ts, :, :, :].view(-1, self.input_features)

                # Hidden layer 1
                z = self.fc1(z)
                z, s1 = self.lif1(z, s1)
                hdn_spikes[ts] = z

                # Output layer
                z = self.fc_out(z)
                vo, so = self.out(z, so)
            #     voltages[ts] = vo
            # # The max across all time steps is the logit, the first dimension
            # # [time_step, batch_size, output_neurons]
            # voltages, _ = torch.max(voltages, 0)
        else:
            raise NameError('Wrong flag')

        return vo, hdn_spikes


class FCSNN4(torch.nn.Module):
    """
    2 linear layers with 1024 neurons in the first one
    """
    def __init__(self, input_features, hidden_features, output_features, dt=0.001):
        super().__init__()

        # Linear layers
        self.fc1 = torch.nn.Linear(input_features, 1024, bias=False)
        self.fc2 = torch.nn.Linear(1024, hidden_features, bias=False)
        self.fc_out = torch.nn.Linear(hidden_features, output_features, bias=False)

        # Neurons
        self.lif1 = LIFCell(p=LIFParameters(alpha=100, v_th=torch.tensor(0.25)), dt=dt)
        self.lif2 = LIFCell(p=LIFParameters(alpha=100, v_th=torch.tensor(0.25)), dt=dt)
        self.out = LICell(dt=dt)

        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_neurons = output_features

    def forward(self, x, flag=None):
        # Input shape = [time_step,batch_size, ... , ... , ... ]
        seq_length, batch_size, _ = x.shape
        s1 = s2 = so = None
        if flag is None:
            for ts in range(seq_length):
                # Flatten the input to [batch_size, input_features]
                z = x[ts]

                # Hidden layer 1
                z = self.fc1(z)
                z, s1 = self.lif1(z, s1)

                # Hidden layer 2
                z = self.fc2(z)
                z, s2 = self.lif2(z, s2)

                # Output layer
                z = self.fc_out(z)
                vo, so = self.out(z, so)
            return vo

        elif flag == "hidden_spikes_and_logits":
            hdn_spikes = torch.zeros(
                seq_length, batch_size, self.hidden_features,
                device=x.device, dtype=x.dtype
            )
            for ts in range(seq_length):
                # Flatten the input to [batch_size, input_features]
                z = x[ts, :, :, :].view(-1, self.input_features)

                # Hidden layer 1
                z = self.fc1(z)
                z, s1 = self.lif1(z, s1)

                # Hidden layer 2
                z = self.fc2(z)
                z, s2 = self.lif2(z, s2)
                hdn_spikes[ts] = z

                # Output layer
                z = self.fc_out(z)
                vo, so = self.out(z, so)
            #     voltages[ts] = vo
            # # The max across all time steps is the logit, the first dimension
            # # [time_step, batch_size, output_neurons]
            # voltages, _ = torch.max(voltages, 0)
        else:
            raise NameError('Wrong flag')

        return vo, hdn_spikes


class FCSNN5(torch.nn.Module):
    """
    As FC4 but with more threshold (2 linear layers with 1024 neurons in the first one)
    """
    def __init__(self, input_features, hidden_features, output_features, dt=0.001):
        super().__init__()

        self.thr = 1

        # Linear layers
        self.fc1 = torch.nn.Linear(input_features, 1024, bias=False)
        self.fc2 = torch.nn.Linear(1024, hidden_features, bias=False)
        self.fc_out = torch.nn.Linear(hidden_features, output_features, bias=False)

        # Neurons
        self.lif1 = LIFCell(p=LIFParameters(alpha=100, v_th=torch.tensor(self.thr)), dt=dt)
        self.lif2 = LIFCell(p=LIFParameters(alpha=100, v_th=torch.tensor(self.thr)), dt=dt)
        self.out = LICell(dt=dt)

        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_neurons = output_features

    def forward(self, x, flag=None):
        # Input shape = [time_step,batch_size, ... , ... , ... ]
        seq_length, batch_size, _ = x.shape
        s1 = s2 = so = None
        if flag is None:
            for ts in range(seq_length):
                # Flatten the input to [batch_size, input_features]
                z = x[ts]

                # Hidden layer 1
                z = self.fc1(z)
                z, s1 = self.lif1(z, s1)

                # Hidden layer 2
                z = self.fc2(z)
                z, s2 = self.lif2(z, s2)

                # Output layer
                z = self.fc_out(z)
                vo, so = self.out(z, so)
            return vo

        elif flag == "hidden_spikes_and_logits":
            hdn_spikes = torch.zeros(
                seq_length, batch_size, self.hidden_features,
                device=x.device, dtype=x.dtype
            )
            for ts in range(seq_length):
                # Flatten the input to [batch_size, input_features]
                z = x[ts, :, :, :].view(-1, self.input_features)

                # Hidden layer 1
                z = self.fc1(z)
                z, s1 = self.lif1(z, s1)

                # Hidden layer 2
                z = self.fc2(z)
                z, s2 = self.lif2(z, s2)
                hdn_spikes[ts] = z

                # Output layer
                z = self.fc_out(z)
                vo, so = self.out(z, so)
        else:
            raise NameError('Wrong flag')

        return vo, hdn_spikes


class FCSNN6(torch.nn.Module):
    """
    As FC5 but with 3 linear layers with 512-256 neurons
    """
    def __init__(self, input_features, hidden_features, output_features, dt=0.001):
        super().__init__()

        self.thr = 1

        # Linear layers
        self.fc1 = torch.nn.Linear(input_features, 512, bias=False)
        self.fc2 = torch.nn.Linear(512, 256, bias=False)
        self.fc3 = torch.nn.Linear(256, hidden_features, bias=False)  # The idea is to use 128
        self.fc_out = torch.nn.Linear(hidden_features, output_features, bias=False)

        # Neurons
        self.lif1 = LIFCell(p=LIFParameters(alpha=100, v_th=torch.tensor(self.thr)), dt=dt)
        self.lif2 = LIFCell(p=LIFParameters(alpha=100, v_th=torch.tensor(self.thr)), dt=dt)
        self.lif3 = LIFCell(p=LIFParameters(alpha=100, v_th=torch.tensor(self.thr)), dt=dt)
        self.out = LICell(dt=dt)

        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_neurons = output_features

    def forward(self, x, flag=None):
        # Input shape = [time_step,batch_size, ... , ... , ... ]
        seq_length, batch_size, _ = x.shape
        s1 = s2 = s3 = so = None
        if flag is None:
            for ts in range(seq_length):
                # Flatten the input to [batch_size, input_features]
                z = x[ts]

                # Hidden layer 1
                z = self.fc1(z)
                z, s1 = self.lif1(z, s1)

                # Hidden layer 2
                z = self.fc2(z)
                z, s2 = self.lif2(z, s2)

                # Hidden layer 3
                z = self.fc3(z)
                z, s3 = self.lif3(z, s3)

                # Output layer
                z = self.fc_out(z)
                vo, so = self.out(z, so)
            return vo

        elif flag == "hidden_spikes_and_logits":
            hdn_spikes = torch.zeros(
                seq_length, batch_size, self.hidden_features,
                device=x.device, dtype=x.dtype
            )
            for ts in range(seq_length):
                # Flatten the input to [batch_size, input_features]
                z = x[ts, :, :, :].view(-1, self.input_features)

                # Hidden layer 1
                z = self.fc1(z)
                z, s1 = self.lif1(z, s1)

                # Hidden layer 2
                z = self.fc2(z)
                z, s2 = self.lif2(z, s2)

                # Hidden layer 3
                z = self.fc3(z)
                z, s3 = self.lif3(z, s3)
                hdn_spikes[ts] = z

                # Output layer
                z = self.fc_out(z)
                vo, so = self.out(z, so)
        else:
            raise NameError('Wrong flag')

        return vo, hdn_spikes


class FCSNN7(torch.nn.Module):
    """
    Bigger model, 4 linear layers (8192-2048-512-128, dividing by 4 all in each layer)
    """
    def __init__(self, input_features, hidden_features, output_features, dt=0.001):
        super().__init__()

        self.thr = 0.5

        # Linear layers
        self.fc1 = torch.nn.Linear(input_features, 8192, bias=False)
        self.fc2 = torch.nn.Linear(8192, 2048, bias=False)
        self.fc3 = torch.nn.Linear(2048, 512, bias=False)
        self.fc4 = torch.nn.Linear(512, hidden_features, bias=False)  # The idea is to use 128
        self.fc_out = torch.nn.Linear(hidden_features, output_features, bias=False)

        # Neurons
        self.lif1 = LIFCell(p=LIFParameters(alpha=100, v_th=torch.tensor(self.thr)), dt=dt)
        self.lif2 = LIFCell(p=LIFParameters(alpha=100, v_th=torch.tensor(self.thr)), dt=dt)
        self.lif3 = LIFCell(p=LIFParameters(alpha=100, v_th=torch.tensor(self.thr)), dt=dt)
        self.lif4 = LIFCell(p=LIFParameters(alpha=100, v_th=torch.tensor(self.thr)), dt=dt)
        self.out = LICell(dt=dt)

        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_neurons = output_features

    def forward(self, x, flag=None):
        # Input shape = [time_step,batch_size, ... , ... , ... ]
        seq_length, batch_size, _ = x.shape
        s1 = s2 = s3 = s4 = so = None
        if flag is None:
            for ts in range(seq_length):
                # Flatten the input to [batch_size, input_features]
                z = x[ts]

                # Hidden layer 1
                z = self.fc1(z)
                z, s1 = self.lif1(z, s1)

                # Hidden layer 2
                z = self.fc2(z)
                z, s2 = self.lif2(z, s2)

                # Hidden layer 3
                z = self.fc3(z)
                z, s3 = self.lif3(z, s3)

                # Hidden layer 4
                z = self.fc4(z)
                z, s4 = self.lif4(z, s4)

                # Output layer
                z = self.fc_out(z)
                vo, so = self.out(z, so)
            return vo

        elif flag == "hidden_spikes_and_logits":
            hdn_spikes = torch.zeros(
                seq_length, batch_size, self.hidden_features,
                device=x.device, dtype=x.dtype
            )
            for ts in range(seq_length):
                # Flatten the input to [batch_size, input_features]
                z = x[ts, :, :, :].view(-1, self.input_features)

                # Hidden layer 1
                z = self.fc1(z)
                z, s1 = self.lif1(z, s1)

                # Hidden layer 2
                z = self.fc2(z)
                z, s2 = self.lif2(z, s2)

                # Hidden layer 3
                z = self.fc3(z)
                z, s3 = self.lif3(z, s3)

                # Hidden layer 4
                z = self.fc4(z)
                z, s4 = self.lif4(z, s4)
                hdn_spikes[ts] = z

                # Output layer
                z = self.fc_out(z)
                vo, so = self.out(z, so)
        else:
            raise NameError('Wrong flag')

        return vo, hdn_spikes


class FC8(torch.nn.Module):
    def __init__(self, input_features, hidden_features, output_features, dt=0.001):
        super().__init__()
        from torch.nn.functional import relu
        # Linear layers
        self.fc1 = torch.nn.Linear(250, 8192, bias=False)
        self.fc2 = torch.nn.Linear(8192, 2048, bias=False)
        self.fc3 = torch.nn.Linear(2048, 512, bias=False)
        self.fc4 = torch.nn.Linear(512, hidden_features, bias=False)  # The idea is to use 128
        self.fc_out = torch.nn.Linear(hidden_features, output_features, bias=False)

        self.relu = relu

    def forward(self, z):
        z = self.relu(self.fc1(z))
        z = self.relu(self.fc2(z))
        z = self.relu(self.fc3(z))
        z = self.relu(self.fc4(z))
        z = self.fc_out(z)

        return z