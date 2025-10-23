import torch
import zuko

from config import MLConfig
ml_config = MLConfig()


class ConvTrunk(torch.nn.Module):
    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, 8, kernel_size=7, padding=3)
        self.pool1 = torch.nn.MaxPool1d(4)
        self.conv2 = torch.nn.Conv1d(8, 16, kernel_size=3, padding=1)
        self.pool2 = torch.nn.MaxPool1d(4)
        self.fc1 = torch.nn.Linear(16 * 64, 128)
        self.activation = torch.nn.Softplus()

    def forward(self, x):
        ## Necessary for in_channels=1 (spatially-averaged spectrum)
        ## (x from data loader is [batch, 1024] but has to be [batch, 1, 1024] for conv1)
        #if x.dim() == 2:
        #    x = x.unsqueeze(1)

        x = self.activation(self.conv1(x))
        x = self.pool1(x)
        x = self.activation(self.conv2(x))
        x = self.pool2(x)
        x = x.flatten(1)
        return self.activation(self.fc1(x))

class ConvSpectraFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = ConvTrunk()
        self.nsf = zuko.flows.NSF(features=ml_config.dim_output_parameters,
                                  context=128, transforms=3, hidden_features=(256,))

    def nll(self, x, y):
        context = self.trunk(x)
        q_dist = self.nsf(context)
        log_q = q_dist.log_prob(y)
        return -log_q.mean()

    def forward(self, x: torch.Tensor) -> zuko.distributions.NormalizingFlow:
        c = self.trunk(x)
        return self.nsf(c)


#################################################
# More architectures below for testing purposes #
#################################################

class ConvSpectraNet(torch.nn.Module):
    def __init__(self, in_channels: int = 4, output_size: int = ml_config.dim_output_parameters * 2):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, 8, kernel_size=7, padding=3)
        self.pool1 = torch.nn.MaxPool1d(kernel_size=4)

        self.conv2 = torch.nn.Conv1d(8, 16, kernel_size=3, padding=1)
        self.pool2 = torch.nn.MaxPool1d(kernel_size=4)

        self.fc1 = torch.nn.Linear(16 * 64, 128)
        self.fc2 = torch.nn.Linear(128, output_size)

        self.activation = torch.nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x from data loader is [batch, 1024] but has to be [batch, 1, 1024] for conv1
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.activation(self.conv1(x))  # [batch,  8, 1024]
        x = self.pool1(x)                   # [batch,  8,  256]

        x = self.activation(self.conv2(x))  # [batch, 16,  256]
        x = self.pool2(x)                   # [batch, 16,   64]

        x = x.flatten(start_dim=1)          # [batch, 16*64=1024]

        x = self.activation(self.fc1(x))    # [batch, 128]
        x = self.fc2(x)                     # [batch, output_size]
        return x

class pileupNN_parameter_estimator(torch.nn.Module):
    def __init__(self, input_size=ml_config.dim_input_parameters,
                 hidden_size=128,
                 output_size=ml_config.dim_output_parameters):
        super(pileupNN_parameter_estimator, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc4 = torch.nn.Linear(hidden_size, output_size)

        self.activation = torch.nn.Softplus()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)  # Linear combination
        return x

class PileupNN_spectral_estimator(torch.nn.Module):
    def __init__(self, input_size=ml_config.dim_input_parameters,
                 hidden_size=256,
                 output_size=ml_config.dim_input_parameters):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc4 = torch.nn.Linear(hidden_size, output_size)

        self.activation = torch.nn.Softplus()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)  # Linear combination
        x = self.activation(x)  # activation on final output to ensure non-negative counts (Poisson likelihood)
        return x

class pileupNN_variance_estimator(torch.nn.Module):
    def __init__(self, input_size=ml_config.dim_input_parameters,
                 hidden_size=256,
                 output_size=ml_config.dim_output_parameters * 2):
        # Output dimensions: First half are the model parameters, the last half are the log variance
        super(pileupNN_variance_estimator, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc4 = torch.nn.Linear(hidden_size, output_size)

        self.activation = torch.nn.Softplus()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
