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
