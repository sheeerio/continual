import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import misc
from config import get_parser

config = get_parser().parse_args()


class MLP(nn.Module):
    def __init__(self, i, h, o):
        super().__init__()
        self.fc1 = nn.Linear(i, h)
        if config.activation in ("crelu", "fourier", "cleaky_relu"):
            self.fc2 = nn.Linear(2 * h, h)
            # self.fc3 = nn.Linear(2 * h, h)
            self.fc4 = nn.Linear(2 * h, o)
        else:
            self.fc2 = nn.Linear(h, h)
            # self.fc3 = nn.Linear(h, h)
            self.fc4 = nn.Linear(h, o)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if config.initialization == "kaiming":
                    nn.init.kaiming_uniform_(
                        m.weight,
                        a=config.alpha if config.activation == "adalin" else 0,
                        nonlinearity=(misc.activation_map[config.activation]),
                    )
                elif config.initialization == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif config.initialization == "normal":
                    nn.init.normal_(
                        m.weight, mean=config.normal_mean, std=config.normal_std
                    )
                elif config.initialization == "uniform":
                    nn.init.uniform_(m.weight, a=config.uniform_a, b=config.uniform_b)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if config.activation == "relu":
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            # x = F.relu(self.fc3(x))
        elif config.activation == "leaky_relu":
            x = F.leaky_relu(self.fc1(x))
            x = F.leaky_relu(self.fc2(x))
            # x = F.leaky_relu(self.fc3(x))
        elif config.activation == "tanh":
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            # x = torch.tanh(self.fc3(x))
        elif config.activation == "crelu":
            x1 = self.fc1(x)
            x = torch.cat([F.relu(x1), F.relu(-x1)], 1)
            x2 = self.fc2(x)
            x = torch.cat([F.relu(x2), F.relu(-x2)], 1)
            # x3 = self.fc3(x)
            # x = torch.cat([F.relu(x3), F.relu(-x3)], 1)
        elif config.activation == "cleaky_relu":
            x = self.fc1(x)
            x = torch.cat([F.leaky_relu(x), F.leaky_relu(-x)], 1)
            x = self.fc2(x)
            x = torch.cat([F.leaky_relu(x), F.leaky_relu(-x)], 1)
            # x = self.fc3(x)
            # x = torch.cat([F.leaky_relu(x), F.leaky_relu(-x)], 1)
        elif config.activation == "adalin":
            x = F.leaky_relu(self.fc1(x), negative_slope=config.alpha)
            x = F.leaky_relu(self.fc2(x), negative_slope=config.alpha)
            # x = F.leaky_relu(self.fc3(x), negative_slope=config.alpha)
        elif config.activation == "fourier":
            x = torch.cat(
                [torch.sin(self.fc1(x * 5.0)), torch.cos(self.fc1(x * 5.0))], 1
            )
            x = torch.cat(
                [torch.sin(self.fc2(x * 5.0)), torch.cos(self.fc2(x * 5.0))], 1
            )
            # x = torch.cat(
            #     [torch.sin(self.fc3(x * 5.0)), torch.cos(self.fc3(x * 5.0))], 1
            # )
        elif config.activation == "softplus":
            x = F.softplus(self.fc1(x))
            x = F.softplus(self.fc2(x))
            # x = F.softplus(self.fc3(x))
        elif config.activation == "swish":
            x = self.fc1(x) * torch.sigmoid(self.fc1(x))
            x = self.fc2(x) * torch.sigmoid(self.fc2(x))
            # x = self.fc3(x) * torch.sigmoid(self.fc3(x))
        else:
            x = self.fc1(x)
            x = self.fc2(x)
            # x = self.fc3(x)
        return self.fc4(x)


class BatchNormMLP(nn.Module):
    def __init__(self, i, h, o):
        super().__init__()
        self.fc1 = nn.Linear(i, h)
        self.bn1 = nn.BatchNorm1d(h)
        if config.activation in ("crelu", "fourier", "cleaky_relu"):
            self.fc2 = nn.Linear(2 * h, h)
            # self.fc3 = nn.Linear(2 * h, h)
            # self.fc4 = nn.Linear(2 * h, h)
            self.fc5 = nn.Linear(2 * h, o)
            self.bn2 = nn.BatchNorm1d(2 * h)
            # self.bn3 = nn.BatchNorm1d(2 * h)
            self.bn4 = nn.BatchNorm1d(h)
        else:
            self.fc2 = nn.Linear(h, h)
            # self.fc3 = nn.Linear(h, h)
            # self.fc4 = nn.Linear(h, h)
            self.fc5 = nn.Linear(h, o)
        self.bn2 = nn.BatchNorm1d(h)
        # self.bn3 = nn.BatchNorm1d(h)
        # self.bn4 = nn.BatchNorm1d(h)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if config.initialization == "kaiming":
                    nn.init.kaiming_uniform_(
                        m.weight,
                        a=config.alpha if config.activation == "adalin" else 0,
                        nonlinearity=misc.activation_map[config.activation],
                    )
                elif config.initialization == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif config.initialization == "normal":
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                elif config.initialization == "uniform":
                    nn.init.uniform_(m.weight, a=-0.1, b=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if config.activation == "relu":
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
            # x = F.relu(self.bn3(self.fc3(x)))
            # x = F.relu(self.bn4(self.fc4(x)))
        elif config.activation == "leaky_relu":
            x = F.leaky_relu(self.bn1(self.fc1(x)))
            x = F.leaky_relu(self.bn2(self.fc2(x)))
            # x = F.leaky_relu(self.bn3(self.fc3(x)))
            # x = F.leaky_relu(self.bn4(self.fc4(x)))
        elif config.activation == "adalin":
            x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=config.alpha)
            x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=config.alpha)
            # x = F.leaky_relu(self.bn3(self.fc3(x)), negative_slope=config.alpha)
            # x = F.leaky_relu(self.bn4(self.fc4(x)), negative_slope=config.alpha)
        elif config.activation == "crelu":
            x1 = self.fc1(x)
            x = torch.cat([F.relu(self.bn1(x1)), F.relu(-x1)], 1)
            x2 = self.fc2(x)
            x = torch.cat([F.relu(self.bn2(x2)), F.relu(-x2)], 1)
            # x3 = self.fc3(x)
            # x = torch.cat([F.relu(self.bn3(x3)), F.relu(-x3)], 1)
            # x4 = self.fc4(x)
            # x = torch.cat([F.relu(self.bn4(x4)), F.relu(-x4)], 1)
        elif config.activation == "softplus":
            x = F.softplus(self.bn1(self.fc1(x)))
            x = F.softplus(self.bn2(self.fc2(x)))
            # x = F.softplus(self.bn3(self.fc3(x)))
            # x = F.softplus(self.bn4(self.fc4(x)))
        elif config.activation == "swish":
            x = self.fc1(x) * torch.sigmoid(self.bn1(self.fc1(x)))
            x = self.fc2(x) * torch.sigmoid(self.bn2(self.fc2(x)))
            # x = self.fc3(x) * torch.sigmoid(self.bn3(self.fc3(x)))
            # x = self.fc4(x) * torch.sigmoid(self.bn4(self.fc4(x)))
        return self.fc5(x)
