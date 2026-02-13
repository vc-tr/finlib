"""
Temporal Convolutional Network (TCN) for time series forecasting.

Uses dilated causal convolutions for long-range dependencies.
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """Remove last elements to ensure causal convolution."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = self.downsample(x)
        return self.relu(out + res)


class PriceTCN(nn.Module):
    """
    Temporal Convolutional Network for price forecasting.
    """

    def __init__(
        self,
        input_dim: int = 1,
        n_channels: int = 64,
        kernel_size: int = 3,
        num_levels: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        num_channels = [input_dim] + [n_channels] * num_levels
        for i in range(num_levels):
            dilation_size = 2**i
            in_ch = num_channels[i]
            out_ch = num_channels[i + 1]
            padding = (kernel_size - 1) * dilation_size
            layers.append(
                TemporalBlock(
                    in_ch, out_ch, kernel_size, stride=1, dilation=dilation_size,
                    padding=padding, dropout=dropout,
                )
            )
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(n_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim)
        returns: (batch, 1)
        """
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        x = self.tcn(x)  # (batch, n_channels, seq_len)
        x = x[:, :, -1]  # last time step
        return self.head(x)
