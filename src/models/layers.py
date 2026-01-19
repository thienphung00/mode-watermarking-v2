"""
Shared neural network layers for detector models.

Common building blocks used across different detector architectures.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Double convolution block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU.

    Standard building block for U-Net style architectures.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        batch_norm: bool = True,
        dropout: float = 0.0,
    ):
        """
        Initialize double convolution block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            mid_channels: Number of intermediate channels (default: out_channels)
            batch_norm: Whether to use batch normalization
            dropout: Dropout probability
        """
        super().__init__()

        # Ensure all channel values are integers
        assert isinstance(in_channels, int), f"in_channels must be int, got {type(in_channels)}"
        assert isinstance(out_channels, int), f"out_channels must be int, got {type(out_channels)}"
        
        if mid_channels is None:
            mid_channels = out_channels
        else:
            assert isinstance(mid_channels, int), f"mid_channels must be int, got {type(mid_channels)}"
        
        assert mid_channels > 0, f"mid_channels must be positive, got {mid_channels}"
        assert out_channels > 0, f"out_channels must be positive, got {out_channels}"

        layers = []

        # First conv
        layers.append(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1))
        if batch_norm:
            layers.append(nn.BatchNorm2d(mid_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        # Second conv
        layers.append(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.conv(x)


class DownBlock(nn.Module):
    """Downsampling block for U-Net encoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        batch_norm: bool = True,
        dropout: float = 0.0,
    ):
        """
        Initialize downsampling block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            batch_norm: Whether to use batch normalization
            dropout: Dropout probability
        """
        super().__init__()

        # Ensure channel values are integers
        assert isinstance(in_channels, int), f"in_channels must be int, got {type(in_channels)}"
        assert isinstance(out_channels, int), f"out_channels must be int, got {type(out_channels)}"
        assert in_channels > 0, f"in_channels must be positive, got {in_channels}"
        assert out_channels > 0, f"out_channels must be positive, got {out_channels}"

        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels, batch_norm=batch_norm, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.maxpool(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    """Upsampling block for U-Net decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        batch_norm: bool = True,
        dropout: float = 0.0,
    ):
        """
        Initialize upsampling block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            batch_norm: Whether to use batch normalization
            dropout: Dropout probability
        """
        super().__init__()

        # Ensure channel values are integers
        assert isinstance(in_channels, int), f"in_channels must be int, got {type(in_channels)}"
        assert isinstance(out_channels, int), f"out_channels must be int, got {type(out_channels)}"
        assert in_channels > 0, f"in_channels must be positive, got {in_channels}"
        assert out_channels > 0, f"out_channels must be positive, got {out_channels}"
        
        # Ensure in_channels // 2 is valid
        mid_ch = in_channels // 2
        assert isinstance(mid_ch, int), f"mid_ch must be int, got {type(mid_ch)}"
        assert mid_ch > 0, f"mid_ch must be positive, got {mid_ch}"

        self.up = nn.ConvTranspose2d(in_channels, mid_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, batch_norm=batch_norm, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connection.

        Args:
            x: Input tensor
            skip: Skip connection from encoder

        Returns:
            Output tensor
        """
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x

