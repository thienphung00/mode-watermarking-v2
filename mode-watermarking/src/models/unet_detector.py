"""
UNet-based binary classifier for watermark detection.
Processes latent representations [B, 4, 64, 64] → logits [B, 1]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block with optional batch norm and dropout."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        
        if use_batch_norm:
            # Insert batch norm after first conv
            layers.insert(1, nn.BatchNorm2d(out_channels))
            # Insert batch norm after second conv
            layers.insert(-1, nn.BatchNorm2d(out_channels))
        
        if dropout > 0.0:
            layers.append(nn.Dropout2d(dropout))
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetDetector(nn.Module):
    """
    UNet-based binary classifier for watermark detection.
    Processes latent representations [B, 4, 64, 64] → logits [B, 1]
    
    Architecture:
    - Encoder: Downsample latent features (64→32→16→8)
    - Bottleneck: Feature aggregation at 8x8
    - Decoder: Upsample with skip connections (8→16→32→64)
    - Classifier head: Global average pool → FC → binary logit
    """
    
    def __init__(
        self,
        input_channels: int = 4,
        base_channels: int = 64,
        num_classes: int = 1,
        depth: int = 4,
        use_batch_norm: bool = True,
        dropout: float = 0.0
    ):
        """
        Initialize UNet detector.
        
        Args:
            input_channels: Input latent channels (default: 4)
            base_channels: Base channel count (default: 64)
            num_classes: Output classes (default: 1 for binary)
            depth: Number of downsampling levels (default: 4)
            use_batch_norm: Use batch normalization (default: True)
            dropout: Dropout rate (default: 0.0)
        """
        super().__init__()
        self.input_channels = input_channels
        self.base_channels = base_channels
        self.num_classes = num_classes
        self.depth = depth
        
        # Encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        in_ch = input_channels
        
        for i in range(depth):
            out_ch = base_channels * (2 ** i)
            self.encoder_blocks.append(
                DoubleConv(in_ch, out_ch, use_batch_norm, dropout)
            )
            in_ch = out_ch
        
        # Bottleneck
        bottleneck_ch = base_channels * (2 ** depth)
        self.bottleneck = DoubleConv(in_ch, bottleneck_ch, use_batch_norm, dropout)
        
        # Decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        
        for i in range(depth - 1, -1, -1):
            in_ch = bottleneck_ch if i == depth - 1 else base_channels * (2 ** (i + 1))
            out_ch = base_channels * (2 ** i)
            
            # Upsampling
            self.up_convs.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            )
            
            # Decoder block (with skip connection)
            self.decoder_blocks.append(
                DoubleConv(in_ch + out_ch, out_ch, use_batch_norm, dropout)
            )
        
        # Classifier head
        # Global average pooling + fully connected layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(base_channels, base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(base_channels // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Latent tensor [B, 4, 64, 64]
        
        Returns:
            Logits tensor [B, 1]
        """
        # Encoder path with skip connections
        skip_connections = []
        
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            skip_connections.append(x)
            # Downsample (except last encoder block)
            if x.shape[-1] > 8:  # Only downsample if > 8x8
                x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with skip connections
        for i, (up_conv, decoder_block) in enumerate(zip(self.up_convs, self.decoder_blocks)):
            # Upsample
            x = up_conv(x)
            
            # Skip connection
            skip = skip_connections[-(i + 1)]
            # Handle size mismatch (can happen with odd dimensions)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([skip, x], dim=1)
            x = decoder_block(x)
        
        # Classifier head
        x = self.global_pool(x)  # [B, base_channels, 1, 1]
        x = x.view(x.size(0), -1)  # [B, base_channels]
        x = self.classifier(x)  # [B, num_classes]
        
        return x
