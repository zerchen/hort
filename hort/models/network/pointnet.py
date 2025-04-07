import torch
import torch.nn as nn


class PointNetEncoder(nn.Module):
    """Encoder for Pointcloud
    """
    def __init__(self, in_channels: int=3, output_channels: int=768):
        super().__init__()

        block_channel = [64, 128, 256, 512]
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
            nn.LayerNorm(block_channel[3]),
            nn.ReLU(),
        )

        self.final_projection = nn.Sequential(
            nn.Linear(block_channel[-1], output_channels),
            nn.LayerNorm(output_channels)
        )
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x