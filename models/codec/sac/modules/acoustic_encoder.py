import math

import torch
import torch.nn as nn
from models.base.modules.dac_utils.layers import Snake1d, WNConv1d


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y
    

class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)
    

class AcousticEncoder(nn.Module):
    """
    Acoustic Encoder adjusted from descript-audio-codec (https://github.com/descriptinc/descript-audio-codec)
    """
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: list = [2, 4, 8, 8],
        latent_dim: int = 64,
    ):
        super().__init__()
        # Create first convolution
        self.block = [WNConv1d(1, encoder_dim, kernel_size=7, padding=3)]

        # Create EncoderBlocks that double channels as they downsample by `stride`
        for stride in encoder_rates:
            encoder_dim *= 2
            self.block += [EncoderBlock(encoder_dim, stride=stride)]

        # Create last convolution
        self.block += [
            Snake1d(encoder_dim),
            WNConv1d(encoder_dim, latent_dim, kernel_size=3, padding=1),
        ]

        # Wrap black into nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = encoder_dim

    def forward(self, x):
        return self.block(x)


if __name__ == "__main__":
    model = AcousticEncoder(
        input_dim = 100,
        vq_dim=128,
        out_dim = 512,
        num_latents = 32,
        levels = [4, 4, 4, 4, 4, 4],
        num_quantizers = 1,
    )
    mel = torch.randn(8, 200, 100)
    x_vector, d_vector = model(mel)
    print('x-vector shape', x_vector.shape)
    print('d-vector shape', d_vector.shape)
    num_params = sum(param.numel() for param in model.parameters())
    print("{} M".format(num_params / 1e6))