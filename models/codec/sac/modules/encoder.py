import torch
import torch.nn as nn
from models.codec.sac.modules.sampler import SamplingBlock
from models.codec.sac.modules.vocos import VocosBackbone


class Encoder_with_downsample(nn.Module):
    """ Encoder module with convnext
    """
    def __init__(
        self,
        input_channels: int,
        vocos_dim: int,
        vocos_intermediate_dim: int,
        vocos_num_layers: int,
        out_channels: int,     
        sample_ratios: tuple = [1, 1],
    ):
        super().__init__()

        self.encoder = VocosBackbone(
                input_channels=input_channels,
                dim=vocos_dim,
                intermediate_dim=vocos_intermediate_dim,
                num_layers=vocos_num_layers,
                condition_dim=None,
            )
        
        modules = [
                    nn.Sequential(
                        SamplingBlock(
                            dim=vocos_dim,
                            groups=vocos_dim,
                            downsample_scale=ratio,
                        ),
                        VocosBackbone(
                            input_channels=vocos_dim,
                            dim=vocos_dim,
                            intermediate_dim=vocos_intermediate_dim,
                            num_layers=2,
                            condition_dim=None,
                        )
                    ) for ratio in sample_ratios
                ]

        self.downsample = nn.Sequential(*modules)

        self.project = nn.Linear(vocos_dim, out_channels)

    def forward(self, x: torch.Tensor, *args):
        """encoder forward.
        
        Args:
            x (torch.Tensor): (batch_size, input_channels, length)
        
        Returns:
            x (torch.Tensor): (batch_size, encode_channels, length)
        """
        x = self.encoder(x)
        x = self.downsample(x)
        x = self.project(x)
        return x.transpose(1,2)


# test
if __name__ == '__main__':
    from utils.commons import test_successful
    test_input = torch.randn(8, 1024, 50)  # Batch size = 8, 1024 channels, length = 50
    encoder = Encoder_with_downsample(input_channels=1024, vocos_dim=384, vocos_intermediate_dim=2048, vocos_num_layers=12, out_channels=256, sample_ratios= [2,2])
    
    output = encoder(test_input)
    print(output.shape)   # torch.Size([8, 256, 12])
    test_successful()