import torch
import torch.nn as nn

from models.base.fsq.residual_fsq import ResidualFSQ
from models.base.modules.ecapa_tdnn import ECAPA_TDNN_GLOB_c512
from models.base.modules.perceiver_encoder import PerceiverResampler

"""
x-vector + d-vector
"""

class SpeakerEncoder(nn.Module):
    """

    Args:
        input_dim (int): acoustic feature dim, e.g., 100 for mel
        n_head (int): head number
        query_dim (int): dim of context feature. e.g, dim of codec feature
        out_dim (int): dim of output.

    Return:
        speaker_embs: (B, T2, out_dim)
    """

    def __init__(
        self,
        input_dim: int = 100,
        out_dim: int = 512,
        vq_dim: int = 128,
        num_latents: int = 32,
        levels = [4, 4, 4, 4, 4, 4],
        num_quantizers = 1,
    ):
        super(SpeakerEncoder, self).__init__()

        self.speaker_encoder = ECAPA_TDNN_GLOB_c512(feat_dim=input_dim, embed_dim=out_dim)
        self.perceiver_sampler = PerceiverResampler(dim=vq_dim, dim_context=512*3, num_latents=num_latents)
        self.quantizer = ResidualFSQ(
                            levels = levels,
                            num_quantizers = num_quantizers,
                            dim = vq_dim,
                            is_channel_first = True,
                            quantize_dropout = False
                        )

        self.project = nn.Linear(vq_dim * num_latents, out_dim)

    def get_codes_from_indices(self, indices):
        zq = self.quantizer.get_codes_from_indices(indices.transpose(1,2))
        return zq.transpose(1,2)

    def get_indices(self, mels):
        mels = mels.transpose(1,2)
        x = self.perceiver_sampler(mels).transpose(1,2)
        zq, indices = self.quantizer(x)
        return indices


    def forward(self, mels):
        """FincoSpeakerEncoder forward.

        Args:
            mels: (B, D_mel, T1)
            query: (B, D_query, T2). e.g., codec features.

        Return:
            speaker_embs: (B, T2, out_dim)
        """
        # mels = mels.transpose(1,2)

        x_vector, features = self.speaker_encoder(mels, True) # global feature，temporal feature

        x = self.perceiver_sampler(features.transpose(1,2)).transpose(1,2)

        zq, indices = self.quantizer(x)

        x = zq.reshape(zq.shape[0], -1)
        
        d_vector = self.project(x)

        return x_vector, d_vector


if __name__ == "__main__":
    model = SpeakerEncoder(
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