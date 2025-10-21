import torch
import torch.nn as nn
from models.codec.sac.modules.utils.ERes2Net import ERes2Net
from models.codec.sac.modules.utils.fbank import FBank

ERes2Net_VOX = {
    "args": {"feat_dim": 80, "embedding_size": 192},
}

class SpeakerEmbedder(nn.Module):
    """Load pretrained speaker model and extract embeddings from waveform tensor."""

    def __init__(self,
                 pretrained_dir: str,
                 freeze: bool = True
                 ):
        super().__init__()
        self.model = ERes2Net(**ERes2Net_VOX["args"])
        state = torch.load(f"{pretrained_dir}/pretrained_eres2net.ckpt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)

        self.feat_extractor = FBank(80, sample_rate=16000, mean_nor=True)
        self.embedding_dim = ERes2Net_VOX["args"]["embedding_size"]

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wav (torch.Tensor): [B, T] waveform, already 16kHz mono
        Returns:
            embeddings: [B, embedding_dim]
        """
        if wav.ndim == 1:  # single example
            wav = wav.unsqueeze(0)
        feats = [self.feat_extractor(w) for w in wav]       # list of [T_f, 80]
        feats = torch.stack(feats)                          # [B, T_f, 80]
        emb = self.model(feats).detach()                    # [B, D_emb]
        return emb