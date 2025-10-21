from pathlib import Path
from typing import Dict, Optional

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.log as log
from audiotools import AudioSignal
from models.base.loss.ssim import SSIM
from omegaconf import DictConfig
from utils.checkpoint import strip_prefix
from utils.file import load_config


class SAC(nn.Module):
    def __init__(
        self,
        loss_config: DictConfig = None,
        semantic_encoder: nn.Module = None,
        semantic_adapter: nn.Module = None,
        acoustic_encoder: nn.Module = None,
        acoustic_quantizer: nn.Module = None,
        speaker_predictor: nn.Module = None,
        prenet: nn.Module = None,
        semantic_decoder: nn.Module = None,
        acoustic_decoder: nn.Module = None,
        speaker_encoder: nn.Module = None,
        **kwargs,
    ):
        super().__init__()
        self.loss_config = loss_config

        self.semantic_encoder = semantic_encoder
        self.semantic_adapter = semantic_adapter
        self.acoustic_encoder = acoustic_encoder
        self.acoustic_quantizer = acoustic_quantizer
        self.prenet = prenet
        self.speaker_encoder = speaker_encoder
        self.speaker_predictor = speaker_predictor
        self.semantic_decoder = semantic_decoder
        self.acoustic_decoder = acoustic_decoder

        if loss_config is not None:
            self.init_loss_function(loss_config)

        if self.semantic_encoder.from_pretrained is not None:
            self.semantic_encoder = self.init_semantic_encoder(
                cfg=self.semantic_encoder.from_pretrained
            )

    def init_semantic_encoder(self, cfg: dict = None):
        from models.codec.sac.modules.semantic_encoder import WhisperVQEncoder
        model_path = cfg.get("local_ckpt") or cfg.get("hf_repo")
        encoder = WhisperVQEncoder.from_pretrained(model_path)
        # if int(os.environ.get("RANK", 0)) == 0:
        #     log.info(f"Semantic encoder loaded from {model_path}")

        load_codebook_only = cfg.get("load_codebook_only", False)
        if load_codebook_only:
            encoder._prune_to_codebook_only()

        if cfg.get("freeze", False):
            encoder._freeze_parameters()
        return encoder

    @classmethod
    def load_from_checkpoint(
        cls, 
        config_path: Path, 
        ckpt_path: Path, 
        device: torch.device, **kwargs
    ):
        """
        Load pre-trained model

        Args:
            config_path (Path): path to the model model configuration.
            ckpt_path (Path): path of model checkpoint.
            device (torch.device): The device to load the model onto.

        Kwargs:
            ema_load (bool): If True and EMA weights are present, prefer 'ema_generator'.
            
        Returns:
            model (nn.Module): The loaded model instance.
        """
        cfg = load_config(config_path)
        if "config" in cfg.keys():
            cfg = cfg["config"]

        cls.device = device
        gen_cfg = cfg["model"]["generator"]
        semantic_encoder = hydra.utils.instantiate(gen_cfg["semantic_encoder"])
        semantic_adapter = hydra.utils.instantiate(gen_cfg["semantic_adapter"]) if gen_cfg.get("semantic_adapter") else False
        acoustic_encoder = hydra.utils.instantiate(gen_cfg["acoustic_encoder"]) if gen_cfg.get("acoustic_encoder") else False
        acoustic_quantizer = hydra.utils.instantiate(gen_cfg["acoustic_quantizer"]) if gen_cfg.get("acoustic_quantizer") else False
        speaker_predictor = hydra.utils.instantiate(gen_cfg["speaker_predictor"]) if gen_cfg.get("speaker_predictor") else False
        speaker_encoder = hydra.utils.instantiate(gen_cfg["speaker_encoder"]) if gen_cfg.get("speaker_encoder") else False
        prenet = hydra.utils.instantiate(gen_cfg["prenet"])
        semantic_decoder = hydra.utils.instantiate(gen_cfg["semantic_decoder"])
        acoustic_decoder = hydra.utils.instantiate(gen_cfg["acoustic_decoder"])

        model = cls(
            loss_config=None,
            semantic_encoder=semantic_encoder,
            semantic_adapter=semantic_adapter,
            acoustic_encoder=acoustic_encoder,
            acoustic_quantizer=acoustic_quantizer,
            speaker_predictor=speaker_predictor,
            prenet=prenet,
            semantic_decoder=semantic_decoder,
            acoustic_decoder=acoustic_decoder,
            speaker_encoder=speaker_encoder,
        )

        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        want_ema = bool(cfg.get("ema_update", False)) and bool(kwargs.get("ema_load", False))
        key = "ema_generator" if (want_ema and "ema_generator" in state_dict) else "generator"
        sd = state_dict[key]

        if key == "ema_generator":
            sd = strip_prefix(sd, "ema_model.")
            log.info(f"[SAC] Loading EMA weights from: {ckpt_path}")
        else:
            log.info(f"[SAC] Loading weights from: {ckpt_path}")

        missing_keys, unexpected_keys = model.load_state_dict(
            sd, strict=False
        )

        for key in missing_keys:
            log.info("[SAC] Missing tensor: {}".format(key))
        for key in unexpected_keys:
            log.info("[SAC] Unexpected tensor: {}".format(key))

        model.to(device).eval()
        return model

    def forward(self, inputs: dict,):
        """
        Forward pass of Semantic-acoustic dual-stream codec (SAC).
        """
        semantic_tokens = inputs['semantic_tokens']
        wav = inputs['wav']
        ssl_feat = inputs.get("ssl_feat", None)
        sim_feat = inputs.get("sim_feat", None)
        acoustic_only = False
        semantic_only = False

        # note: not yet support online feature extraction
        feat = semantic_tokens

        if not acoustic_only:
            sem_emb = self.semantic_encoder.embed_ids(feat)  # B x T_s x D_s

        if self.semantic_adapter:
            sem_emb = self.semantic_adapter(sem_emb.transpose(1, 2)).transpose(1, 2)

        if self.acoustic_encoder:
            acoustic_encoder_out = self.acoustic_encoder(wav)   # B x 1 x T_a -> B x D_a x T_a

            if self.acoustic_quantizer:
                aq_outputs = self.acoustic_quantizer(acoustic_encoder_out)
                if len(aq_outputs) == 7:
                    zq_a, a_indices, a_commit_loss, a_codebook_loss, _, a_perplexity, a_cluster_size = aq_outputs
                    vq_loss = (a_commit_loss + a_codebook_loss).mean()
                elif len(aq_outputs) == 5:
                    zq_a, a_indices, vq_loss, a_perplexity, a_cluster_size = aq_outputs
                    if isinstance(vq_loss, tuple):
                        vq_loss = vq_loss[0]
                elif len(aq_outputs) == 2:
                    zq_a, a_indices = aq_outputs
                    vq_loss, a_perplexity, a_cluster_size = 0.0, 0.0, 0
                else:
                    raise RuntimeError("Unexpected acoustic_quantizer output format.")
            else:
                zq_a = acoustic_encoder_out
                vq_loss, a_perplexity, a_cluster_size = 0.0, 0.0, 0
            
            acu_emb = zq_a.transpose(1, 2)  # B x T_a x D_a

        else:
            acoustic_encoder_out = None
            zq_a = None
            vq_loss, a_perplexity, a_cluster_size = 0, 0, 0
            combined_emb = sem_emb  # B x T_s x D_s

        if self.acoustic_encoder and not semantic_only:
            if not acoustic_only:
                combined_emb = torch.cat([sem_emb, acu_emb], dim=2)  # B x T x (D_s + D_a)
            else:
                combined_emb = acu_emb  # B x T_a x D_a
        else:
            combined_emb = sem_emb

        prenet_in = combined_emb.transpose(1, 2)
        x = self.prenet(prenet_in)
        pred_feat = self.semantic_decoder(x) if self.semantic_decoder else None
        y = self.acoustic_decoder(x)

        if self.speaker_predictor:
            proj_sim_feat = self.speaker_predictor(x)

        if self.speaker_encoder:
            with torch.no_grad():
                sim_feat = self.speaker_encoder(wav)

        return {
            'recons': y,
            'pred': pred_feat,
            'zq': zq_a, 
            'z': acoustic_encoder_out,
            'vqloss': vq_loss,
            'perplexity': a_perplexity,
            'cluster_size': a_cluster_size,
            'ssl_feat': ssl_feat,
            'step':inputs.get('step', 0),
            'quantized_tokens': a_indices if self.acoustic_quantizer else None,
            'pred_sim_feat': proj_sim_feat if self.speaker_predictor else None,
            'sim_feat': sim_feat,
        }

    def generative_loss(
        self,
        inputs: dict,
    ):
        """
        Compute the generator-side composite loss.

        Args:
            inputs (dict): A dictionary that should contains the following elemetns:
                - 'recons' (torch.Tensor): Synthetic audios with shape [B, T]
                - 'audios' (torch.Tensor): Ground-truth audios with shape [B, T]

        Returns:
            loss_dict (dict):
                - 'loss'
        """
        loss_dict = dict()

        # vq loss
        loss_dict['vq_loss'] = inputs["vqloss"]
        loss_dict['perplexity'] = inputs['perplexity']
        loss_dict['cluster_size'] = inputs['cluster_size']
        # reconstruction loss of ssl feature
        loss_dict["mse_loss"] = self.compute_mse_loss(inputs['pred'], inputs['ssl_feat'])
        
        # reconstruction loss of speaker feature
        if 'pred_sim_feat' in inputs and inputs['pred_sim_feat'] is not None and 'sim_feat' in inputs and inputs['sim_feat'] is not None:
            loss_dict["sim_mse_loss"] = self.compute_mse_loss(inputs['pred_sim_feat'], inputs['sim_feat'])
            
        # if inputs['step'] < self.d_vector_train_start:
        #       loss_dict["speaker_loss"] = self.compute_mse_loss(inputs['x_vector'].detach(), inputs['d_vector'])
        # loss_dict["ssim_loss"] = 1 - self.compute_ssim(inputs['pred'].unsqueeze(1), inputs['ssl_feat'].unsqueeze(1))

        # mel reconstruction loss
        signal = AudioSignal(inputs["audios"], self.loss_config["sample_rate"])
        recons = AudioSignal(inputs["recons"], signal.sample_rate)
        loss_dict["mel_loss"] = self.compute_mel_loss(recons, signal)
       
        loss = sum(
            [
                v * loss_dict[k]
                for k, v in self.loss_config["loss_weights"].items()
                if k in loss_dict
            ]
        )
        loss_dict = {
            k: v.item() for k, v in loss_dict.items() if not isinstance(v, int)
        }
        loss_dict["loss"] = loss
        loss_dict["gen_loss"] = loss.item()

        return loss_dict

    def init_loss_function(self, loss_config):
        from models.codec.sac.blocks import loss as losses

        # In the inference process, initialization of this function can be skipped.
        if loss_config is None:
            return
        # Init loss function for training process
        # self.compute_stft_loss = losses.MultiScaleSTFTLoss()
        self.compute_ssim = SSIM()
        self.compute_mse_loss = nn.MSELoss()
        self.compute_waveform_loss = losses.L1Loss()
        self.compute_mel_loss = losses.MelSpectrogramLoss(**loss_config["mel_loss"])
    
    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                # print(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    @torch.inference_mode()
    def encode(
        self,
        wav: torch.Tensor,
        semantic_tokens: Optional[torch.LongTensor] = None,
        return_zq: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode waveform to discrete tokens (semantic + acoustic).
        """
        device = next(self.parameters()).device

        # normalize wav shape -> [B, 1, L]
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)
        wav = wav.to(device)

        if semantic_tokens is None:
            outputs = self.semantic_encoder(wav)
            sem_tok = outputs.quantized_token_ids
        else:
            sem_tok = semantic_tokens.to(device)

        assert self.acoustic_encoder is not None and self.acoustic_quantizer is not None, \
            "encode() needs acoustic_encoder & acoustic_quantizer"
        z_e = self.acoustic_encoder(wav)                          # [B, D_a, T_a]
        z_q, a_idx, *_ = self.acoustic_quantizer(z_e)

        a_indices = a_idx.to(torch.long)

        ret = {
            "semantic_tokens": sem_tok,
            "acoustic_tokens": a_indices,
        }
        if return_zq:
            ret["zq"] = z_q
        return ret

    @torch.inference_mode()
    def decode(
        self,
        semantic_tokens: torch.LongTensor,
        acoustic_tokens: torch.LongTensor,
        acoustic_masked: bool = False,
        semantic_masked: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode (semantic_ids, acoustic_ids) back to waveform.
        """
        aggregate_strategy = None
        global_acu_emb = None

        device = next(self.parameters()).device
        sem_tok = semantic_tokens.to(device)
        a_idx   = acoustic_tokens.to(device)

        sem_emb = self.semantic_encoder.embed_ids(sem_tok)
        if self.semantic_adapter:
            sem_emb = sem_emb.transpose(1, 2)
            sem_emb = self.semantic_adapter(sem_emb)
            sem_emb = sem_emb.transpose(1, 2)

        assert self.acoustic_quantizer is not None, "decode() needs acoustic_quantizer"
        zq_a = self.acoustic_quantizer.vq2emb(a_idx, out_proj=True)  # [B, D_a, T_a]
        acu_emb = zq_a.transpose(1, 2)                               # [B, T_a, D_a]

        if semantic_masked:
            sem_emb = torch.zeros_like(sem_emb)
        if acoustic_masked:
            acu_emb = torch.zeros_like(acu_emb)

        if (global_acu_emb is not None) and (aggregate_strategy == "concat"):
            comb = torch.cat([sem_emb, acu_emb, global_acu_emb], dim=2)    # [B,T, D_s + D_a + D_a]
        else:
            comb = torch.cat([sem_emb, acu_emb], dim=2)                    # [B,T, D_s + D_a]

        x = self.prenet(comb.transpose(1, 2))   # [B, D_in, T] -> prenet -> [B, D_mid, T]
        y = self.acoustic_decoder(x)            # [B, 1, L'] -> waveform

        return {"recons": y}

    @torch.inference_mode()
    def get_embedding(
        self,
        wav: torch.Tensor,
        semantic_tokens: Optional[torch.LongTensor] = None,
        acoustic_only: bool = False,
        semantic_only: bool = False,
    ) -> torch.Tensor:
        """
        Extract embedding (semantic-only, acoustic-only, or combined).

        Args:
            wav (torch.Tensor): Input waveform, shape [B, L] or [B, 1, L].
            semantic_tokens (torch.LongTensor, optional): Pre-computed semantic tokens [B, T_s].
            acoustic_only (bool): If True, return only acoustic embedding.
            semantic_only (bool): If True, return only semantic embedding.

        Returns:
            torch.Tensor: Embedding tensor, shape [B, T, D].
        """
        device = next(self.parameters()).device

        # normalize wav shape -> [B, 1, L]
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)
        wav = wav.to(device)

        # semantic branch
        if semantic_tokens is None:
            outputs = self.semantic_encoder(wav)
            sem_tok = outputs.quantized_token_ids
        else:
            sem_tok = semantic_tokens.to(device)

        sem_emb = self.semantic_encoder.embed_ids(sem_tok)  # [B, T_s, D_s]

        if semantic_only:
            return sem_emb

        if self.semantic_adapter:
            sem_emb = sem_emb.transpose(1, 2)
            sem_emb = self.semantic_adapter(sem_emb)
            sem_emb = sem_emb.transpose(1, 2)

        # acoustic branch
        acu_emb, zq_a = None, None
        if self.acoustic_encoder is not None:
            z_e = self.acoustic_encoder(wav)  # [B, D_a, T_a]
            if self.acoustic_quantizer is not None:
                zq_a, *_ = self.acoustic_quantizer(z_e)
            else:
                zq_a = z_e
            acu_emb = zq_a.transpose(1, 2)  # [B, T_a, D_a]

        if acoustic_only:
            emb = acu_emb
        else:
            if sem_emb is not None and acu_emb is not None:
                emb = torch.cat([sem_emb, acu_emb], dim=2)  # [B, T, D_s+D_a]
            else:
                raise RuntimeError("Either semantic or acoustic branch is None.")
        
        return emb


class WavDiscriminator(nn.Module):
    """
    Patch-discriminator on waveform.
    """
    def __init__(
        self, loss_config: dict = None, discriminator: nn.Module = None, **kwargs
    ):
        super().__init__()
        self.loss_config = loss_config
        self.discriminator = discriminator

    def forward(self, fake: AudioSignal, real: AudioSignal):
        d_fake = self.discriminator(fake.audio_data)
        d_real = self.discriminator(real.audio_data)
        return d_fake, d_real

    def discriminative_loss(self, inputs: dict):
        """
        Compute D loss:
            E[(D(fake))^2] + E[(1 - D(real))^2]

        Args:
            inputs (dict): A dictionary that should contains the following elemetns:
                - 'recons' (torch.Tensor): Synthetic audios with shape [B, T]
                - 'audios' (torch.Tensor): Ground-truth audios with shape [B, T]

        Returns:
            loss_dict (dict):
                - 'd_loss'
        """
        loss_dict = dict()
        signal = AudioSignal(inputs["audios"], self.loss_config["sample_rate"])
        recons = AudioSignal(inputs["recons"], signal.sample_rate)
        d_loss = self.compute_discriminator_loss(recons, signal)
        loss_dict["loss"] = d_loss
        loss_dict["d_loss"] = d_loss.item()

        return loss_dict

    def adversarial_loss(self, inputs: dict):
        """
        Compute G-side GAN losses:
            - adv_gen_loss: sum over scales of (1 - D(fake_last))^2
            - adv_feat_loss: feature matching L1 between D(fake) and D(real)

        Args:
            inputs (dict): A dictionary that should contains the following elemetns:
                - 'recons' (torch.Tensor): Synthetic audios with shape [B, T]
                - 'audios' (torch.Tensor): Ground-truth audios with shape [B, T]

        Returns:
            loss_dict (dict):
                - 'loss'
        """
        loss_dict = dict()
        signal = AudioSignal(inputs["audios"], self.loss_config["sample_rate"])
        recons = AudioSignal(inputs["recons"], signal.sample_rate)

        d_fake, d_real = self.forward(recons, signal)

        loss_g = 0
        for x_fake in d_fake:
            loss_g += torch.mean((1 - x_fake[-1]) ** 2)

        loss_feature = 0

        for i in range(len(d_fake)):
            for j in range(len(d_fake[i]) - 1):
                loss_feature += F.l1_loss(d_fake[i][j], d_real[i][j].detach())

        loss_dict["adv_gen_loss"] = loss_g
        loss_dict["adv_feat_loss"] = loss_feature

        loss = sum(
            [
                v * loss_dict[k]
                for k, v in self.loss_config["loss_weights"].items()
                if k in loss_dict
            ]
        )

        loss_dict = {
            k: v.item() for k, v in loss_dict.items() if not isinstance(v, int)
        }
        loss_dict["loss"] = loss

        return loss_dict

    def compute_discriminator_loss(self, fake, real):
        d_fake, d_real = self.forward(fake.clone().detach(), real)
        loss_d = 0
        for x_fake, x_real in zip(d_fake, d_real):
            loss_d += torch.mean(x_fake[-1] ** 2)
            loss_d += torch.mean((1 - x_real[-1]) ** 2)

        return loss_d


# test
if __name__ == "__main__":
    import torchaudio.transforms as TT
    from models.base.modules.fsq_encoder import SpeakerEncoder
    from models.codec.base.quantizer.quantizer_gumbel import VectorQuantization
    from models.codec.sac.modules.decoder import Decoder_with_upsample
    from models.codec.sac.modules.encoder import Encoder_with_downsample
    from models.codec.sac.modules.vocoder.wave_discriminator import Discriminator
    from models.codec.sac.modules.vocoder.wave_generator import Decoder

    loss_config = {
        "sample_rate": 16000,
        "mel_loss": {
            "window_lengths": [2048, 512],
            "clamp_eps": 1e-5,
            "mag_weight": 1.0,
            "log_weight": 1.0,
            "pow": 2.0,
            "weight": 1.0,
            "match_stride": False,
            "window_type": None,
        },
        "loss_weights": {
            "mse_loss": 1,
            "ssim_loss": 1,
            "vq_loss": 1,
            "mel_loss": 100.0,
            "adv_gen_loss": 1.0,
            "adv_feat_loss": 2.0,
        },
    }

    config = {
        "sample_rate": 16000,
        "n_fft": 1024,
        "win_length": 640,
        "hop_length": 320,
        "mel_fmin": 10,
        "mel_fmax": None,
        "num_mels": 128,
    }

    mel_transformer = TT.MelSpectrogram(
            config["sample_rate"],
            config["n_fft"],
            config["win_length"],
            config["hop_length"],
            config["mel_fmin"],
            config["mel_fmax"],
            n_mels=config["num_mels"],
            power=1,
            norm="slaney",
            mel_scale="slaney",
        )

    encoder = Encoder_with_downsample(
        input_channels = 1024,
        vocos_dim = 384,
        vocos_intermediate_dim = 2048,
        vocos_num_layers = 12,
        out_channels = 1024,
        sample_ratios=[2,2]  
    )
    
    prenet = Decoder_with_upsample(
        input_channels = 1024,
        vocos_dim = 384,
        vocos_intermediate_dim = 2048,
        vocos_num_layers = 12,    
        out_channels = 1024,
        condition_dim = 1024,
        sample_ratios=[2,2]
    )
    
    semantic_decoder = Decoder_with_upsample(
        input_channels = 1024,
        vocos_dim = 384,
        vocos_intermediate_dim = 2048,
        vocos_num_layers = 6,    
        out_channels = 1024
    )

    quantizer = VectorQuantization(
        dim = 1024,
        codebook_dim = 8,
        codebook_size = 4096,
        kmeans_init = False,
        kmeans_iters = 0,
        commitment_weight = 1.0,
        orthogonal_reg_weight = 0.,
        straight_through = True,
        reinmax = True,
        threshold_ema_dead_code = 0.2,
        stochastic_sample_codes = False
    )

    generator = Decoder(
        1024,
        channels=1536,
        rates=[8, 5, 4, 2],
        kernel_sizes=[16,11,8,4]
    )
    
    acoustic_encoder = SpeakerEncoder(
        input_dim = 128,
        out_dim = 1024,
        num_latents = 128,
        levels = [4, 4, 4, 4, 4, 4],
        num_quantizers = 1
    )

    model_g = SAC(
        loss_config=loss_config,
        encoder=encoder,
        quantizer=quantizer,
        prenet=prenet,
        semantic_decoder=semantic_decoder,
        decoder=generator,
        acoustic_encoder=acoustic_encoder
    )

    model_d = WavDiscriminator(
        loss_config=loss_config, discriminator=Discriminator(sample_rate=16000)
    )

    duration = 0.96
    x = torch.randn(2, 1, int(duration * 16000))
    feat = torch.randn(2, 1024, int(duration * 50))
    mel = mel_transformer(x.squeeze(1))
    inputs = {'feat': feat,
              'mel': mel,
              'step': 10000}
    y = model_g(inputs)

    for k, value in y.items():
        if "loss" not in k and isinstance(value, torch.Tensor):
            print(f"{k} shape:", value.shape)
    print("============gen loss ================")
    y.update({"audios": x})

    loss_dict = model_g.generative_loss(y)
    loss_dict_adv = model_d.adversarial_loss(y)
    loss_dict["loss"] += loss_dict_adv["loss"]
    loss_dict_adv.pop("loss")
    loss_dict.update(loss_dict_adv)
    print(loss_dict)
    print("=======discriminative loss============")
    loss_dict = model_d.discriminative_loss(y)
    print(loss_dict)