import hydra
import numpy as np
import torch
import utils.log as log
from models.codec.sac.third_party.hf_whisper.modeling_whisper import (
    WhisperVQEncoder,
)
from transformers import WhisperFeatureExtractor
from utils.audio import audio_highpass_filter, load_audio


def load_whisper_tokenizer(tokenizer_path: str, device: torch.device):
    log.info(f"[SAC] Loading Whisper-based tokenizer from: {tokenizer_path}")
    model = WhisperVQEncoder.from_pretrained(tokenizer_path).eval().to(device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_path)
    return model, feature_extractor


def inference_factory(cfg, args_dict):
    model = hydra.utils.instantiate(cfg["model"]["generator"])
    model = model.load_from_checkpoint(
        args_dict["config"], args_dict["ckpt"], **args_dict
    )
    if hasattr(model, "remove_weight_norm"):
        model.remove_weight_norm()
    model.eval().to(args_dict["device"])
    return model


def process_audio(wav_path: str, cfg, hop_length: int):
    """Return wav_in (np.float32), already hp-filtered & padded to hop multiple."""
    wav = load_audio(
        wav_path,
        sampling_rate=cfg["sample_rate"],
        volume_normalize=cfg.get("volume_normalize", False),
    )
    if wav is None:
        raise ValueError(f"Failed to load audio from {wav_path}")

    hp_cut = float(cfg.get("highpass_cutoff_freq", 0.0))
    if hp_cut != 0:
        wav = audio_highpass_filter(wav, cfg["sample_rate"], hp_cut)

    remainder = len(wav) % hop_length
    if remainder != 0:
        pad_len = hop_length - remainder
        wav = np.pad(wav, (0, pad_len), mode="constant", constant_values=0.0)

    return wav.astype(np.float32)