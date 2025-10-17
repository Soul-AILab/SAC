import glob
import io
import math
import os
import tarfile

import numpy as np
import safetensors
import torch
import torchaudio
from models.codec.sac.third_party.hf_whisper.configuration_whisper import (
    WhisperVQConfig,
)
from models.codec.sac.third_party.hf_whisper.modeling_whisper import (
    WhisperVQEncoder,
)
from transformers import WhisperFeatureExtractor, WhisperTokenizerFast
from utils.audio import load_audio


def load_quantize_encoder(model_path):
    config = WhisperVQConfig.from_pretrained(model_path)
    config.quantize_encoder_only = True
    model = WhisperVQEncoder(config)
    state_dict = {}
    for path in glob.glob(os.path.join(model_path, "model*.safetensors")):
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("model.encoder."):
                    new_key = key[len("model.encoder."):]
                    if new_key.startswith("layer_norm"):
                        continue
                    if new_key.startswith("layers"):
                        layer_id = int(new_key.split(".")[1])
                        if layer_id >= config.quantize_position:
                            continue
                    state_dict[new_key] = f.get_tensor(key)
    model.load_state_dict(state_dict)
    model.eval()
    model.cuda()
    return model


_resample_buffer: dict[int, torchaudio.transforms.Resample] = {}


def extract_speech_token(model: WhisperVQEncoder, feature_extractor: WhisperFeatureExtractor, utts, sample_rate=None):
    with torch.no_grad():
        audios, indices = [], []
        for idx, utt in enumerate(utts):
            if isinstance(utt, tuple):
                audio, sample_rate = utt
            elif isinstance(utt, np.ndarray):
                sample_rate = sample_rate or 16000
                audio = torch.from_numpy(utt).unsqueeze(0).float()
            else:
                sample_rate = sample_rate or 16000
                audio = load_audio(utt, sampling_rate=sample_rate, volume_normalize=True)
                audio = torch.from_numpy(audio).unsqueeze(0).float()
            audio = audio.cuda()
            if sample_rate != 16000:
                if sample_rate not in _resample_buffer:
                    _resample_buffer[sample_rate] = torchaudio.transforms.Resample(
                        orig_freq=sample_rate,
                        new_freq=16000
                    ).to('cuda')
                audio = _resample_buffer[sample_rate](audio)
            # if audio.shape[0] > 1:
            #     audio = audio[:1]
            audio = audio[0]
            audio = audio.cpu().numpy()
            time_step = 0
            while time_step * 16000 < audio.shape[0]:
                audio_segment = audio[time_step * 16000: (time_step + 30) * 16000]
                audios.append(audio_segment)
                indices.append(idx)
                time_step += 30
        pooling_kernel_size = model.config.pooling_kernel_size or 1
        stride = model.conv1.stride[0] * model.conv2.stride[0] * pooling_kernel_size * feature_extractor.hop_length
        all_speech_tokens = [[] for _ in range(len(utts))]
        batch_size = 128
        for start in range(0, len(audios), batch_size):
            features = feature_extractor(audios[start: start + batch_size], sampling_rate=16000,
                                         return_attention_mask=True, return_tensors="pt", device='cuda',
                                         padding="longest", pad_to_multiple_of=stride)
            features = features.to(device="cuda")
            outputs = model(**features)
            speech_tokens = outputs.quantized_token_ids
            attention_mask = features.attention_mask[:, ::model.conv1.stride[0] * model.conv2.stride[0]]
            attention_mask = attention_mask[:, ::model.config.pooling_kernel_size]
            assert attention_mask.shape == speech_tokens.shape
            for i in range(len(speech_tokens)):
                idx = indices[start + i]
                speech_token = speech_tokens[i][attention_mask[i].bool()].tolist()
                all_speech_tokens[idx].extend(speech_token)
        return all_speech_tokens
