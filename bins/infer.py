import os

import soundfile as sf
import torch
from models.codec.sac.utils import (
    inference_factory,
    load_whisper_tokenizer,
    process_audio,
)
from tools.features.extract_tokens import extract_speech_token
from utils.file import load_config

# ---- Configurations ----
semantic_tokenizer_path = "/path/to/glm-4-voice-tokenizer"
config_path = "configs/sac_16k_62_5.yaml"
model_path = "/path/to/SAC-16k-62_5Hz.pt"

input_wav = "./example/audio-zh.wav"
output_wav = "./example/audio-zh_recon.wav"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
semantic_latent_hop_length = 1280
sample_rate = 16000
ema_load = True


def main():
    cfg_all = load_config(config_path)
    cfg = cfg_all.get("config", cfg_all)

    whisper_tok_model, feature_extractor = load_whisper_tokenizer(
        semantic_tokenizer_path, device
    )
    model = inference_factory(cfg, {
        "ckpt": model_path,
        "ema_load": ema_load,
        "device": device,
        "config": config_path,
    })

    os.makedirs(os.path.dirname(output_wav) or ".", exist_ok=True)

    wav_in = process_audio(input_wav, cfg, semantic_latent_hop_length)
    wav_in_t = torch.from_numpy(wav_in).unsqueeze(0).unsqueeze(1).float().to(device)

    semantic_tokens = extract_speech_token(
        whisper_tok_model, feature_extractor, [input_wav], sample_rate=cfg["sample_rate"]
    )[0]
    semantic_tokens = torch.tensor(semantic_tokens, dtype=torch.long).unsqueeze(0).to(device)

    # Direct reconstruction example
    # batch = {"semantic_tokens": semantic_tokens, "wav": wav_in_t}
    # with torch.no_grad():
    #     outputs = model(batch)
    #     wav_rec = outputs["recons"].squeeze().detach().cpu().numpy()

    # Reconstruction with encoding & decoding steps
    with torch.no_grad():
        enc_out = model.encode(wav=wav_in_t, semantic_tokens=semantic_tokens, return_zq=False)
        semantic_tokens = enc_out["semantic_tokens"]
        acoustic_tokens = enc_out["acoustic_tokens"]
        dec_out = model.decode(semantic_tokens=semantic_tokens, acoustic_tokens=acoustic_tokens)
        wav_rec = dec_out["recons"].squeeze().detach().cpu().numpy()

    sf.write(output_wav, wav_rec, samplerate=cfg["sample_rate"])
    print(f"[OK] Reconstructed wav saved to: {output_wav}")


if __name__ == "__main__":
    main()

# python -m bins.infer