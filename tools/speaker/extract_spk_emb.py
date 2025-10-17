#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
from pathlib import Path

import fire
import soundfile as sf
import torch
from tools.speaker.utils.ecapa_tdnn import ECAPA_TDNN_SMALL
from torchaudio.transforms import Resample
from tqdm import tqdm

MODEL_NAME = "wavlm_large"


def init_model(checkpoint: str = None, use_gpu: bool = True) -> torch.nn.Module:
    model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type=MODEL_NAME, config_path=None)
    if checkpoint:
        sd = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(sd.get("model", sd), strict=False)
    if use_gpu and torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model


def load_and_resample(path: str, target_sr: int = 16000, device="cpu") -> torch.Tensor:
    wav, sr = sf.read(path, always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = torch.from_numpy(wav).float().unsqueeze(0)  # (1, T)
    if sr != target_sr:
        wav = Resample(orig_freq=sr, new_freq=target_sr)(wav)
    return wav.to(device)


def extract_embeddings_single(
    input_dir: str,
    use_gpu: bool = True,
    checkpoint: str = None,
    pattern: str = "**/*",
    exts=("*.wav", "*.flac", "*.mp3", "*.m4a", "*.ogg"),
    overwrite: bool = False,
):
    indir = Path(input_dir).expanduser().resolve()
    assert indir.is_dir(), f"Input directory does not exist: {indir}"

    out_dir = indir / "speaker_embedding"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for ext in exts:
        files.extend(glob.glob(str(indir / pattern / ext), recursive=True))
    files = sorted(set(files))
    if not files:
        print(f"No audio files found in {indir}. Supported: {exts}")
        return

    device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
    model = init_model(checkpoint=checkpoint, use_gpu=(device == "cuda"))

    pbar = tqdm(files, desc="Extracting speaker embeddings", dynamic_ncols=True)
    for f in pbar:
        fpath = Path(f)
        out_path = out_dir / f"{fpath.stem}.pt"
        if out_path.exists() and not overwrite:
            pbar.set_postfix_str("skip (exists)")
            continue

        try:
            wav = load_and_resample(str(fpath), target_sr=16000, device=device)  # (1, T)
            if wav.numel() == 0:
                pbar.set_postfix_str("skip (empty)")
                continue
            with torch.no_grad():
                emb = model(wav)
                if isinstance(emb, (list, tuple)):
                    emb = emb[0]
                emb = emb.squeeze(0).detach().cpu()
            torch.save(emb, out_path)
            pbar.set_postfix_str("ok")
        except Exception as e:
            pbar.set_postfix_str(f"error: {e}")

    print(f"Done. Output directory: {out_dir}")


if __name__ == "__main__":
    fire.Fire(extract_embeddings_single)


# python -m tools.speaker.extract_spk_emb --use_gpu True --checkpoint /path/to/wavlm_large_finetune.pth --input_dir /path/to/your_wav_dir