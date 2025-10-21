#!/usr/bin/env python3
import argparse
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from models.codec.sac.utils import (
    inference_factory,
    load_whisper_tokenizer,
    process_audio,
)
from tools.features.extract_tokens import extract_speech_token
from tqdm import tqdm
from utils.file import load_config, read_jsonl

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

def pad_stack_wavs(
    wavs: List[np.ndarray],
    audio_per_acu: int,               # e.g., 640 for 16kHz/25Hz
) -> Tuple[torch.Tensor, torch.BoolTensor, torch.LongTensor]:
    """
    Pad 1D wav arrays to batch max length, and build acoustic-step mask by ratio.
    """
    lengths = [w.shape[0] for w in wavs]
    Lmax = max(lengths)

    # batch-level wav pad
    batch = np.stack([np.pad(w, (0, Lmax - w.shape[0]), mode="constant") for w in wavs], axis=0)
    wav_batch = torch.from_numpy(batch).unsqueeze(1).float()  # [B,1,Lmax]

    # acoustic steps
    T_a_max = ceil_div(Lmax, audio_per_acu)
    acu_len = torch.tensor([ceil_div(L, audio_per_acu) for L in lengths], dtype=torch.long)  # [B]

    return wav_batch, acu_len


def pad_stack_tokens(tokens_list: List[List[int]], pad_id: int) -> torch.LongTensor:
    """Pad token sequences to max length in batch -> LongTensor [B,Tmax]."""
    max_T = max(len(t) for t in tokens_list)
    padded = np.full((len(tokens_list), max_T), pad_id, dtype=np.int64)
    for i, seq in enumerate(tokens_list):
        padded[i, :len(seq)] = np.asarray(seq, dtype=np.int64)
    return torch.from_numpy(padded)  # [B,T]


def chunked(lst: List[Any], n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def trim_semantic_by_pad_id(
    sem_batch: torch.LongTensor,  # [B, Tmax]
    pad_id: int
) -> List[torch.Tensor]:
    """
    Return per-item trimmed semantic tokens (remove trailing pad_id).
    """
    trimmed = []
    for i in range(sem_batch.size(0)):
        row = sem_batch[i]
        valid = (row != pad_id).nonzero(as_tuple=False).squeeze(-1)
        end = int(valid[-1].item()) + 1 if valid.numel() > 0 else 0
        trimmed.append(row[:end].clone())
    return trimmed


def save_tokens_pt(
    indices: List[str],
    sem_batch: torch.LongTensor,   # [B, T_s_max]
    acu_batch: torch.LongTensor,   # [B, T_a_max]
    output_dir: str,
    pad_id: int,
    acu_len: Optional[torch.LongTensor] = None,
    save_apart: bool = False,
    sem_per_acu: int = 2
):
    sad_dir = os.path.join(output_dir, "sac_token")
    os.makedirs(sad_dir, exist_ok=True)

    if save_apart:
        sem_dir = os.path.join(output_dir, "semantic")
        acu_dir = os.path.join(output_dir, "acoustic")
        os.makedirs(sem_dir, exist_ok=True)
        os.makedirs(acu_dir, exist_ok=True)

    sem_trimmed = trim_semantic_by_pad_id(sem_batch, pad_id)  # List[Tensor], 1-D long each

    for i, idx in enumerate(indices):
        s = sem_trimmed[i].to("cpu")
        a = (acu_batch[i, : int(acu_len[i])] if acu_len is not None else acu_batch[i]).to("cpu")

        if save_apart:
            torch.save(s, os.path.join(sem_dir, f"{idx}.pt"))
            torch.save(a, os.path.join(acu_dir, f"{idx}.pt"))

        # ratio check: len(a) must be exactly sem_per_acu * len(s)
        if a.numel() != sem_per_acu * s.numel():
            raise ValueError(f"[{idx}] acoustic length {a.numel()} != {sem_per_acu} * semantic length {s.numel()}")

        # interleave as 1:sem_per_acu
        Ls = s.numel()
        s = s.view(-1).long()
        a = a.view(-1).long()
        parts = []
        for j in range(Ls):
            parts.append(s[j:j+1])
            parts.append(a[sem_per_acu*j:sem_per_acu*j+sem_per_acu])
        sad = torch.cat(parts, dim=0)

        torch.save(sad, os.path.join(sad_dir, f"{idx}.pt"))


def parse_args() -> Dict[str, Any]:
    ap = argparse.ArgumentParser("True batched encode for semantic/acoustic tokens.")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--jsonfile", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--semantic_tokenizer_path", type=str, required=True)
    ap.add_argument("--latent_hop_length", type=int, default=1280)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--audio_per_acu", type=int, default=640)  # e.g., 640 for 16kHz/25Hz
    ap.add_argument(
        "--ema_load", type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=False,
        help="whether to load ema weights (true/false)"
    )
    return vars(ap.parse_args())


def main(args_dict: Dict[str, Any]):
    # ---- load cfg, models ----
    cfg = load_config(args_dict["config"])
    if "config" in cfg.keys():
        cfg = cfg["config"]
    device = args_dict["device"]
    audio_per_acu = args_dict["audio_per_acu"]
    sem_per_acu = int(cfg["sample_rate"] // (audio_per_acu * 12.5))  # e.g., 2 for 16kHz+25Hz, 4 for 16kHz+50Hz

    whisper_model, feature_extractor = load_whisper_tokenizer(
        args_dict["semantic_tokenizer_path"], device
    )
    model = inference_factory(cfg, args_dict)

    # pad id for semantic tokens
    pad_id = getattr(getattr(model, "config", object()), "speech_pad_token", None)
    if pad_id is None:
        pad_id = getattr(getattr(model, "config", object()), "pad_token_id", 0)
    pad_id = int(pad_id)

    # ---- read manifest ----
    metadata = read_jsonl(args_dict["jsonfile"])
    items = [(m["index"], m["wav_path"]) for m in metadata]

    desc = f"Batched-encoding from {os.path.basename(args_dict['jsonfile'])}"
    for batch_items in tqdm(list(chunked(items, args_dict["batch_size"])), desc=desc):
        indices = [x[0] for x in batch_items]
        paths   = [x[1] for x in batch_items]

        try:
            wavs = [process_audio(p, cfg, args_dict["latent_hop_length"]) for p in paths]
            wav_batch, acu_len = pad_stack_wavs(wavs, audio_per_acu)
            wav_batch = wav_batch.to(device)

            sem_list: List[List[int]] = extract_speech_token(
                whisper_model, feature_extractor, paths, sample_rate=cfg["sample_rate"]
            )
            sem_batch = pad_stack_tokens(sem_list, pad_id=pad_id).to(device)  # [B,Tmax]

            with torch.no_grad():
                enc_out = model.encode(wav=wav_batch, semantic_tokens=sem_batch, return_zq=False)

            sem_ids = enc_out["semantic_tokens"]        # [B,T_s_max]
            acu_ids = enc_out["acoustic_tokens"]        # [B,T_a_max]

            save_tokens_pt(
                indices=indices,
                sem_batch=sem_ids,
                acu_batch=acu_ids,
                output_dir=args_dict["save_dir"],
                pad_id=pad_id,
                acu_len=acu_len.to(acu_ids.device),
                save_apart=False,
                sem_per_acu=sem_per_acu
            )

        except Exception as e:
            print(f"[WARN] batch starting with index={indices[0]} failed: {e}")
            continue


if __name__ == "__main__":
    args = parse_args()
    main(args)
