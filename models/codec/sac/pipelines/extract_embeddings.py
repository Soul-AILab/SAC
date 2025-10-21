#!/usr/bin/env python3
import argparse
import os
from typing import Any, Dict, List

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


def parse_args() -> Dict[str, Any]:
    ap = argparse.ArgumentParser("Extract embeddings (semantic/acoustic/combined) from JSONL manifest.")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--jsonfile", type=str, required=True, help="Manifest with fields: index, wav_path")
    ap.add_argument("--semantic_tokenizer_path", type=str, required=True)
    ap.add_argument("--latent_hop_length", type=int, default=1280)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--ema_load", type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=False,
        help="whether to load ema weights (true/false)"
    )
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument(
        "--acoustic_only", action="store_true",
        help="Return only acoustic embedding"
    )
    ap.add_argument(
        "--semantic_only", action="store_true",
        help="Return only semantic embedding"
    )
    return vars(ap.parse_args())


def main(args: Dict[str, Any]):
    # ---- load cfg, model ----
    cfg = load_config(args["config"])
    if "config" in cfg.keys():
        cfg = cfg["config"]
    device = args["device"]

    whisper_model, feature_extractor = load_whisper_tokenizer(
        args["semantic_tokenizer_path"], device
    )
    model = inference_factory(cfg, args)

    os.makedirs(args["save_dir"], exist_ok=True)

    # ---- read manifest ----
    metadata = read_jsonl(args["jsonfile"])
    items = [(m["index"], m["wav_path"]) for m in metadata]

    for idx, wav_path in tqdm(items, desc="Extracting embeddings"):
        try:
            # ---- preprocess audio ----
            wav = process_audio(wav_path, cfg, args["latent_hop_length"])
            wav_tensor = torch.from_numpy(wav).unsqueeze(0).to(device)  # [1,L]

            # ---- extract semantic tokens ----
            sem_list: List[List[int]] = extract_speech_token(
                whisper_model, feature_extractor, [wav_path],
                sample_rate=cfg["sample_rate"]
            )
            sem_tensor = torch.tensor(sem_list[0], dtype=torch.long, device=device).unsqueeze(0)  # [1,T]

            # ---- get embedding ----
            with torch.no_grad():
                emb = model.get_embedding(
                    wav=wav_tensor,
                    semantic_tokens=sem_tensor,
                    acoustic_only=args["acoustic_only"],
                    semantic_only=args["semantic_only"],
                )  # [1,T,D]

            # ---- save ----
            save_path = os.path.join(args["save_dir"], f"{idx}.pt")
            torch.save(emb.cpu(), save_path)
        except Exception as e:
            print(f"[WARN] failed at index={idx}, wav={wav_path}, error: {e}")
            continue


if __name__ == "__main__":
    args = parse_args()
    main(args)
