#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from pathlib import Path

import fire
import torch
from tqdm import tqdm


def compute_similarity(ref_path: str, rec_path: str, out_name: str = "similarity.json", spk_model: str = "wavlm_large"):
    """
    Compute cosine similarity between embeddings in ref_path and rec_path.
    
    Args:
        ref_path: Path to reference directory (contains speaker_embedding/*.pt)
        rec_path: Path to recognition directory (contains speaker_embedding/*_rec.pt)
        out_name: JSON filename for saving similarity result (default: similarity.json)
    """
    if spk_model == "wavlm_large":
        sub_dir_name = "speaker_embedding"
    elif spk_model == "eres2net_base":
        sub_dir_name = "speaker_embeddings-eres2net_base"
        out_name = "similarity_eres2net_base.json"
    else:
        raise ValueError(f"Unsupported spk_model: {spk_model}")

    ref_emb_dir = Path(ref_path) / sub_dir_name
    rec_emb_dir = Path(rec_path) / sub_dir_name

    assert ref_emb_dir.is_dir(), f"Reference embedding folder not found: {ref_emb_dir}"
    assert rec_emb_dir.is_dir(), f"Recognition embedding folder not found: {rec_emb_dir}"

    featfiles = [f for f in os.listdir(rec_emb_dir) if f.endswith(".pt")]
    sims = []

    for featfile in tqdm(featfiles, desc="Compute similarities"):
        # rec file like "xxx_rec.pt"
        stem = Path(featfile).stem
        index = stem[:-4] if stem.endswith("_rec") else stem
        ref_file = ref_emb_dir / f"{index}.pt"
        rec_file = rec_emb_dir / featfile

        if not ref_file.exists():
            tqdm.write(f"[Skip] No reference file for {featfile}")
            continue

        try:
            ref_feat = torch.load(ref_file)
            rec_feat = torch.load(rec_file)

            sim = torch.cosine_similarity(
                ref_feat.unsqueeze(0), rec_feat.unsqueeze(0)
            ).item()
            sims.append({"index": index, "sim": sim})
        except Exception as e:
            tqdm.write(f"[Error] {featfile}: {e}")

    # aggregate
    avg_sim = sum(d["sim"] for d in sims) / len(sims) if sims else 0.0
    result = {"average_sim": avg_sim, "details": sims}

    # save json
    out_file = Path(rec_path) / out_name
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Done. Results saved to {out_file}")
    print(f"Average SIM: {avg_sim:.4f}")


if __name__ == "__main__":
    fire.Fire(compute_similarity)

# Example usage:
# python tools/speaker/spk_sim_cal.py --ref_path /path/to/LibriSpeech/test-clean --rec_path /path/to/your_recon_dir