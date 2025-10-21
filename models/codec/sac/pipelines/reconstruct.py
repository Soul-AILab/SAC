import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from models.codec.sac.utils import (
    inference_factory,
    load_whisper_tokenizer,
    process_audio,
)
from tools.features.extract_tokens import extract_speech_token
from tqdm import tqdm
from utils.file import load_config, read_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Run wav codec inference.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file."
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the .pt file.")
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Path to save generated audios"
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device number")
    parser.add_argument(
        "--jsonfile", type=str, required=True, help="Path to JSONL file"
    )
    parser.add_argument("--latent_hop_length", type=int, default=320)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--semantic_tokenizer_path", type=str)
    parser.add_argument(
        "--ema_load", type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=False,
        help="whether to load ema weights (true/false)"
    )
    parser.add_argument(
        "--plot_codebook_distribution",
        type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=False,
        help="whether to plot codebook distribution (true/false)"
    )
    args = parser.parse_args()
    args_dict = vars(args)

    args_dict["device"] = torch.device(f"cuda:{args.device}")

    return args_dict


def plot_codebook_distribution(code_counts, save_dir, basename, utilization, activated):
    """
    Plot codebook usage distribution (index order + sorted by prob).
    
    Args:
        code_counts (np.ndarray): counts of each code.
        save_dir (str): root save directory.
        basename (str): experiment name.
        utilization (float): utilization percentage.
        activated (int): number of activated codes.
    """
    os.makedirs(save_dir, exist_ok=True)
    prob_dist = code_counts / code_counts.sum()  # normalize

    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(len(code_counts)), prob_dist, width=1.0, color="steelblue", edgecolor="black")
    plt.xlabel("Codebook Index")
    plt.ylabel("Probability")
    plt.title(f"Codebook Usage (Index Order) - {activated}/{len(code_counts)}, Util={utilization:.2f}%")
    plt.tight_layout()
    out_path1 = os.path.join(save_dir, f"{basename}_codebook_distribution.png")
    plt.savefig(out_path1, dpi=300)
    plt.close()
    print(f"Saved: {out_path1}")

    sorted_probs = np.sort(prob_dist)[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(len(sorted_probs)), sorted_probs, width=1.0, color="darkorange", edgecolor="black")
    plt.xlabel("Rank (sorted)")
    plt.ylabel("Probability")
    plt.title(f"Codebook Usage (Sorted by Prob) - {activated}/{len(code_counts)}, Util={utilization:.2f}%")
    plt.tight_layout()
    out_path2 = os.path.join(save_dir, f"{basename}_codebook_distribution_sorted.png")
    plt.savefig(out_path2, dpi=300)
    plt.close()
    print(f"Saved: {out_path2}")


def update_code_counts(tokens: np.ndarray, K: int):
    t = np.asarray(tokens).reshape(-1)
    mask = (t >= 0) & (t < K)
    if mask.any():
        vals = t[mask]
        counts = np.bincount(vals, minlength=K)
        return counts
    return np.zeros(K, dtype=np.int64)


def main(args_dict):
    cfg = load_config(args_dict["config"])
    if "config" in cfg.keys():
        cfg = cfg["config"]

    whisper_model, feature_extractor = load_whisper_tokenizer(args_dict['semantic_tokenizer_path'], args_dict["device"])
    model = inference_factory(cfg, args_dict)
    
    jsonfile = args_dict["jsonfile"]

    basename = os.path.basename(jsonfile).split(".")[0]
    save_dir = os.path.join(args_dict["save_dir"], basename, "wav")
    os.makedirs(save_dir, exist_ok=True)

    metadata = read_jsonl(jsonfile)
    K = int(model.acoustic_quantizer.codebook_size)
    code_counts = np.zeros(K, dtype=np.int64)
    
    desc = f"Reconstructing audio from {basename}"
    for meta in tqdm(metadata, desc=desc):
        index = meta["index"]
        save_path = f"{save_dir}/{index}_rec.wav"
        wav_path = meta["wav_path"]
        wav_in = process_audio(wav_path, cfg, args_dict["latent_hop_length"])
        wav_in = torch.from_numpy(wav_in).unsqueeze(0).float().to(args_dict["device"])
        wav_in = wav_in.unsqueeze(1)  # Add channel dimension
        semantic_tokens = extract_speech_token(whisper_model, feature_extractor, [wav_path], sample_rate=cfg["sample_rate"])[0]
        semantic_tokens = torch.tensor(semantic_tokens, dtype=torch.int32).unsqueeze(0)
        semantic_tokens = semantic_tokens.to(torch.long).to(args_dict["device"])
        batch = {'semantic_tokens': semantic_tokens, "wav": wav_in}

        try:
            with torch.no_grad():
                outputs = model(batch)
            wav = outputs["recons"].squeeze().detach().cpu().numpy()
            quantized_tokens = outputs["quantized_tokens"].squeeze().detach().cpu().numpy()
            sf.write(save_path, wav, samplerate=cfg["sample_rate"])

            code_counts += update_code_counts(quantized_tokens, K)
        # inputs.append(wav_raw)
        # results.append(wav)
        except Exception as e:
            print(e)
    #     continue
    activated = (code_counts > 0).sum()
    utilization = activated / K * 100
    print(f"Activated codes: {activated}/{K}, Utilization: {utilization:.2f}%")

    save_dir_ = os.path.join(args_dict["save_dir"], basename)
    if args_dict.get("plot_codebook_distribution", False):
        plot_codebook_distribution(code_counts, save_dir_, basename, utilization, activated)

if __name__ == "__main__":
    args_dict = parse_args()
    main(args_dict)
