#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torchaudio
import utils.log as log
from models.codec.sac.utils import inference_factory
from tqdm import tqdm
from utils.file import load_config

SR = 16000  # target sample rate


def parse_args() -> Dict[str, Any]:
    ap = argparse.ArgumentParser("Decode wavs from interleaved SAC tokens")
    ap.add_argument("--input", type=str, required=True, help="a .pt file or a folder of .pt")
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--sem_per_acu", type=int, default=2, help="semantic tokens per acoustic token")
    ap.add_argument(
        "--ema_load", type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=False,
        help="whether to load ema weights (true/false)"
    )
    ap.add_argument(
        "--acoustic_masked", type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=False,
        help="whether to use acoustic masked tokens (true/false)"
    )
    ap.add_argument(
        "--semantic_masked", type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=False,
        help="whether to use semantic masked tokens (true/false)"
    )
    ap.add_argument(
        "--normalize", type=lambda x: str(x).lower() in ["true", "1", "yes"],
        default=False,
        help="whether to normalize the waveform to [-1, 1] (true/false)"
    )
    return vars(ap.parse_args())


def list_pt_files(p: Path) -> List[Path]:
    if p.is_file() and p.suffix == ".pt":
        return [p]
    if p.is_dir():
        return sorted([x for x in p.glob("*.pt") if x.is_file()])
    raise FileNotFoundError(f"Invalid --input: {p}")


def split_interleaved(tokens: torch.Tensor, sem_per_acu: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
    t = tokens.view(-1).long()
    if t.numel() % (1 + sem_per_acu) != 0:
        raise ValueError(
            f"Interleaved length {t.numel()} not divisible by (1 + sem_per_acu={1+sem_per_acu})"
        )
    Ls = t.numel() // (1 + sem_per_acu)

    sem = t[0::(1 + sem_per_acu)]

    acu = torch.empty(sem_per_acu * Ls, dtype=torch.long, device=t.device)
    for i in range(sem_per_acu):
        acu[i::sem_per_acu] = t[(i + 1)::(1 + sem_per_acu)]

    return sem, acu


@torch.no_grad()
def decode_file(model, pt_path: Path, save_dir: Path, device: str, 
                acoustic_masked: bool = False, semantic_masked: bool = False,
                normalize: bool = True, sem_per_acu: int = 2) -> Tuple[float, float, float]:
    """Return (rtf, gen_sec, wall_sec)."""
    toks = torch.load(str(pt_path), map_location="cpu")
    if not isinstance(toks, torch.Tensor):
        raise TypeError(f"{pt_path.name}: expected 1-D LongTensor of interleaved tokens")
    sem, acu = split_interleaved(toks, sem_per_acu)

    sem = sem.unsqueeze(0).to(device)  # [1, Ts]
    acu = acu.unsqueeze(0).to(device)  # [1, Ta]

    t0 = time.perf_counter()
    out = model.decode(semantic_tokens=sem, acoustic_tokens=acu, 
                       acoustic_masked=acoustic_masked, semantic_masked=semantic_masked)
    y = out["recons"].detach().float().cpu().squeeze()
    wall = time.perf_counter() - t0

    if y.ndim == 2:  # [C, T] -> take mono
        y = y[0]
    gen_sec = y.numel() / SR
    rtf = wall / max(gen_sec, 1e-8)

    if normalize:
        peak = y.abs().max()
        if peak > 0:
            y = y / peak  # scale waveform to [-1, 1]

    wav_path = save_dir / (pt_path.stem + ".wav")
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(wav_path), y.view(1, -1), SR)
    return rtf, gen_sec, wall


def main(args: Dict[str, Any]):
    cfg = load_config(args["config"])
    if "config" in cfg:  # support nested
        cfg = cfg["config"]
    model = inference_factory(cfg, args)

    in_path = Path(args["input"])
    save_dir = Path(args["save_dir"])
    files = list_pt_files(in_path)
    if not files:
        raise FileNotFoundError(f"No .pt files under {in_path}")
    
    acoustic_masked = bool(args.get("acoustic_masked", False))
    semantic_masked = bool(args.get("semantic_masked", False))
    normalize = bool(args.get("normalize", False))
    sem_per_acu = int(args.get("sem_per_acu", 2))

    if acoustic_masked:
        log.info("[SAC] Acoustic tokens are masked during decoding!")
    if semantic_masked:
        log.info("[SAC] Semantic tokens are masked during decoding!")
    if normalize:
        log.info("[SAC] Waveform will be normalized to [-1, 1]!")

    tot_gen, tot_wall = 0.0, 0.0
    for f in tqdm(files, desc="Decoding", unit="file"):
        rtf, gen_sec, wall = decode_file(model, f, save_dir, args["device"], acoustic_masked, semantic_masked, normalize, sem_per_acu)
        tot_gen += gen_sec
        tot_wall += wall
        # print(f"[{f.name}] dur={gen_sec:.2f}s time={wall:.2f}s RTF={rtf:.3f}")

    if tot_gen > 0:
        print(f"\nSummary: files={len(files)} total_dur={tot_gen:.2f}s total_time={tot_wall:.2f}s "
              f"avg_RTF={tot_wall / tot_gen:.3f}")


if __name__ == "__main__":
    main(parse_args()) 
