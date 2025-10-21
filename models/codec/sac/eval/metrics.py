# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import argparse
import torch
from tqdm import tqdm
import soundfile as sf
import numpy as np
import torchaudio
from utils.audio import load_audio
from utils.audio import audio_highpass_filter

from tools.evaluation.speech_evaluator import SpeechQualityEvaluator

def _to_mono_float32(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:  # (T, C)
        x = x.mean(axis=1)
    if x.dtype != np.float32:
        x = x.astype(np.float32)
    return x


def _resample_np(wav_np: np.ndarray, orig_sr: int, new_sr: int) -> np.ndarray:
    if orig_sr == new_sr:
        return wav_np
    wav_t = torch.from_numpy(wav_np).unsqueeze(0)  # (1, T)
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=new_sr)
    wav_t = resampler(wav_t)  # (1, T')
    return wav_t.squeeze(0).contiguous().numpy()


def process_audio(wav_path, args_dict):
    """Return wav_in (np.float32), already hp-filtered & padded to hop multiple."""
    wav_raw = load_audio(
        wav_path,
        sampling_rate=args_dict["sample_rate"],
        volume_normalize=args_dict["volume_normalize"],
    )
    if wav_raw is None:
        raise ValueError(f"Failed to load audio from {wav_path}")

    wav_in = wav_raw

    hp_cut = float(args_dict["highpass_cutoff_freq"])
    if hp_cut != 0:
        wav_in = audio_highpass_filter(wav_in, args_dict["sample_rate"], hp_cut)

    return wav_in


def parse_args():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Wav reconstruct quality evaluation.")

    # Add arguments to the parser
    parser.add_argument(
        "--rec_path", type=str, required=True, help="Path to reconstructed wavs."
    )
    parser.add_argument("--ref_path", type=str, required=True, help="Path to original wavs")
    parser.add_argument("--device", type=int, default=0, help="CUDA device number")
    parser.add_argument(
        "--ref_texts_path", type=str, default=None, help="Path to the reference texts file (jsonl format)."
    )
    # parser.add_argument(
    #     "--sample_rate", type=int, default=16000, help="Sample rate of the wavs."
    # )
    # parser.add_argument(
    #     "--volume_normalize",
    #     type=lambda x: str(x).lower() in ["true", "1", "yes"],
    #     default=True,
    #     help="Whether to volume normalize the wavs (true/false)",
    # )
    # parser.add_argument(
    #     "--highpass_cutoff_freq",
    #     type=float,
    #     default=0.0,
    #     help="Highpass cutoff frequency. 0.0 means no highpass filtering.",
    # )
    
    # Parse the arguments
    args = parser.parse_args()
    # Convert Namespace to dictionary
    args_dict = vars(args)

    args_dict["device"] = torch.device(f"cuda:{args.device}")

    return args_dict


def extract_ref_path(args_dict, wavfile):
    """
    Extract the reference path from the reconstructed wav file name.
    """
    rec_filename = os.path.basename(wavfile)  # '61-70968-0000_rec.wav'
    if not rec_filename.endswith("_rec.wav"):
        if rec_filename.endswith(".wav"):
            utt_id = rec_filename.replace(".wav", "")
        elif rec_filename.endswith(".flac"):
            utt_id = rec_filename.replace(".flac", "")
        else:
            raise ValueError(f"Unsupported file format for {rec_filename}. Expected .wav or .flac.")
    else:
        utt_id = rec_filename.replace("_rec.wav", "")  # '61-70968-0000'

    speaker, chapter, _ = utt_id.split("-")  # '61', '70968'
    ref_file = f"{utt_id}.flac"
    ref_path = os.path.join(args_dict['ref_path'], speaker, chapter, ref_file)
    
    return ref_path


def load_text_map(ref_texts_path):
    text_map = {}
    with open(ref_texts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            wav_path = obj.get("wav_path") or obj.get("path") or ""
            text = obj.get("text", "")
            if not wav_path:
                continue
            base = os.path.basename(wav_path)
            text_map[base] = text
    return text_map


def main(args_dict):
    rec_path_dir = os.path.join(args_dict["rec_path"], "wav")
    wavfiles = os.listdir(rec_path_dir)

    ref_texts_path = args_dict.get("ref_texts_path", None)
    ref_text_map = load_text_map(ref_texts_path) if ref_texts_path else {}

    inputs, results, ref_texts = [], [], []
    Evaluator = SpeechQualityEvaluator(args_dict["device"])

    for wavfile in tqdm(wavfiles, desc='load wav files'):
        if os.path.splitext(wavfile)[-1] != '.wav' and os.path.splitext(wavfile)[-1] != '.flac':
            continue
        if wavfile.endswith('_rec.wav'):
            index = '_'.join(os.path.splitext(wavfile)[0].split('_')[:-1])
        else:
            index = os.path.splitext(wavfile)[0]
        rec_path = os.path.join(rec_path_dir, wavfile)
        ref_path = os.path.join(args_dict['ref_path'], f'{index}.wav')

        if not os.path.exists(ref_path):
            ref_path = extract_ref_path(args_dict, rec_path)
            if not os.path.exists(ref_path):
                raise FileNotFoundError(f"Reference file {ref_path} does not exist.")

        # wav_raw = process_audio(ref_path, args_dict)
        # wav_rec = process_audio(rec_path, args_dict)
        wav_raw, raw_sr = sf.read(ref_path)
        wav_rec, rec_sr = sf.read(rec_path)

        wav_raw = _to_mono_float32(wav_raw)
        wav_rec = _to_mono_float32(wav_rec)

        if rec_sr != raw_sr:
            wav_rec = _resample_np(wav_rec, rec_sr, raw_sr)
            rec_sr = raw_sr

        inputs.append(wav_raw)
        results.append(wav_rec)

        if ref_texts_path is not None:
            ref_base = os.path.basename(ref_path)
            text = ref_text_map.get(ref_base, "")
            ref_texts.append(text)

    # rec_path_dir = args_dict["rec_path"]
    # from glob import glob
    # flac_files = sorted(glob(os.path.join(rec_path_dir, "**", "*.flac"), recursive=True))

    # for flac_path in tqdm(flac_files, desc="Loading flac files"):
    #     wav, _ = sf.read(flac_path)
    #     inputs.append(wav)
    #     results.append(wav)

        
    # caculate metrics
    metrics = Evaluator.list_infer(inputs, results, sample_rate=16000, ref_texts=ref_texts)

    # caculate sim
    if os.path.exists(os.path.join(args_dict['rec_path'], 'speaker_embedding')):
        featfiles = os.listdir(os.path.join(args_dict['rec_path'], 'speaker_embedding'))
        sims = []
        for featfile in tqdm(featfiles, desc='load speaker embeddings'):
            if os.path.splitext(featfile)[-1] != '.pt': continue
            if featfile.endswith('_rec.pt'):
                index = '_'.join(os.path.splitext(featfile)[0].split('_')[:-1])
            else:
                index = os.path.splitext(featfile)[0]
            rec_path = os.path.join(args_dict['rec_path'],'speaker_embedding', featfile)
            ref_path = os.path.join(args_dict['ref_path'], 'speaker_embedding', f'{index}.pt')

            feat_raw  = torch.load(ref_path)
            feat = torch.load(rec_path)
            sim = torch.cosine_similarity(feat_raw.unsqueeze(0), feat.unsqueeze(0))
            sims.append(sim.item())
        sim = sum(sims) / len(sims)
        metrics.update({'SIM': sim})
    print(metrics)
    with open(f"{args_dict['rec_path']}/metric2.json", "w") as f:
        f.write(json.dumps(metrics, indent=-4, ensure_ascii=False))
        print(f"Metrics saved to {args_dict['rec_path']}/metric2.json")

if __name__ == "__main__":
    args_dict = parse_args()
    main(args_dict)