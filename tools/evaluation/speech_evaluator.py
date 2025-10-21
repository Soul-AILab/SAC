import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
import soundfile
import soxr
import torch
import torchaudio
from tools.evaluation.base_evaluator import BaseQualityEvaluator
from tools.evaluation.metircs.compute_wer import Calculator, characterize, normalize
from tools.evaluation.metircs.periodicity import VUVF1
from tools.evaluation.metircs.utmos import UTMOSScore
from torchmetrics.audio import (
    PerceptualEvaluationSpeechQuality,
    ShortTimeObjectiveIntelligibility,
)
from tqdm import tqdm
from transformers import HubertForCTC, Wav2Vec2Processor


class SpeechQualityEvaluator(BaseQualityEvaluator):
    def __init__(self, device: torch.device = None):
        super().__init__()
        self.sr_pesq = 16000
        self.sr_stoi = 16000
        self.sr_utmos = 16000
        self.device = device
        self.stoi = ShortTimeObjectiveIntelligibility(self.sr_stoi)
        self.utmos = UTMOSScore(device)
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.hubert = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
        self.calculator = Calculator()
        self.pesq_wb = PerceptualEvaluationSpeechQuality(self.sr_pesq, mode="wb")
        self.pesq_nb = PerceptualEvaluationSpeechQuality(self.sr_pesq, mode="nb")

    def pesq_score(
        self, ref_waveform: np.ndarray, rec_waveform: np.ndarray
    ):
        pesq_nb = self.pesq_nb(
            torch.from_numpy(rec_waveform).float(),
            torch.from_numpy(ref_waveform).float(),
        ).numpy()

        pesq_wb = self.pesq_wb(
            torch.from_numpy(rec_waveform).float(),
            torch.from_numpy(ref_waveform).float(),
        ).numpy()

        return pesq_nb, pesq_wb


    def stoi_score(
        self, ref_waveform: np.ndarray, rec_waveform: np.ndarray
    ):
        return self.stoi(
            torch.from_numpy(ref_waveform).float(),
            torch.from_numpy(rec_waveform).float(),
        ).numpy()
    
    def utmos_score(
        self, ref_waveform: np.ndarray
    ):
        return self.utmos.score(
            torch.from_numpy(ref_waveform).float().to(self.device)
        ).numpy()

    def cal_wer(
        self, ref_texts: str, rec_list: List[np.ndarray]
    ):
        results = []
        for ref_text, rec_np in tqdm(zip(ref_texts, rec_list), 
                                    total=len(ref_texts), 
                                    desc="Calculating WER"):
            audio = torch.from_numpy(rec_np).float().unsqueeze(0)
            logits = self.hubert(audio).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.decode(predicted_ids[0])
            transcription = characterize(transcription)
            transcription = normalize(transcription, [], False, None)
            ref_text = characterize(ref_text)
            ref_text = normalize(ref_text, [], False, None)
            result = self.calculator.calculate(transcription, ref_text)
            results.append(result)
        
        N, S, D, I = 0.0, 0.0, 0.0, 0.0
        for result in results:
            N += result["all"]
            S += result["sub"]
            D += result["del"]
            I += result["ins"]
        print ("N", N, S, D, I)
        wer = (S + D + I) / N
        return wer

    def commit(
        self, ref_waveform: np.ndarray, rec_waveform: np.ndarray, sample_rate: int
    ):
        
        ref_waveform, rec_waveform = self.process_audio_pair(
            ref_waveform, rec_waveform, src_sr=sample_rate, tgt_sr=self.sr_stoi
        )

        pesq_nb, pesq_wb = self.pesq_score(ref_waveform, rec_waveform)
        
        stoi = self.stoi_score(ref_waveform, rec_waveform)

        utmos = self.utmos_score(rec_waveform)

        return {"pesq_nb": pesq_nb, "pesq_wb": pesq_wb, "stoi": stoi, "utmos": utmos}

    def list_infer(
        self, ref_list: List[np.ndarray], rec_list: List[np.ndarray], sample_rate: int, ref_texts: List[str] = None
    ):
        results = defaultdict(list)
        assert len(ref_list) == len(rec_list)
        
        # futures = []
        # with ThreadPoolExecutor(max_workers=16) as pool:
        #     for ref, rec in tqdm(zip(ref_list, rec_list), total=len(ref_list), desc="caculating metrics"):
        #         futures.append(pool.submit(self.commit, ref, rec, sample_rate))

        #     for f in tqdm(futures, desc='cal'):
        #         outputs = f.result()
        #         for k, v in outputs.items():
        #             results[k].append(v)

        for ref, rec in tqdm(zip(ref_list, rec_list), desc="calculating metrics", total=len(ref_list)):
            try:
                outputs = self.commit(ref, rec, sample_rate)
                for k, v in outputs.items():
                    results[k].append(v)
            except Exception as e:
                print(f"Error processing sample: {ref} vs {rec}, error: {e}")

        for k in results.keys():
            values = np.array(results[k])
            m, std = values.mean(), values.std()
            results[k] = [float(m), float(std)]
        
        if ref_texts is not None and len(ref_texts) != 0:
            wer = self.cal_wer(ref_texts, rec_list)
            results["wer"] = [wer, 0.0]

        return results

    def read_audio_folder(
        self, ref_folder: str, rec_folder: str, ref_text_file: str = None
    ):
        ref_list, rec_list = [], []
        ref_text_dict = {}
        if ref_text_file is not None:
            with open(ref_text_file, "r") as f:
                for line in f.readlines():
                    line = line.strip()
                    sid, text = line.split("|")
                    ref_text_dict[sid + ".wav"] = text
        ref_texts = []
        for ref_file in tqdm(os.listdir(ref_folder), desc="reading audio files"):
            if ref_text_file is not None and ref_file not in ref_text_dict:
                continue
            ref_path = os.path.join(ref_folder, ref_file)
            rec_path = os.path.join(rec_folder, ref_file)
            ref, _ = soundfile.read(ref_path)
            rec, _ = soundfile.read(rec_path)
            ref_list.append(ref)
            rec_list.append(rec)
            ref_texts.append(ref_text_dict[ref_file])
        return ref_list, rec_list, ref_texts

    def audio_folder_infer(
        self, ref_folder: str, rec_folder: str, sample_rate: int, ref_text_file: str = None
    ):
        ref_list, rec_list, ref_texts = self.read_audio_folder(ref_folder, rec_folder, ref_text_file)
        return self.list_infer(ref_list, rec_list, sample_rate, ref_texts)


if __name__ == "__main__":
    device_id = 0
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    evaluator = SpeechQualityEvaluator(device)
    ref_dir = ""
    syn_dir = ""
    ref_text_file = "test_clean_trans_norm.txt"
    print (evaluator.audio_folder_infer(ref_dir, syn_dir, 16000, ref_text_file))
