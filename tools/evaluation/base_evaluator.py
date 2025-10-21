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


from pathlib import Path

import os
import soxr
import torch
import torchaudio
import numpy as np


class BaseQualityEvaluator:
    """
    This is the base quality evaluation class.
    Other evaluator should inherit this class.
    """
    def __init__(self) -> None:
        pass

    def _load_audio_paths(self):
        # Override this method in subclasses for specific dataset logic
        raise NotImplementedError

    def resample_audio(self, waveform, org_sr, tgt_sr):
        
        waveform = soxr.resample(
            waveform, org_sr, tgt_sr, quality="VHQ"
            )

        return waveform


    def align_audio(self, ref_waveform, rec_waveform):
        min_len = min(len(ref_waveform), len(rec_waveform))
        ref_waveform = ref_waveform[:min_len]
        rec_waveform = rec_waveform[:min_len]
        return ref_waveform, rec_waveform
        
    def process_audio_pair(self, ref_waveform, rec_waveform, src_sr, tgt_sr):
        if src_sr != tgt_sr:
            ref_waveform = self.resample_audio(ref_waveform, src_sr, tgt_sr)
            rec_waveform = self.resample_audio(rec_waveform, src_sr, tgt_sr)
        ref_waveform, rec_waveform = self.align_audio(ref_waveform, rec_waveform)
        
        return ref_waveform, rec_waveform

    def calculate_metric(self, metric_name, **kwargs):
        metric_function = getattr(self, metric_name, None)
        if metric_function and callable(metric_function):
            return metric_function(**kwargs)
        else:
            raise ValueError(f"Metric {metric_name} is not implemented.")

    def build_path_pairs(self, path1: Path, path2: Path = None, suffix1: str = None, suffix2: str = None):
        """build path pairs of compared data.

        Args;
            path1 (Path):
            path2 (Path): if the generated data and gt data are stored in different folder,
                path2 should be provided.
            suffix1 (src): 
                suffix of data1, e.g., '_rec'
            suffix2 (src):
                suffix of data2, e.g., '_gt'
            
                
        """
        