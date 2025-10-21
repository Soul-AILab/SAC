import os
import random
from typing import Dict, List

import numpy as np
import torch
from models.base.base_dataloader import BaseDataset
from omegaconf import DictConfig
from utils.audio import audio_highpass_filter, load_audio

class SSLWAVDataset(BaseDataset):
    """Initialize the dataset with the given configuration.

    Args:
        config (DictConfig):
            Configuration dictionary specifying dataset parameters.
        mode (str):
            Dataset mode, typically 'train' or 'val'.
    """

    def __init__(
        self,
        config: DictConfig,
        mode: str = "train",
        **kwargs,
    ) -> None:
        super().__init__(config, mode)

    def fetch_data(self, elem: Dict) -> Dict:
        """Fetch a single sample.

        Args:
            elem (Dict): A dictionary with the key 'index' as index.

        Returns:
            Dict:
        """
        utt = elem["utt"]
        cfg = self.config
        ssl_path = elem['ssl_path'] if 'ssl_path' in elem else None
        semantic_token_path = elem['semantic_token_path'] if 'semantic_token_path' in elem else None
        wav_dir = elem["wav_path"]

        try:
            sr = int(cfg["sample_rate"])
            hop = int(cfg["latent_hop_length"])
            seg_dur = float(cfg["segment_duration"])
            hp_cut = float(cfg["highpass_cutoff_freq"])
            align_k = int(cfg["align_multiple"])
            offline = bool(cfg["offline_feature_extracted"])
            ssl_ratio = int(cfg["ssl_per_sem_ratio"])
            feat, ssl_feat, sim_feat = None, None, None

            if offline:
                feat = torch.load(f'{semantic_token_path}/{utt}.pt', weights_only=False)
                ssl_feat = torch.load(f'{ssl_path}/{utt}.pt', weights_only=False)
                T_tok = int(feat.shape[0])
                T_ssl = int(ssl_feat.shape[0])

                if feat is None or ssl_feat is None:
                    raise ValueError("semantic feat / ssl_feat is None")
            else:
                T_tok, T_ssl = None, None

            wav = load_audio(wav_dir, sr, volume_normalize=True, length=None)
            if hp_cut != 0:
                wav = audio_highpass_filter(wav, sr, hp_cut)
                if wav is None:
                    raise ValueError("highpass returned None")

            T_wav = int(len(wav) // hop)
            if T_tok is None:
                T = T_wav
            else:
                T_sem_from_ssl = (T_ssl // ssl_ratio) if T_ssl is not None else T_wav
                T = min(T_tok, T_wav, T_sem_from_ssl)
            
            length = T // align_k * align_k
            wav_length = length * hop
            wav = wav[:wav_length]

            if feat is not None:
                feat = feat[:length]

            if ssl_feat is not None:
                ssl_len = length * ssl_ratio
                ssl_feat = ssl_feat[:ssl_len]

            if not self.train:
                cur_dur = (length * hop) / sr
                seg_dur_eff = min(cur_dur, float(cfg["max_val_duration"]))
            else:
                seg_dur_eff = seg_dur

            seg_T = int(sr * seg_dur_eff // hop)
            seg_T = (seg_T // align_k) * align_k
            wav_segment_length = seg_T * hop
            ssl_segment_length = seg_T * ssl_ratio

            if wav_segment_length > wav_length:
                wav = np.pad(wav, (0, int(wav_segment_length - wav_length)))
                if feat is not None:                    
                    pad_tok = torch.zeros(seg_T - length, dtype=feat.dtype, device=feat.device)
                    feat = torch.cat([feat, pad_tok], dim=0)
                if ssl_feat is not None:
                    Dssl = ssl_feat.shape[1]
                    ssl_pad = torch.zeros(ssl_segment_length - (length * ssl_ratio), Dssl,
                                          dtype=ssl_feat.dtype, device=ssl_feat.device)
                    ssl_feat = torch.cat([ssl_feat, ssl_pad], dim=0)

                start_indice = 0
            
            else:
                if not self.train:
                    start_indice = 0
                else:
                    hi = max(0, length - seg_T)
                    start_indice = random.randint(0, hi)

            wav = torch.from_numpy(wav)
 
            end_indice = start_indice + seg_T
            wav_start_indice = start_indice * hop
            wav_end_indice = end_indice * hop

            feat_segment = feat[start_indice:end_indice] if feat is not None else None
            wav_segment = wav[wav_start_indice:wav_end_indice]

            if ssl_feat is not None:
                ssl_start_indice = start_indice * ssl_ratio
                ssl_end_indice = end_indice * ssl_ratio
                ssl_feat_segment = ssl_feat[ssl_start_indice:ssl_end_indice]
                ssl_feat_segment = ssl_feat_segment.transpose(0, 1)  # [D, T]
            else:
                ssl_feat_segment = None

            return {
                "index": utt,
                "semantic_tokens": feat_segment.to(torch.long) if feat_segment is not None else None,
                "wav": wav_segment.float(),
                "ssl_feat": ssl_feat_segment.float() if ssl_feat_segment is not None else None,
                "sim_feat": sim_feat.float() if sim_feat is not None else None,
            }

        except Exception as e:
            print(f"[SSLWAVDataset] Bad case in fetch_data (utt={utt}): {e}")
            return {
                "index": utt,
                "semantic_tokens": None,
                "wav": None,
                "ssl_feat": None,
            }

    def filter(self, elem: dict):
        """Filter out bad data. Return True if the data is kept."""
        if elem["semantic_tokens"] is not None:
            return True
        else:
            return False

    def padding(self, batch: List[dict]):
        """Padding the batch data into training data

        Args:
            batch (List[dict])
        """
        assert isinstance(batch, list)
        collate_batch = {}

        for k in ("index",):
            collate_batch[k] = [b[k] for b in batch]

        for k in ("semantic_tokens", "wav", "ssl_feat", "sim_feat"):
            v = [b[k] for b in batch]
            if v[0] is not None:
                collate_batch[k] = torch.stack(v, dim=0)
            else:
                collate_batch[k] = None

        return collate_batch


# test
if __name__ == "__main__":
    import os
    import time

    import hydra
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from utils.file import load_config

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "10081"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )

    config = load_config("./configs/sac_16k_37_5.yaml")

    train_dataset_sampler = hydra.utils.instantiate(config["dataloader"], config)

    val_dataset_sampler = hydra.utils.instantiate(
        config["dataloader"], config, mode="train"
    )
    train_dataset = train_dataset_sampler.sample()
    val_dataset = val_dataset_sampler.sample()

    # dataset_sampler = SQcodeDataset(config)
    # dataset = dataset_sampler.sample()

    generator = torch.Generator()
    generator.manual_seed(111)

    data_loader = DataLoader(
        val_dataset,
        batch_size=None,
        pin_memory=False,
        num_workers=8,
        persistent_workers=True,
        generator=generator,
        prefetch_factor=10,
    )

    sample_num = 0
    for batch in tqdm(data_loader):
        print(
            "batch_size {} feat size {}".format(
                len(batch["index"]), batch["feat"].shape
            )
        )
        sample_num += len(batch["index"])

    print('total sample', sample_num)