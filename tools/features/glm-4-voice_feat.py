import os
import sys
from argparse import ArgumentParser

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchaudio
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

sys.path.append("/path/to/GLM-4-Voice-test")

from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import WhisperFeatureExtractor
from utils.file import read_jsonl


class AudioFeatureDataset(Dataset):

    def __init__(self, jsonlfile):
        super().__init__()
        self.metadata = read_jsonl(jsonlfile)
        self._resample_buffer: dict[int, torchaudio.transforms.Resample] = {}

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        meta = self.metadata[idx]
        index = meta["index"]
        utt = meta['utt']
        wav_path = meta['wav_path']
        wav, sr = torchaudio.load(wav_path)
        if sr != 16000:
            if sr not in self._resample_buffer:
                self._resample_buffer[sr] = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=16000
                )
            wav = self._resample_buffer[sr](wav).squeeze()
            sr = 16000
        else: 
            wav = wav.squeeze()
            sr = 16000

        return index, wav, len(wav), utt, sr

    def collate_fn(self, batch):
        wavs = []
        sample_rate = [b[-1] for b in batch]
        indexs = [b[0] for b in batch]
        utts = [b[-2] for b in batch]
        for i in range(len(batch)):
            _, waveform, wav_length, _, _ = batch[i]
            wavs.append(waveform)
        return indexs, utts, wavs, sample_rate


def setup_process(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def extract(rank, args):
    setup_process(rank, args.world_size)
    device = torch.device(f"cuda:{rank}")

    whisper_model = (
        WhisperVQEncoder.from_pretrained(args.tokenizer_path).eval().to(device)
    )
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.tokenizer_path)

    whisper_model = DDP(whisper_model, device_ids=[rank])

    dataset = AudioFeatureDataset(
        args.jsonlfile
    )
    sampler = DistributedSampler(
        dataset, num_replicas=args.world_size, rank=rank, shuffle=False
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=12,
        collate_fn=dataset.collate_fn,
        sampler=sampler,
    )

    save_dir_mix = args.save_dir
    if not os.path.exists(save_dir_mix) and rank == 0:
        os.makedirs(save_dir_mix, exist_ok=True)

    for batch in tqdm(dataloader):
        indexs, utts, wavs, sample_rate = batch
        audios, indices = [], []
        with torch.no_grad():
            for i, wav in enumerate(wavs):
                wav = wav.cpu().numpy()
                time_step = 0
                while time_step * 16000 < wav.shape[0]:
                    audio_segment = wav[time_step * 16000 : (time_step + 30) * 16000]
                    audios.append(audio_segment)
                    indices.append(i)
                    time_step += 30

            pooling_kernel_size = whisper_model.module.config.pooling_kernel_size or 1
            stride = (
                whisper_model.module.conv1.stride[0]
                * whisper_model.module.conv2.stride[0]
                * pooling_kernel_size
                * feature_extractor.hop_length
            )
            # all_speech_tokens = [[] for _ in range(len(indexs))]
            all_speech_tokens = [[] for _ in range(len(utts))]
            all_whisper_hidden_states_50hz = [[] for _ in range(len(utts))]

            batch_size = args.batch_size
            for start in range(0, len(audios), batch_size):
                features = feature_extractor(
                    audios[start : start + batch_size],
                    sampling_rate=16000,
                    return_attention_mask=True,
                    return_tensors="pt",
                    device="cuda",
                    padding="longest",
                    pad_to_multiple_of=stride,
                )
                features = features.to(device="cuda")
                outputs = whisper_model(**features)
                # speech_tokens = outputs.quantized_token_ids
                speech_tokens = outputs["quantized_token_ids"]
                whisper_hidden_states_50hz = outputs.get("whisper_hidden_states_50hz", None)
                # print(outputs)

                if whisper_hidden_states_50hz is None:
                    raise ValueError("whisper_hidden_states_50hz is None")

                attention_mask = features.attention_mask[
                    :,
                    :: whisper_model.module.conv1.stride[0]
                    * whisper_model.module.conv2.stride[0],
                ]
                ori_attention_mask = attention_mask
                attention_mask = attention_mask[
                    :, :: whisper_model.module.config.pooling_kernel_size
                ]
                assert attention_mask.shape == speech_tokens.shape
                for i in range(len(speech_tokens)):
                    idx = indices[start + i]
                    speech_token = speech_tokens[i][attention_mask[i].bool()].tolist()
                    whisper_feature_50hz = whisper_hidden_states_50hz[i].transpose(0, 1)
                    whisper_feature_50hz = whisper_feature_50hz[ori_attention_mask[i].bool()]

                    all_speech_tokens[idx].extend(speech_token)
                    all_whisper_hidden_states_50hz[idx].append(whisper_feature_50hz)

        for i in range(len(utts)):
            # index = indexs[i]
            utt = utts[i]
            speech_token = (
                torch.tensor(all_speech_tokens[i], dtype=torch.int32).detach().cpu()
            )

            wh50_save = torch.cat(all_whisper_hidden_states_50hz[i], dim=0).half().cpu()


            if not torch.isnan(speech_token).any():
                torch.save(speech_token, f"{save_dir_mix}/{utt}.pt")

            if not torch.isnan(wh50_save).any():
                torch.save(wh50_save, f"{args.feat_50hz_save_dir}/{utt}.pt")


def main(args):
    # extract(0, args)
    mp.spawn(extract, args=(args,), nprocs=args.world_size)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--jsonlfile", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--feat_50hz_save_dir", type=str, required=True)
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count())
    args = parser.parse_args()

    main(args)

# Example usage:
# python /path/to/glm_tokenize.py --tokenizer_path /path/to/glm-4-voice-tokenizer --jsonlfile /path/to/LibriSpeech_960h_train_mini_test.jsonl --batch_size 256 --save_dir /path/to/data --feat_50hz_save_dir /path/to/data