#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

# Set default parameters
semantic_tokenizer_path="/path/to/glm-4-voice-tokenizer"
ckpt_path="/path/to/SAC-16k-62_5Hz.pt"
config_path="configs/sac_16k_62_5.yaml"
device="cuda"
batch_size=8        # multi-batch will result in sub-optimal performance
latent_hop=1280     # semantic token hop length
ema_load=true
audio_per_acu=320   # 640 for 37.5Hz, 320 for 62.5Hz

# Set data paths
jsonfile="/path/to/LibriSpeech_test-clean.jsonl"

# Save path
save_dir="/path/to/SAC_62_5"

python -m models.codec.sac.pipelines.encode_tokens \
    --config "${config_path}" \
    --ckpt "${ckpt_path}" \
    --jsonfile "${jsonfile}" \
    --semantic_tokenizer_path "${semantic_tokenizer_path}" \
    --device "${device}" \
    --batch-size "${batch_size}" \
    --latent_hop_length "${latent_hop}" \
    --save_dir "${save_dir}" \
    --ema_load "${ema_load}"  \
    --audio_per_acu "${audio_per_acu}"

# bash scripts/batch/encode.sh