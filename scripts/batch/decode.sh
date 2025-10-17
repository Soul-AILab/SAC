#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

ckpt_path="/path/to/SAC-16k-62_5Hz.pt"
config_path="configs/sac_16k_62_5.yaml"
device="cuda"
ema_load=true
acoustic_masked=false
semantic_masked=false
normalize=false
sem_per_acu=4   # semantic tokens per acoustic token, e.g., 2 for 37.5Hz, 4 for 62.5Hz

input="/path/to/SAC_62_5/sac_token"  # input .pt file or folder of .pt files

# Save path
save_dir="/path/to/SAC_62_5/decode/wav"

python -m models.codec.sac.pipelines.decode_tokens \
    --input ${input} \
    --save_dir ${save_dir} \
    --config ${config_path} \
    --ckpt ${ckpt_path} \
    --device ${device} \
    --ema_load ${ema_load} \
    --sem_per_acu ${sem_per_acu} \
    --acoustic_masked ${acoustic_masked} \
    --semantic_masked ${semantic_masked} \
    --normalize ${normalize}

# bash scripts/batch/decode.sh