#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

# ==== Config ====
ckpt_path="/path/to/SAC-16k-37_5Hz.pt"
config_path="./configs/sac_16k_37_5.yaml"
semantic_tokenizer_path="/path/to/glm-4-voice-tokenizer"
device="cuda"
latent_hop_length=1280  # 12.5Hz -> 1280 (semantic stream frequency)
ema_load=true

# Data paths
jsonfile="/path/to/your_dataset.jsonl"  # JSON file listing the audio files for embedding extraction

# Save path
save_dir="/path/to/your_save_dir"

# ==== Run ====
# By default, outputs combined embedding. For single stream, add --acoustic_only or --semantic_only
python -m models.codec.sac.pipelines.extract_embeddings \
    --config "${config_path}" \
    --ckpt "${ckpt_path}" \
    --jsonfile "${jsonfile}" \
    --semantic_tokenizer_path "${semantic_tokenizer_path}" \
    --device "${device}" \
    --latent_hop_length "${latent_hop_length}" \
    --ema_load "${ema_load}"  \
    --save_dir "${save_dir}"   \
    # --semantic_only


# bash scripts/batch/extract_embeddings.sh
