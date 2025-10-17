#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Set default parameters
semantic_tokenizer_path="/path/to/glm-4-voice-tokenizer"
ckpt_path="/path/to/SAC-16k-37_5Hz.pt"
config_path="configs/sac_16k_37_5.yaml"
save_dir="/path/to/SAC_37_5_recon"
ema_load=true
device=0
semantic_latent_hop_length=1280 # 12.5Hz -> 1280 (semantic stream frequency)

# Set data path
test_data_root="/path/to/LibriSpeech"
jsonfiles=(
    'LibriSpeech_test-clean_local.jsonl'
)

# Run inference for each JSON file
for jsonfile in "${jsonfiles[@]}"; do
    jsonlpath=${test_data_root}/${jsonfile}
    # python -m debugpy --wait-for-client --listen 5678 -m models.codec.sac.pipelines.reconstruct \
    python -m models.codec.sac.pipelines.reconstruct \
        --ckpt "${ckpt_path}" \
        --config "${config_path}" \
        --save_dir "${save_dir}" \
        --jsonfile "${jsonlpath}" \
        --device "${device}" \
        --semantic_tokenizer_path "${semantic_tokenizer_path}" \
        --latent_hop_length "${semantic_latent_hop_length}" \
        --ema_load "${ema_load}"    
        # --plot_codebook_distribution # Uncomment to enable codebook distribution plotting
done

# bash scripts/batch/reconstruct.sh