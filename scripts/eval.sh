#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Set default parameters
device=0
ref_path="/path/to/LibriSpeech/test-clean"    # test-clean, test-other
rec_path="/path/to/your_reconstructed_audio"  # Path to reconstructed audio files
ref_texts_path="/path/to/LibriSpeech/LibriSpeech_test-clean.jsonl"       # optional, for WER calculation


python -m models.codec.sac.eval.metrics \
    --ref_path "${ref_path}" \
    --rec_path "${rec_path}" \
    --device "${device}" \
    --ref_texts_path "${ref_texts_path}" \


# bash scripts/eval.sh