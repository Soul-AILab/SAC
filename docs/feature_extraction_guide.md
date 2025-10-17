# 🧩 Feature Extraction Guide

This document describes how to extract **semantic tokens (12.5 Hz)** and **semantic features (50 Hz)** offline. During SAC training, they are used as the input of semantic stream and the target of semantic supervision, respectively.

## 1. Extraction Script

We provide a batch extraction script at [`tools/features/glm-4-voice_feat.py`](../../tools/features/glm-4-voice_feat.py). You can place it under the cloned [GLM-4-Voice](https://github.com/zai-org/GLM-4-Voice) repository and run it there. The input `.jsonl` file should include both the audio path and the utterance ID (`utt`).


## 2. Enabling Semantic Feature Output

To enable the output of **semantic features**, you need to slightly modify  
`speech_tokenizer/modeling_whisper.py` in the GLM-4-Voice repository. Specifically, update the `forward()` function of the `WhisperVQEncoder` class to also return the semantic feature. We provide a reference implementation in [`tools/features/modeling_whisper.py`](../../tools/features/modeling_whisper.py) for your convenience.
