# 🧪 Evaluation Guide

This document describes how to evaluate reconstruction performance for SAC.


## 1. Speaker Similarity Evaluation

To evaluate speaker similarity, you need to extract **speaker embeddings** from the reconstructed speech and the reference speech. We use the [WavLM-based](https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification) model for speaker verification, please set up the environment as described in that repository (e.g., the "s3prl" package).

For reference, we provide an embedding extraction script at [`tools/speaker/extract_spk_emb.py`](../../main/tools/speaker/extract_spk_emb.py).


## 2. Evaluation Directory Structure

In the `scripts/eval.sh` script, set the variable `rec_path` to the reconstructed audio directory, which should follow this structure:

```
/path/to/reconstructed_wav
├── speaker_embedding/
└── wav/
```


## 3. WER Evaluation

To evaluate **word error rate (WER)**, set `ref_texts_path` to the reference transcription file. The input `.jsonl` file should include:
- `wav_path`: path to the reference audio  
- `text`: corresponding ground-truth transcript
