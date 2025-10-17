# ⚙️ Batch Scripts Guide

1. In the scripts for [audio reconstruction](./scripts/batch/reconstruct.sh), [encoding](./scripts/batch/encode.sh), and [embedding extraction](./scripts/batch/extract_embeddings.sh), you need to provide a **JSON file** specifying the audio files to be processed.
   - Each entry in the JSON file should contain an **`index`** field (used as the basename for storing processed outputs) and a **`wav_path`** field (indicating the path to the input audio).
   An example can be found in [`example/batch_script_data.jsonl`](../example/batch_script_data.jsonl).  
2. To explore **speech decoupling experiments**, we introduce two parameters in the [decoding](./scripts/batch/decode.sh) script: `acoustic_masked` and `semantic_masked`.
   - You can toggle these options to mask the corresponding encoding streams and observe the reconstruction results.
   - Note that when setting `acoustic_masked=true`, it is recommended to also set `normalize=true` to ensure the output volume remains audible.