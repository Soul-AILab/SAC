<div align="center">
  <h1>
  SAC: Neural Speech Codec with Semantic-Acoustic Dual-Stream Quantization
  </h1>

  <p>
    <a href="https://sac-codec.github.io/">
      <img src="https://img.shields.io/badge/🌐%20Demo-Page-brightgreen" alt="Demo Page">
    </a>
    <a href="https://arxiv.org/abs/2510.16841">
      <img src="https://img.shields.io/badge/arXiv-2510.16841-blueviolet?logo=arxiv&logoColor=white" alt="arXiv">
    </a>
    <a href="https://huggingface.co/collections/Soul-AILab/sac-68f1df9572a6314d1dc1f91e">
      <img src="https://img.shields.io/badge/🤗%20SAC-Models-yellow" alt="Hugging Face">
    </a>
  </p>

  <p align="center">
    <i>A semantic–acoustic dual-stream speech codec achieving state-of-the-art performance in speech reconstruction and semantic representation across bitrates.</i>
  </p>
</div>


## 🛠️ Environment Setup
```bash
conda create -n sac python=3.10
conda activate sac
pip install -r requirements.txt  # pip version == 24.0
```


## 🧩 Model Checkpoints

To use SAC, you need to prepare the pretrained dependencies, including the [GLM-4-Voice-Tokenizer](https://huggingface.co/zai-org/glm-4-voice-tokenizer) for semantic tokenization and the [ERes2Net](https://modelscope.cn/models/iic/speech_eres2net_sv_en_voxceleb_16k) speaker encoder for speaker feature extraction (during codec training). Make sure the corresponding model paths are correctly set in your configuration file (e.g., `configs/xxx.yaml`).

The following table lists the available SAC checkpoints:

| Model Name | Hugging Face | Sample Rate | Token Rate | BPS |
|:-----------:|:------------:|:------------:|:-----------:|:---:|
| SAC | [🤗 Soul-AILab/SAC-16k-37_5Hz](https://huggingface.co/Soul-AILab/SAC-16k-37_5Hz) | 16 kHz | 37.5 Hz | 525 |
| SAC | [🤗 Soul-AILab/SAC-16k-62_5Hz](https://huggingface.co/Soul-AILab/SAC-16k-62_5Hz) | 16 kHz | 62.5 Hz | 875 |


## 🎧 Inference

To perform audio reconstruction, you can use the following command:

```bash
python -m bins.infer
```

We also provide batch scripts for [audio reconstruction](./scripts/batch/reconstruct.sh), [encoding](./scripts/batch/encode.sh), [decoding](./scripts/batch/decode.sh), and [embedding extraction](./scripts/batch/extract_embeddings.sh) in the `scripts/batch` directory as references (you can refer to the [batch scripts guide](./docs/batch_scripts_guide.md) for details).


## 🧪 Evaluation

You can run the following command to perform evaluation:

```bash
bash scripts/eval.sh
```

For details on dataset preparation and evaluation setup, please first refer to the [evaluation guide](./docs/evaluation_guide.md).


## 🚀 Training
### Step 1: Prepare training data
Before training, organize your dataset in **JSONL** format. You can refer to `example/training_data.jsonl`. Each entry should include:
- **utt** — unique utterance ID (customizable)
- **wav_path** — path to raw audio
- **ssl_path** — path to offline-extracted Whisper features (for semantic supervision)
- **semantic_token_path** — path to offline-extracted semantic tokens

To accelerate training, you need to **extract semantic tokens and Whisper features offline** first before starting.  Refer to the [feature extraction guide](./docs/feature_extraction_guide.md) for detailed instructions.

### Step 2: Modify configuration files
You can adjust training and DeepSpeed configurations by editing:
- [`configs/xxx.yaml`](./configs) — main training configuration  
- [`configs/ds_stage2.json`](./configs/ds_stage2.json) — DeepSpeed configuration

### Step 3: Start training
Run the following script to start SAC training:

```bash
bash scripts/train.sh
```


## 🙏 Acknowledgement
Our codebase builds upon the awesome [SparkVox](https://github.com/SparkAudio/SparkVox) and [DAC](https://github.com/descriptinc/descript-audio-codec). We thank the authors for their excellent work.

## 🔖 Citation
If you find this work useful in your research, please consider citing:
```bibtex
@article{chen2025sac,
  title={SAC: Neural Speech Codec with Semantic-Acoustic Dual-Stream Quantization},
  author={Chen, Wenxi and Wang, Xinsheng and Yan, Ruiqi and Chen, Yushen and Niu, Zhikang and Ma, Ziyang and Li, Xiquan and Liang, Yuzhe and Wen, Hanlin and Yin, Shunshun and others},
  journal={arXiv preprint arXiv:2510.16841},
  year={2025}
}
```

## 📜 License
This project is licensed under the Apache 2.0 License.
