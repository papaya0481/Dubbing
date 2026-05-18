# Dubbing

Research code for AI video dubbing. The project studies how to generate or adapt speech so that the produced audio follows target timing, phoneme durations, and optional lip-motion cues.

At a high level, the repository combines three ideas:

1. **Alignment**: Montreal Forced Aligner (MFA) produces phoneme-level TextGrid files for speech.
2. **Timing conversion**: mel spectrograms or semantic features are stretched from one timing pattern to another.
3. **Generation**: Conditioned Flow Matching (CFM) models learn to turn those conditions into clean target mel spectrograms, which are then vocoded back to waveform audio.

## Overview

The project has three main components:

| Directory | Purpose |
|---|---|
| **[dubbing/](dubbing/README.md)** | Main training and inference package: experiment dispatch, configs, datasets, CFM models, MFA helpers, and tests |
| **[index-tts2/](index-tts2/README.md)** | Vendored reference implementation of IndexTTS2. `dubbing/modules/cfm_index/` is a standalone port of its `s2mel` CFM component |
| **[mel_convert/](mel_convert/README.md)** | Data preparation scripts for generating and filtering paired original/time-stretched audio or feature data |

## Quick start: environment

```bash
conda create -n dubbing python=3.11 -y
conda activate dubbing
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
conda install -c conda-forge montreal-forced-aligner kaldi=*=*cpu*
```

For CUDA 11.8 systems, the older PyTorch 2.7.1 wheel line used by this repo is:

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

## Model architecture (high level)

Three experiment tracks are dispatched by `config.model_name` in `dubbing/run.py`:

| Model | Description |
|---|---|
| **LipSyncCFM** | Original DiT-based flow matching path: phoneme ids + time-stretched mel -> clean mel |
| **CFM_Index** | IndexTTS2-style path: semantic condition + speaker style + reference prompt mel -> target mel |
| **CFM_Index_Lips** | `CFM_Index` plus cross-attention to lip hidden states for video-aware conditioning |

All three use CFM training and ODE-style inference. The `CFM_Index` variants are designed to load the CFM submodule from IndexTTS2's `s2mel.pth` checkpoint with `strict=False`.

## Training

```bash
# Original LipSyncCFM
python dubbing/run.py --config dubbing/configs/default_cfm.yaml

# IndexTTS2-style CFM (multi-GPU)
accelerate launch --multi_gpu --num_processes=2 dubbing/run.py \
    --config dubbing/configs/default_cfm_index.yaml

# CFM with lips cross-attention
python dubbing/run.py --config dubbing/configs/default_cfm_index_phase1_lips.yaml
```

Override any config key via CLI:

```bash
python dubbing/run.py --config dubbing/configs/default_cfm.yaml \
    data.root=/my/path training.learning_rate=5e-4 system.gpu=1
```

## Running tests

```bash
TEST_GPU=0 conda run -n dubbing python -m pytest dubbing/tests/ -xvs
```

## Data pipeline

1. **Prepare aligned data**: `mel_convert/` and MFA scripts produce audio pairs, TextGrid files, mel spectrograms, semantic features, and metadata.
2. **Load a training split**: `dubbing/data_provider/` maps `config.data.dataset` to the correct dataset and collate function.
3. **Train a CFM model**: the model learns a vector field that maps noise/conditions toward the target clean mel.
4. **Infer audio**: the trained model generates mel frames; BigVGAN converts those frames into waveform output.

## Key dependencies

- **Montreal Forced Aligner (MFA)** — phoneme-level alignment for all audio
- **IndexTTS2** — reference zero-shot TTS; pretrained GPT, s2mel CFM, BigVGAN vocoder
- **BigVGAN** — neural vocoder for mel → waveform conversion
- **HuggingFace Accelerate** — multi-GPU training
- **CAMPPlus** — speaker embedding extraction (192-dim)
- **W2v-BERT** — semantic feature extraction (1024-dim)

## Package READMEs

- **[dubbing/](dubbing/README.md)** — package overview, experiment dispatch, config system
- **[index-tts2/](index-tts2/README.md)** — reference TTS engine, quickstart, Python API
- **[mel_convert/](mel_convert/README.md)** — data generation pipeline
- **[CLAUDE.md](CLAUDE.md)** — assistant-oriented development notes

## Datasets

The released datasets are hosted on Hugging Face:

- [MELD raw dataset](https://huggingface.co/datasets/BigfufuOuO/chem_raw). `raw` keeps the original sample content but rewrites the directory layout to fit this project's pipeline.
- [MELD clips dataset](https://huggingface.co/datasets/BigfufuOuO/meld_clips_v3). `clips` contains concatenated and cleaned samples that can be used directly as training sources.
- [Chem raw dataset](https://huggingface.co/datasets/BigfufuOuO/chem_raw).
- **Chem clips dataset**. Not implemented; it may not be necessary because the original dataset has limited emotion diversity.
- [V2C raw dataset](https://huggingface.co/datasets/BigfufuOuO/V2C_raw).
- [V2C clips dataset](https://huggingface.co/datasets/BigfufuOuO/V2C_clips_v3).

> **Warning**
> The datasets **V2C raw**, **V2C clips**, and **MELD clips** are not publicly available yet. The Hugging Face datasets are private and require access approval.

### Make datasets

Dataset-making scripts live under [`dataset/`](dataset/README.md). They first convert each source corpus into the project raw layout:

```text
<dataset_root>/
|-- metadata.csv
|-- videos/
`-- audios/
    `-- ost/
```

`metadata.csv` should contain the transcript (`Utterance` or `Text`) and audio/video paths. This layout is the input for later steps such as vocal separation, MFA alignment, semantic generation, and CFM training.

Common entry points:

```bash
# MELD raw: copy MELD utterance clips, extract 16 kHz wav, write metadata.csv
bash dataset/V2C/MELD/make_raw.sh

# MELD clips: build 2-3 utterance emotion-change samples, then cut Friends episodes
cd dataset/V2C/MELD
python extract_samples.py --input train_sent_emo.csv --output train.csv --prefer-longer
bash make_clips.sh

# V2C raw and V2C clips
cd dataset/V2C
bash make_origin.sh
bash make.sh

# CHEM raw: merge pre-clipped CHEM videos into the same raw layout
cd dataset/V2C/chem
python merge_to_raw.py --input-dir /path/to/chem_processed/videos --output-dir /path/to/chem_raw
```

After a raw dataset exists, run optional alignment and flow-data generation:

```bash
# Generate TextGrid alignments under audios/aligned/
python dataset/mfa_align_enhanced.py /path/to/raw_dataset --clean --single-speaker

# Generate semantic CFM metadata and features consumed by cfm_index_phase1
bash mel_convert/generate/gen_semantic.sh
```

See [`dataset/README.md`](dataset/README.md) for required input files, output structures, and script-specific notes.
