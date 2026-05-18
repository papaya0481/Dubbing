# Dubbing

Research code for AI video dubbing. The project studies how to generate or adapt speech so that the produced audio follows target timing, phoneme durations, and optional lip-motion cues.

At a high level, the repository combines three ideas:

1. **Alignment**: Montreal Forced Aligner (MFA) produces phoneme-level TextGrid files for speech.
2. **Timing conversion**: mel spectrograms or semantic features are stretched from one timing pattern to another.
3. **Generation**: continuous flow matching (CFM) models learn to turn those conditions into clean target mel spectrograms, which are then vocoded back to waveform audio.

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
