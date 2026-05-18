# index-tts2/indextts/

Core IndexTTS engine code. For the Dubbing project, this directory is mainly a reference implementation for the `s2mel` CFM module that was ported into `dubbing/modules/cfm_index/`.

## Directory structure

### `s2mel/` — Semantic-to-mel generation

The CFM-based mel generator maps semantic conditions, speaker style, and prompt mel frames to target mel spectrograms.

| Path | Purpose |
|---|---|
| `s2mel/modules/` | DiT, BigVGAN vocoder, CAMPPlus speaker encoder, audio utilities |
| `s2mel/wav2vecbert_extract.py` | W2v-BERT feature extraction for semantic conditioning |
| `s2mel/hf_utils.py` | HuggingFace model download utilities |
| `s2mel/optimizers.py` | Optimizer configurations for training |

### `gpt/` — GPT-based text-to-semantic-token generation

Autoregressive transformer that converts text + speaker prompt → semantic token sequence.

| File | Purpose |
|---|---|
| `model.py` | `UnifiedVoice` — GPT model for IndexTTS1 |
| `model_v2.py` | `UnifiedVoice` — GPT model for IndexTTS2 (duration-controllable) |
| `conformer_encoder.py` | Conformer-based prompt encoder |
| `hmm.py` | HMM-based duration model |
| `perceiver.py` | Perceiver-based audio conditioning |
| `transformers_gpt2.py` | GPT-2 backbone implementation |
| `transformers_beam_search.py` | Beam search for autoregressive decoding |
| `transformers_generation_utils.py` | Generation utilities (top-k, top-p sampling) |
| `transformers_modeling_utils.py` | Base modeling utilities |

### `vqvae/` — Audio codec

| File | Purpose |
|---|---|
| `xtts_dvae.py` | XTTS-style discrete VAE for audio compression/reconstruction |

### `BigVGAN/` — Neural vocoder

NVIDIA BigVGAN vocoder for mel-spectrogram → waveform conversion. Includes ECAPA_TDNN speaker encoder and alias-free activation functions.

### Inference entry points

| File | Purpose |
|---|---|
| `infer.py` | `IndexTTS` class — IndexTTS1 inference (GPT → s2mel → BigVGAN) |
| `infer_v2.py` | `IndexTTS2` class — IndexTTS2 inference with emotional control, emo_alpha, emo_vector, and text-based emotion guidance |
| `cli.py` | Legacy command-line interface. It imports `IndexTTS` from `infer.py`, not `IndexTTS2` |

### `utils/` — Shared utilities

Text normalization, tokenization, feature extraction (mel spectrogram), checkpoint loading, maskgct utilities.

## Usage (Python API)

```python
from indextts.infer_v2 import IndexTTS2

tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints")
tts.infer(
    spk_audio_prompt="voice.wav",
    text="Hello, this is a test.",
    output_path="output.wav",
    emo_audio_prompt="emo_sad.wav",  # optional emotional reference
    emo_alpha=0.9,                    # emotion intensity
)
```

## Relationship to dubbing/

The model architecture defined here, especially DiT dimensions, WaveNet config, style embedding size, and content condition dimensions, is mirrored in `dubbing/configs/default_cfm_index.yaml`. The local `dubbing/modules/cfm_index/` module can be trained independently while loading the CFM part of the original `s2mel.pth` checkpoint with `strict=False`.
