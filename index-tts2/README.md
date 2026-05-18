# index-tts2/

Vendored reference implementation of IndexTTS2, an emotionally expressive, duration-aware, autoregressive zero-shot TTS system from [github.com/index-tts/index-tts](https://github.com/index-tts/index-tts).

This directory is kept in this repository as reference code, not as the main training package. The Dubbing code uses it in two ways:

1. `dubbing/modules/cfm_index/` ports the IndexTTS2 `s2mel` CFM architecture into the local training framework.
2. `dubbing/tests/test_cfm_index.py` compares the local port against this reference implementation when the required checkpoints are available.

## Directory structure

```
index-tts2/
├── indextts/            # Core engine — models, inference, utilities (see indextts/README.md)
├── tests/               # Regression and padding tests (see tests/README.md)
├── tools/               # i18n utilities (see tools/README.md)
├── examples/            # Example input cases (see examples/README.md)
├── archive/             # Archived upstream docs
├── checkpoints/         # Model weights, created after download
├── pyproject.toml       # Python project config (uv-based)
├── setup.py             # Legacy setup script
├── webui.py             # Gradio web demo launcher
├── batch_gen2.py        # Local batch generation script
├── run_duration_second.py     # Duration-controlled generation demo
├── gen.sh / run.sh            # Shell launch wrappers
└── test.py                    # Quick test script
```

## Quickstart

1. Install dependencies:
   ```bash
   cd index-tts2
   uv sync
   ```

2. Download models:
   ```bash
   hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints
   ```

   ModelScope is also supported by the upstream project:
   ```bash
   modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
   ```

3. Run the IndexTTS2 smoke-test script:
   ```bash
   PYTHONPATH=$PYTHONPATH:. uv run python indextts/infer_v2.py
   ```

   `indextts/infer_v2.py` expects the checkpoint directory and example prompt audio to exist. Some example audio files may be supplied through Git LFS or downloaded model/example assets rather than normal source files.

4. Run the legacy CLI:
   ```bash
   PYTHONPATH=$PYTHONPATH:. uv run python indextts/cli.py "Hello world" -v examples/voice_01.wav -o output.wav
   ```

   This CLI imports `indextts.infer.IndexTTS` rather than `IndexTTS2`. Use `indextts/infer_v2.py` or the Python API below for IndexTTS2-specific emotion and duration controls.

5. Launch web demo:
   ```bash
   PYTHONPATH=$PYTHONPATH:. uv run webui.py
   ```

## Python API (IndexTTS2)

```python
from indextts.infer_v2 import IndexTTS2

tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints")
tts.infer(
    spk_audio_prompt="voice.wav",
    text="Hello, this is a test.",
    output_path="output.wav",
    emo_audio_prompt="emo_sad.wav",       # optional emotional reference
    emo_alpha=0.9,                         # emotion blend strength
    emo_vector=[0, 0, 0, 0, 0, 0, 0.45, 0],# or an 8-dim emotion vector
    use_emo_text=True,                     # or text-based emotion guidance
)
```

Emotion vector order follows the upstream IndexTTS2 convention: `Happy | Angry | Sad | Fear | Hate | Low | Surprise | Neutral`.

## Key components

| Component | Location | Purpose |
|---|---|---|
| GPT | `indextts/gpt/model_v2.py` | Text → semantic tokens (autoregressive, duration-controllable) |
| s2mel CFM | `indextts/s2mel/` | Semantic tokens + style + prompt mel → target mel |
| BigVGAN | `indextts/BigVGAN/` | Mel spectrogram → waveform vocoder |
| VQ-VAE | `indextts/vqvae/` | Audio codec for discrete representation |
| CAMPPlus | `indextts/s2mel/modules/campplus/` | Speaker embedding extraction |
| W2v-BERT | `indextts/s2mel/wav2vecbert_extract.py` | Semantic feature extraction |
| Qwen3 | (Modelscope) | Emotion text description → emotion vector |

## Local scripts

| Script | Purpose |
|---|---|
| `batch_gen2.py` | Batch generation for ESD-style data and emotion vectors |
| `run_duration_second.py` | Duration-specified generation demo |
| `test.py` | Quick manual inference script |
| `webui.py` | Gradio demo |

Several upstream batch scripts were removed from this checkout. If older notes mention `batch_gen.py`, `batch_gen2_speech_input.py`, `batch_gen2_ablation_spc_input.py`, or `batch_gen2_timing.py`, treat those references as stale for the current tree.

## Subdirectories with their own READMEs

- **[indextts/](indextts/README.md)** — Core engine: GPT, s2mel, BigVGAN, vqvae, inference APIs
- **[tests/](tests/README.md)** — Regression and padding tests
- **[tools/](tools/README.md)** — i18n tools
- **[examples/](examples/README.md)** — Example input JSONL
- **[archive/](archive/README.md)** — Original upstream README material retained for reference
