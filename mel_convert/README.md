# mel_convert/

Data preparation utilities for the Dubbing experiments. This directory is used before model training to create aligned audio/feature metadata, generate IndexTTS2 outputs, stretch mel or semantic features, and filter low-quality pairs.

Most scripts are research pipeline scripts rather than polished command-line tools. Check the paths inside each shell script before running; several defaults point to local `/data2/ruixin/...` datasets.

## Directory structure

```
mel_convert/
├── meldataset.py        # Mel spectrogram extraction utilities (shared with dubbing/modules/)
├── transform.py         # Audio/mel transformation functions
├── visual.py            # Mel spectrogram visualization with TextGrid overlay
├── mfa_config.yaml      # Montreal Forced Aligner configuration
├── align.sh             # Batch MFA alignment script
├── codes2phonemes/      # Code-to-phoneme conversion models (see codes2phonemes/README.md)
├── distribution/        # Data filtering and quality analysis scripts (see distribution/README.md)
└── generate/            # Pair generation pipeline scripts (see generate/README.md)
```

## Data pipeline overview

1. **Generate or collect audio/features** (`generate/`): create IndexTTS2-generated wav/pt files and metadata from MELD-style CSV inputs.
2. **Align** (`align.sh`, `mfa_config.yaml`): run Montreal Forced Aligner to produce phoneme-level TextGrid files.
3. **Transform timing** (`transform.py`, semantic stretch scripts): warp mel spectrograms or semantic features from source timing to target timing.
4. **Filter by quality** (`distribution/`): compute and visualize MSE-based quality metrics, then copy or keep only qualifying pairs.

## Key scripts

### `visual.py`

Visualize mel spectrograms with overlaid TextGrid alignments. Used for debugging alignment and warp quality.

### `transform.py`

Core mel/audio transformation logic shared across the pipeline.

### `meldataset.py`

`get_mel_spectrogram()` extracts log-mel spectrograms from audio waveforms. Parameters such as `n_fft`, `hop_size`, `win_size`, `num_mels`, and `sampling_rate` come from the model/vocoder config object passed into the function.

## Subdirectories with their own READMEs

- **[codes2phonemes/](codes2phonemes/README.md)** — Models for converting discrete speech codes to phoneme sequences
- **[distribution/](distribution/README.md)** — Scripts for analyzing and filtering paired data by mel quality
- **[generate/](generate/README.md)** — Pipeline scripts for generating paired training data
