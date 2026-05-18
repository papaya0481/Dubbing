# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment & setup

```bash
conda create -n dubbing python=3.11 -y && conda activate dubbing
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
conda install -c conda-forge montreal-forced-aligner
conda install -c conda-forge kaldi=*=*cpu*
```

## Key commands

### Training

Three experiment types, run from repo root:

```bash
# Original LipSyncCFM (phoneme-conditioned flow matching)
python dubbing/run.py --config dubbing/configs/default_cfm.yaml

# IndexTTS2-style CFM (discrete semantic tokens → mel)
accelerate launch --multi_gpu --num_processes=2 dubbing/run.py \
    --config dubbing/configs/default_cfm_index.yaml

# CFM with lips cross-attention
python dubbing/run.py --config dubbing/configs/default_cfm_index_phase1_lips.yaml
```

Override any config key via CLI: `data.root=/my/path training.learning_rate=5e-4 system.gpu=1`

### Tests

```bash
# Run all tests (requires GPU; set TEST_GPU env var to choose device)
TEST_GPU=0 conda run -n dubbing python -m pytest dubbing/tests/ -xvs

# Run a single test module
conda run -n dubbing python -m dubbing.tests.test_cfm_index
conda run -n dubbing python -m dubbing.tests.test_semantic_warp_correctness

# Run a single test function
TEST_GPU=0 conda run -n dubbing python -m pytest dubbing/tests/test_cfm_index.py::test_cfm_batch_inference -xvs
```

- `TEST_GPU` env var controls which GPU (defaults to `1`); set in `dubbing/tests/conftest.py`.
- Tests use `weights_only=False` in `torch.load` — do not change this, as checkpoints contain config namespaces.

## Architecture

### Experiment dispatch (`dubbing/run.py`)

The entry point selects an experiment class via `config.model_name` → `EXP_MAP`:

| model_name | Exp class | Purpose |
|---|---|---|
| `LipSyncCFM` | `Exp_CFM_Phase1_TrainExpand` | Original CFM: phoneme-ids + stretched_mel → clean_mel |
| `CFM_Index` | `Exp_CFM_Index_Phase1_TrainExpand` | IndexTTS2-style: semantic cond + style → mel (prompt-conditioned) |
| `CFM_Index_Lips` | `Exp_CFM_Index_Phase1_Lips` | Same as above + cross-attention to lips features |

All experiment classes inherit from `Exp_Basic` (`dubbing/exp/basic.py`), which handles device acquisition, checkpoint save/load, and scheduler construction.

### Config system (`dubbing/config.py`)

Nested `SimpleNamespace` loaded from YAML. Supports:
- `extends` key for YAML inheritance (resolved relative to the child file).
- CLI overrides: dotted keys like `data.root=/path training.learning_rate=5e-4`.
- Auto-type casting for CLI values (bool, int, float, None).

### Data pipeline

1. **Dataset classes** in `dubbing/data_provider/data_loader.py`, keyed by `config.data.dataset`:
   - `cfm_phase1` / `cfm_phase1_stretch`: pairs of mel spectrograms with phoneme alignments (TextGrid-based).
   - `cfm_index_phase1`: loads pre-extracted `.pt` files with semantic features, reference mels, and style vectors.
   - `cfm_index_phase1_for_lipsfeat`: adds lips hidden states aligned to source phonemes.

2. **Data factory** (`dubbing/data_provider/data_factory.py`) maps dataset name → Dataset class + collate function.

3. **Collate functions** (`dubbing/data_provider/collate_funcs.py`) handle zero-padding and tensor assembly for variable-length sequences.

4. Val/test always use `batch_size=1`.

### Modules (`dubbing/modules/`)

- **`cfm/`** — Original LipSyncCFM model: DiT backbone + flow matching loss. Forward pass: `clean_mel, stretched_mel, phoneme_ids, x_lens`.
- **`cfm_index/`** — Port of IndexTTS2's CFM. `CFM` class uses DiT + WaveNet final layer. `CrossAttnCFM` extends with lips cross-attention. Forward: `x1_full, cond, style, x_lens`.
- **`mel_strech/`** — Mel warping tools. `GlobalWarpTransformer` handles mel stretching via textgrid-driven interpolation, optionally wrapping BigVGAN vocoder for mel→wav conversion.
- **`lips/`** — Lip-reading models (Conformer-based) for phoneme-lips alignment, with training scripts and data processing for MELD/LRS2 datasets.
- **`semantic_stretch/`** — `semantic_transform.py` (620 lines): semantic-based audio warping using wav2vec feature alignment.
- **`mfa_alinger.py`** — Wraps Montreal Forced Aligner for per-utterance phoneme alignment.

### Submodule: `index-tts2/`

Reference TTS implementation. Contains:
- `indextts/s2mel/` — Reference CFM model that `modules/cfm_index/` was ported from.
- `indextts/vqvae/` — EnCodec-based audio codec.
- `indextts/gpt/` — GPT for generating semantic token sequences.

The ported `modules/cfm_index/` is tested against this reference in `dubbing/tests/test_cfm_index.py` to verify output equivalence.
