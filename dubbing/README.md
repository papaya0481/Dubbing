# dubbing/

Main Python package for Dubbing. It contains the experiment entry point, config loader, data pipeline, CFM model implementations, MFA helpers, and tests.

Start here if you want to train or evaluate a model. The sibling directories are supporting pieces: `index-tts2/` is the reference TTS implementation, and `mel_convert/` prepares training data.

## Package structure

```
dubbing/
├── run.py               # Entry point — experiment dispatch
├── config.py             # YAML config loader with extends inheritance + CLI overrides
├── logger.py             # Rich-based logging with colored console output
├── configs/              # YAML config files (see configs/README.md)
├── data_provider/        # Dataset classes + collate functions + DataLoader factory (see data_provider/README.md)
├── exp/                  # Experiment classes (training, validation, inference loops)
├── modules/              # Model implementations (see modules/README.md)
├── tests/                # Test suite (see tests/README.md)
├── scripts/              # Shell scripts for launching training runs
├── montreal_forced_aligner/  # Vendored MFA helper modules used by the aligner path
└── archived/             # Deprecated/archived code
```

## Entry point: `run.py`

```
python dubbing/run.py --config dubbing/configs/default_cfm.yaml [key=value overrides...]
```

The script:
1. Loads the YAML config (with `extends` inheritance)
2. Applies CLI overrides (e.g. `training.learning_rate=5e-4 system.gpu=1`)
3. Sets random seeds, configures logging, sets up GPU
4. Dispatches to the experiment class via `EXP_MAP[config.model_name]`
5. If `is_training` is true, calls `exp.train(setting)` and then `exp.test(setting)` for each iteration; otherwise it runs only `exp.test(setting)`

### Experiment → config mapping

| `model_name` | Experiment class in `exp/cfm/` | Description |
|---|---|---|
| `LipSyncCFM` | `phase1_train_expand.py` | Phoneme-ids + stretched_mel → clean_mel via DiT CFM |
| `CFM_Index` | `cfm_index_phase1_train_expand.py` | IndexTTS2-style: semantic cond + style → mel, prompt-conditioned |
| `CFM_Index_Lips` | `cfm_index_phase1_lips.py` | Above + cross-attention to lips features |

### Experiment base class (`exp/basic.py`)

`Exp_Basic` provides:
- Device acquisition for the base single-GPU/CPU path. The IndexTTS2-style experiments override this with HuggingFace Accelerate.
- Shared checkpoint/config helpers (`_save_state`, `_save_args`, `best.pth` in experiment subclasses)
- LR scheduler construction (`_build_scheduler`: linear or cosine)
- Training log serialization (`_save_training_log`)

## Config system (`config.py`)

```python
cfg = load_config("dubbing/configs/default_cfm.yaml")
apply_overrides(cfg, ["training.learning_rate=5e-4", "system.gpu=1"])
```

- YAML → nested `SimpleNamespace` (dot-accessible: `cfg.training.learning_rate`)
- `extends` key for YAML inheritance (resolved relative to the child file)
- CLI overrides auto-parse types: bool, int, float, None, str
- Serialize back to dict with `config_to_dict(cfg)`

## Scripts

The recommended commands are the explicit `python`/`accelerate` invocations in the root README. Some older shell wrappers still contain stale config names or old CLI flags, so check the script before using it as a source of truth.

| Script | Current status |
|---|---|
| `scripts/train_cfm_index_phase1.sh` | Current multi-GPU Accelerate wrapper for `default_cfm_index.yaml` |
| `scripts/train_cfm_index_phase1_lips.sh` | Current multi-GPU Accelerate wrapper for `default_cfm_index_phase1_lips.yaml` |
| `scripts/train_cfm_phase1.sh` | Stale: points to `default.yaml`, which is not present in the current tree |
| `scripts/train_cfm_phase1_from_mel.sh` | Stale: uses old CLI flags instead of `--config` plus dotted overrides |

Run scripts from the repository root.

## Sub-packages with their own READMEs

- **[configs/](configs/README.md)** — YAML configuration files and inheritance mechanism
- **[data_provider/](data_provider/README.md)** — Dataset classes, collate functions, DataLoader factory
- **[modules/](modules/README.md)** — CFM models (cfm, cfm_index), mel warping (mel_strech), lips alignment (lips), semantic stretch, MFA aligner
- **[tests/](tests/README.md)** — Test suite, GPU conventions (`TEST_GPU` env var), key test modules
