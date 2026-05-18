# dubbing/configs/

YAML configuration files for Dubbing experiments. A config controls the dataset, model architecture, pretrained checkpoint paths, training hyperparameters, and runtime device settings used by `dubbing/run.py`.

## Config inheritance

Any config file can declare `extends: <other.yaml>` at the top level. The path is resolved relative to the child file. The child only needs to specify the deltas from the parent — the two are deep-merged. Chains are supported (A extends B extends C).

## CLI overrides

Any config key can be overridden at runtime via dotted path:

```bash
python dubbing/run.py --config dubbing/configs/default_cfm.yaml \
    training.learning_rate=5e-4 system.gpu=1
```

Values are auto-parsed: `true`/`false` → bool, `123` → int, `1e-4` → float, `null` → None.

## Files

| File | `model_name` | Current use |
|---|---|---|
| `default_cfm.yaml` | `LipSyncCFM` | Original phoneme-conditioned CFM with DiT backbone |
| `default_cfm_index.yaml` | `CFM_Index` | IndexTTS2-style CFM: discrete semantic tokens → mel, prompt-conditioned |
| `default_cfm_index_phase1_lips.yaml` | `CFM_Index_Lips` | Same as above + lips cross-attention |
| `stretch_entire_mel.yaml` | `LipSyncCFM_StretchEntireMel` | Legacy ablation config. It currently extends `default.yaml`, but that file is not present and this `model_name` is not registered in `EXP_MAP`; update it before use |

## Key config sections

- **`data`** — dataset name, paths, batch_size, num_workers, audio duration limits
- **`model`** — DiT / CFM architecture, style encoder dimensions, wavenet config
- **`preprocess`** — paths to frozen sub-models (w2v-bert, CAMPPlus, s2mel checkpoint), mel spectrogram params
- **`training`** — epochs, learning rate, scheduler, early stopping, inference steps
- **`system`** — seed, log level, checkpoint dir, GPU settings

## Runtime dispatch

`dubbing/run.py` only knows the model names registered in `EXP_MAP`: `LipSyncCFM`, `CFM_Index`, and `CFM_Index_Lips`. Adding a new config is not enough by itself; the `model_name` must either match one of those keys or the new experiment class must be added to `EXP_MAP`.
