# dubbing/data_provider/

Data loading pipeline for all experiment types. This package turns config values into `Dataset`, `collate_fn`, and `DataLoader` objects with the tensor shapes expected by each experiment class.

## Files

| File | Role |
|---|---|
| `data_loader.py` | Dataset classes: `Dataset_CFM_Phase1`, `Dataset_CFM_Phase1_StretchEntireMel`, `Dataset_CFM_Index_Phase1`, `Dataset_CFM_Index_Phase1_ForLipsFeat` |
| `collate_funcs.py` | Per-dataset collate functions: zero-padding and tensor assembly for variable-length sequences |
| `data_factory.py` | `data_provider(args, flag)` — maps `config.data.dataset` → Dataset + collate_fn + DataLoader |
| `utils.py` | `CFMIndexCacheBuilder` and `CFMIndexLipsInferCacheBuilder` — batched pre-computation of heavy frozen-model features (w2v-bert, CAMPPlus) to `.pt` cache files |

## Dataset overview

### cfm_phase1 / cfm_phase1_stretch

Legacy datasets for the original LipSyncCFM model. Discover paired (r1, r2) audio files under `data.root/`, load their mel spectrograms and MFA phoneme alignments from TextGrid files.

### `cfm_index_phase1`

Loads samples from a metadata CSV. Each sample has pre-extracted `.pt` files containing:
- `S_infer` — generated semantic representation from IndexTTS2
- `ref_mel` — reference mel spectrogram (prompt conditioning)
- `style` — CAMPPlus speaker embedding (192-dim)
- `x1_mel` — target mel spectrogram (training target)

Heavy frozen-model inference, including w2v-BERT/length-regulator conditions and CAMPPlus speaker embeddings, is cached per sample by `CFMIndexCacheBuilder`.

### cfm_index_phase1_for_lipsfeat

Extends the above with lips hidden states aligned to source phonemes, loaded from `flow_dataset` `.pt` files. Additional fields: `lips_hidden_states`, `lips_textgrid`, `source_textgrid`.

## Collate conventions

- Val/test always use `batch_size=1`, `shuffle=False`, `drop_last=False`.
- Train uses configurable `batch_size` with `shuffle=True`.
- All collate functions zero-pad to max_len within the batch.
