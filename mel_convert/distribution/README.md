# mel_convert/distribution/

Scripts for analyzing and filtering paired audio datasets by mel spectrogram quality.

## Files

| File | Purpose |
|---|---|
| `filter_by_mse_and_copy.py` | Filter audio pairs by MSE between stretched and target mel, copy qualifying pairs |
| `word_alignment_mse_plot.py` | Plot MSE distributions per word alignment, used for quality analysis |
| `new.py` | Distribution analysis for new data splits |
| `distr.sh` | Shell wrapper to batch-run distribution analysis |
| `filter.sh` | Shell wrapper to batch-run MSE filtering |

## Usage

```bash
bash mel_convert/distribution/filter.sh
bash mel_convert/distribution/distr.sh
```

Both shell wrappers contain local absolute paths. For a new environment, either edit the scripts or call the Python files directly with your own `--audio-dir`, `--aligned-dir`, and `--manifest` paths.
