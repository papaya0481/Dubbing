# mel_convert/generate/

Generation scripts for building flow-training metadata and generated audio/features. These scripts mainly wrap IndexTTS2 inference and MELD-style CSV processing; they are the upstream stage that creates files later consumed by `dubbing/data_provider/`.

## Files

| File | Purpose |
|---|---|
| `gen.py` | Main generation script for MELD-style CSV rows: normalizes text/emotion fields, builds emotion vectors, and writes generated outputs/metadata |
| `gen_once.py` | Single-shot generation helper |
| `gen_semantic.py` | Multi-process IndexTTS2 generation that writes wav/pt files and metadata for semantic CFM training |
| `gen_semantic_stretch.py` | Semantic-stretch enhanced generation path |
| `filter_metadata.py` | Filter and clean metadata CSVs for generated pairs |
| `extract_meld_audio.sh` | Extract audio segments from MELD dataset |
| `gen.sh` | Shell wrapper for batch generation |
| `gen_semantic.sh` | Shell wrapper for semantic data generation |

## Usage

```bash
bash mel_convert/generate/gen.sh
bash mel_convert/generate/gen_semantic.sh
```

## Pipeline overview

1. **Audio extraction** — `extract_meld_audio.sh` extracts clips from the MELD video dataset
2. **Generation** — `gen.py` / `gen_semantic.py` synthesizes wav files and saves the metadata needed by later CFM datasets
3. **Metadata filtering** — `filter_metadata.py` cleans generated metadata before training

The shell wrappers contain local dataset/checkpoint paths. Treat them as examples unless your filesystem matches the original environment.
