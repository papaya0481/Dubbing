# Dataset Preparation

This directory contains dataset conversion scripts for the Dubbing experiments. The scripts are research pipeline utilities, so most shell wrappers include local example paths such as `/data2/ruixin/...`. Edit those paths before running them.

## Target Layout

Most builders normalize a source corpus into this raw layout:

```text
<dataset_root>/
|-- metadata.csv
|-- videos/
|   `-- *.mp4
`-- audios/
    `-- ost/
        |-- *.wav
        `-- *.txt
```

`metadata.csv` is the contract between the dataset builders and the rest of the project. Common columns are:

| Column | Meaning |
|---|---|
| `Sample_ID` | Stable numeric row id |
| `Split` | Split name such as `train`, `dev`, `test`, `v2c`, or `chem` |
| `Audio_Filename` / `Audio_Path` | Audio clip file under `audios/ost/` |
| `Video_Filename` / `Video_Path` | Video clip file under `videos/` |
| `Utterance` | Transcript used for ASR checks and MFA alignment |
| `Speaker` | Speaker label when available |
| `Emotion` | Emotion label when available |
| `ASR_Text`, `WER` | Optional Whisper output and word error rate for quality checks |

The alignment helper also accepts alternate text/audio keys such as `Text`, `Transcript`, `audio`, `filename`, or `wav`, but the columns above are the preferred format.

## Directory Map

```text
dataset/
|-- V2C/
|   |-- README.md                 # detailed V2C movie clipping documentation
|   |-- build_movie_origin.py      # one subtitle row -> one raw clip
|   |-- build_movie_dataset.py     # sliding-window emotion-change clips
|   |-- merge_and_wer.py           # merge multiple V2C raw roots and compute WER
|   |-- make_origin.sh             # example V2C raw command
|   |-- make.sh                    # example V2C clips command
|   |-- MELD/                      # MELD raw and MELD clips builders
|   `-- chem/                      # CHEM raw conversion helpers
|-- seperate/                      # vendored FunCineForge-style long-video pipeline
|-- Mel-Band-Roformer-Vocal-Model/ # vocal separation model wrapper
`-- mfa_align_enhanced.py          # metadata-driven MFA alignment helper
```

## MELD Raw

Use this when you already have `MELD.Raw` with the official split folders and the MELD `*_sent_emo.csv` files.

```bash
python dataset/V2C/MELD/process_meld_raw.py \
    --meld-raw /path/to/MELD.Raw \
    --csv-dir dataset/V2C/MELD \
    --output-dir /path/to/MELD_raw \
    --model large-v3 \
    --language en
```

The script reads:

```text
MELD.Raw/
|-- dev/dev_splits_complete/*.mp4
|-- train/train_splits/*.mp4
`-- test/output_repeated_splits_test/*.mp4
```

It writes `videos/`, `audios/ost/`, and `metadata.csv`. Each row includes split, dialogue id, utterance id, transcript, speaker, emotion, Whisper ASR text, and WER.

The wrapper is:

```bash
bash dataset/V2C/MELD/make_raw.sh
```

Edit the output path inside the wrapper before using it.

## MELD Clips

MELD clips are multi-utterance samples built from consecutive utterances by the same speaker. The builder keeps length-2 or length-3 windows that contain at least one emotion change.

First generate sample CSVs:

```bash
cd dataset/V2C/MELD
python extract_samples.py --input train_sent_emo.csv --output train.csv --prefer-longer
python extract_samples.py --input dev_sent_emo.csv --output dev.csv --prefer-longer
python extract_samples.py --input test_sent_emo.csv --output test.csv --prefer-longer
```

Then cut clips from full Friends episodes:

```bash
python extract_clips.py \
    --episode-map /path/to/friends_episode_map.csv \
    --map-root /path/to/friends_root \
    --samples-dir . \
    --output-dir /path/to/MELD_clips \
    --asr-check
```

`friends_episode_map.csv` must contain `Season`, `Episode`, and `Filepath`. Relative `Filepath` values are resolved against `--map-root`.

The output layout is the same raw contract:

```text
MELD_clips/
|-- metadata.csv
|-- videos/
`-- audios/ost/
```

## V2C Raw

V2C raw clips are cut one subtitle row at a time from movie files.

Required inputs:

- `movie_video_map.csv`: columns `movie`, `filename`, `checked`, and optional `time_offset`
- `movie_speaker_emotion.csv`: columns such as `movie`, `speaker`, `utterance`, `emotion`, `emotion_id`, `start_time`, `end_time`, `srt_index`
- a video root containing the movie files named by `filename`

Example:

```bash
cd dataset/V2C
python build_movie_origin.py \
    --movie-map V2C/DataConstruction/movie_video_map.csv \
    --utterances V2C/DataConstruction/movie_speaker_emotion.csv \
    --video-root /path/to/v2c_movies \
    --output-root /path/to/v2c_origin
```

Use `--movie <name>` for offset debugging on one movie. After checking timing, set that movie's `checked` column to `True` and run the full batch.

The wrapper is:

```bash
bash dataset/V2C/make_origin.sh
```

## V2C Clips

V2C clips combine consecutive utterances from the same speaker. Samples must contain an emotion change. This is the direct V2C clips dataset.

```bash
cd dataset/V2C
python build_movie_dataset.py \
    --movie-map V2C/DataConstruction/movie_video_map.csv \
    --utterances V2C/DataConstruction/movie_speaker_emotion.csv \
    --video-root /path/to/v2c_movies \
    --output-root /path/to/v2c_clips \
    --max-gap 10.0 \
    --max-words 50
```

The wrapper is:

```bash
bash dataset/V2C/make.sh
```

For a full parameter description, see [`V2C/README.md`](V2C/README.md).

## Merge V2C Shards

Use `merge_and_wer.py` when raw V2C data is produced in multiple output roots.

```bash
cd dataset/V2C
python merge_and_wer.py \
    --src /path/to/v2c_origin /path/to/v2c_part2_origin \
    --dst /path/to/v2c_raw_all \
    --model large-v3 \
    --language en
```

Use `--skip-asr` to only merge files and metadata. The script writes a merged `metadata.csv`, copied audio/video files, and `wer_distribution.txt`.

## CHEM Raw

CHEM processing currently targets a raw-style dataset. A clips dataset is not implemented.

For already clipped CHEM videos, use:

```bash
cd dataset/V2C/chem
python merge_to_raw.py \
    --input-dir /path/to/chem_processed/videos \
    --output-dir /path/to/chem_raw \
    --split chem \
    --gpu-id 0 \
    --workers 4
```

Expected input:

```text
chem_processed/videos/
`-- <video_id>/
    |-- metadata.csv
    |-- clipped/
    |   |-- *.mp4
    |   `-- *.wav
    `-- clip_wer.csv or wer.csv   # optional
```

The script copies clips into the raw layout, normalizes utterances when WeTextProcessing is available, filters rows containing `+` or `-`, filters videos where face detection fails, and writes:

```text
chem_raw/
|-- metadata.csv
|-- no_normed.csv
|-- wer_stats.txt
|-- videos/
`-- audios/ost/
```

There is also a simpler pipeline for flat or nested input videos:

```bash
python pipeline.py \
    --intervals-dir /path/to/input_videos \
    --output-dir /path/to/chem_raw \
    --whisper-model large \
    --language en
```

## Long-Video Pipeline

`dataset/seperate/` is a vendored FunCineForge-style pipeline for building datasets from long videos. It can normalize and trim raw videos, separate vocals, segment videos, run speaker diarization, apply multimodal correction, and build final metadata. Start from [`seperate/README.md`](seperate/README.md) when using this path.

The high-level flow is:

```bash
cd dataset/seperate
python normalize_trim.py --root datasets/raw_zh --intro 10 --outro 10
cd speech_separation && python run.py --root ../datasets/clean/zh --gpus 0
cd ../video_clip && bash run.sh --stage 1 --stop_stage 2 --input ../datasets/raw_zh --output ../datasets/clean/zh --lang zh --device cpu
cd ../speaker_diarization && bash run.sh --stage 1 --stop_stage 4 --hf_access_token hf_xxx --root ../datasets/clean/zh --gpus "0"
cd .. && python build_datasets.py --root_zh datasets/clean/zh --root_en datasets/clean/en --out_dir datasets/clean --save
```

## Alignment

After creating a raw dataset, run MFA alignment if the training or evaluation path needs TextGrid files.

```bash
python dataset/mfa_align_enhanced.py /path/to/raw_dataset \
    --dictionary english_us_arpa \
    --acoustic-model english_us_arpa \
    --num-jobs 8 \
    --clean \
    --single-speaker
```

The script reads `metadata.csv`, gets transcripts from `Utterance` or another supported text column, prefers `audios/vocals/<name>_vocals.wav` when available, falls back to `audios/ost/<name>.wav`, and writes TextGrid files under:

```text
<dataset_root>/audios/aligned/
```

## Flow Dataset Generation

The CFM Index training path does not consume the raw dataset directly. It consumes semantic generation metadata with columns:

- `prompt_audio_path`
- `out_pt`
- `out_wav`
- `gen_error`

Generate this from a raw dataset with:

```bash
python mel_convert/generate/gen_semantic.py \
    --csv /path/to/raw_dataset/metadata.csv \
    --output-dir /path/to/flow_dataset/MELD/semantic \
    --model-dir /path/to/index-tts2/checkpoints \
    --gpus 0,1 \
    --num-process 2
```

The default training config then points to:

```yaml
data:
  dataset: cfm_index_phase1
  csv_path: /path/to/flow_dataset/MELD/semantic/metadata.csv
```

For lip-feature training, the configured root should contain semantic metadata and predicted lip features:

```yaml
data:
  dataset: cfm_index_phase1_for_lipsfeat
  flow_dataset_path: /path/to/flow_dataset/MELD
```

See [`mel_convert/generate/README.md`](../mel_convert/generate/README.md) for the generation scripts.

## Common Checks

- Confirm every `Audio_Path` and `Video_Path` in `metadata.csv` exists relative to the dataset root.
- Keep `audios/ost/*.wav` at 16 kHz mono PCM when possible; the builders use ffmpeg settings for this.
- Run a small subset first, especially when tuning `time_offset` for V2C movies.
- Use WER reports to inspect transcript/audio mismatch before alignment or training.
- Do not reuse the same output directory for raw and clips datasets; their `metadata.csv` files describe different sample units.
