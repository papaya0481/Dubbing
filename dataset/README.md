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

Use this when starting from the official MELD download. The expected scratch format is the unpacked `MELD.Raw` directory plus MELD split CSVs. This repo already keeps the split CSVs under `dataset/V2C/MELD/`.

```text
MELD.Raw/
|-- dev/dev_splits_complete/dia*_utt*.mp4
|-- train/train_splits/dia*_utt*.mp4
`-- test/output_repeated_splits_test/dia*_utt*.mp4

dataset/V2C/MELD/
|-- dev_sent_emo.csv
|-- train_sent_emo.csv
`-- test_sent_emo.csv
```

```bash
python dataset/V2C/MELD/process_meld_raw.py \
    --meld-raw /path/to/MELD.Raw \
    --csv-dir dataset/V2C/MELD \
    --output-dir /path/to/MELD_raw \
    --model large-v3 \
    --language en
```

It writes `videos/`, `audios/ost/`, and `metadata.csv`. Each row includes split, dialogue id, utterance id, transcript, speaker, emotion, Whisper ASR text, and WER.

The wrapper is:

```bash
bash dataset/V2C/MELD/make_raw.sh
```

Edit the output path inside the wrapper before using it.

## MELD Clips

MELD clips are multi-utterance samples built from consecutive utterances by the same speaker. The builder keeps length-2 or length-3 windows that contain at least one emotion change.

Starting from scratch, you need:

- MELD `*_sent_emo.csv` files with `Season`, `Episode`, `Dialogue_ID`, `Utterance_ID`, `Speaker`, `Emotion`, `Utterance`, `StartTime`, and `EndTime`.
- Full Friends episode videos.
- An episode map CSV that tells the script where each episode video lives.

Expected episode map format:

```csv
Season,Episode,Filepath
3,12,season_03/episode_12.mp4
8,21,/absolute/path/to/S08E21.mp4
```

Relative `Filepath` values are resolved against `--map-root`.

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

The output layout is the same raw contract:

```text
MELD_clips/
|-- metadata.csv
|-- videos/
`-- audios/ost/
```

## V2C Raw

V2C raw clips are cut one subtitle row at a time from movie files.

Starting from scratch, download or prepare:

- The V2C animation movie videos.
- SRT subtitle files named by movie, for example `Brave.srt`.
- V2C annotation JSON files: `movie_speaker_id.json` and `emotions.json`.

The helper `V2C/DataConstruction/build_speaker_emotion_csv.py` converts the JSON + SRT files into `movie_speaker_emotion.csv`:

```bash
cd dataset/V2C
python V2C/DataConstruction/build_speaker_emotion_csv.py \
    --movie-speaker-json V2C/DataConstruction/movie_speaker_id.json \
    --emotions-json V2C/DataConstruction/emotions.json \
    --srt-dir /path/to/v2c_srt_files \
    --output V2C/DataConstruction/movie_speaker_emotion.csv
```

Expected `movie_speaker_emotion.csv` format:

```csv
movie,speaker,utterance,srt_index,emotion,emotion_id,start_time,end_time
Brave,LordDingwall,Men: Dingwall!,0280,neutral,4,"00:19:46,602","00:19:48,593"
```

You also need a `movie_video_map.csv` that points each movie name to the downloaded movie file and records the checked timing offset:

```csv
movie,filename,time_offset,srt_file,checked
Brave,Brave.mp4,0.0,Brave.srt,True
```

`filename` is resolved relative to `--video-root` unless it is absolute. Set `checked=True` only after confirming the subtitle/video offset. Use `--movie <name>` for that one-movie timing check before running the full batch.

Example:

```bash
cd dataset/V2C

# First check one movie and tune its time_offset in movie_video_map.csv.
python build_movie_origin.py \
    --movie-map V2C/DataConstruction/movie_video_map.csv \
    --utterances V2C/DataConstruction/movie_speaker_emotion.csv \
    --video-root /path/to/v2c_movies \
    --output-root /path/to/v2c_origin_debug \
    --movie Brave

# Then run the checked movies.
python build_movie_origin.py \
    --movie-map V2C/DataConstruction/movie_video_map.csv \
    --utterances V2C/DataConstruction/movie_speaker_emotion.csv \
    --video-root /path/to/v2c_movies \
    --output-root /path/to/v2c_origin
```

The wrapper is:

```bash
bash dataset/V2C/make_origin.sh
```

## V2C Clips

V2C clips combine consecutive utterances from the same speaker. Samples must contain an emotion change. This uses the same scratch inputs as V2C raw:

- `movie_video_map.csv`
- `movie_speaker_emotion.csv`
- downloaded movie videos under `--video-root`

If you do not already have `movie_speaker_emotion.csv`, create it with `V2C/DataConstruction/build_speaker_emotion_csv.py` as shown in the V2C raw section.

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

There are two supported scratch routes.

### Route A: long raw videos -> clipped CHEM videos -> CHEM raw

Use this when the downloaded CHEM data is long videos. The expected scratch layout is any directory tree containing `.mp4` files. Optional clean vocal or instrumental tracks can be placed beside each video:

```text
chem_download/
`-- show_or_movie_name/
    |-- episode01.mp4
    |-- episode01.wav                  # optional raw audio
    |-- vocals/episode01.wav           # optional clean vocal track
    `-- instrumental/episode01.wav     # optional instrumental track
```

First run the VideoClipper stage to produce the intermediate directory that later appears as `/path/to/chem_processed/videos`. From the project root:

```bash
cd dataset/seperate/video_clip

# Stage 1: ASR/VAD/sentence state.
python videoclipper_v2.py \
    --stage 1 \
    --file /path/to/chem_download \
    --output_dir /path/to/chem_processed/videos \
    --lang en \
    --device cuda \
    --skip_processed

# Stage 2: cut sentence-level clips and write metadata.csv + clip_wer.csv.
python videoclipper_v2.py \
    --stage 2 \
    --file /path/to/chem_download \
    --output_dir /path/to/chem_processed/videos \
    --lang en \
    --device cpu \
    --skip_processed
```

For paired adjacent-sentence clips, keep the stage-1 state and final paired output in separate directories:

```bash
cd dataset/seperate/video_clip

python videoclipper_v2.py \
    --stage 1 \
    --file /path/to/chem_download \
    --output_dir /path/to/chem_stage1 \
    --lang en \
    --device cuda \
    --skip_processed

python videoclipper_v2_pair.py \
    --file /path/to/chem_download \
    --state_dir /path/to/chem_stage1 \
    --output_dir /path/to/chem_processed/videos \
    --lang en \
    --device cuda \
    --pause_threshold 1.0 \
    --yes
```

The intermediate directory consumed by `merge_to_raw.py` should look like this:

```text
chem_processed/videos/
`-- <source_folder_or_video>/<video_id>/
    |-- metadata.csv
    |-- clipped/
    |   |-- *.mp4
    |   |-- *.wav
    |   `-- *.srt
    |-- vocals/             # optional, when vocal tracks were available
    |-- instrumental/       # optional
    `-- clip_wer.csv        # written by VideoClipper
```

Then return to the project root and merge the clipped CHEM folders into the project raw layout:

```bash
cd dataset/V2C/chem
python merge_to_raw.py \
    --input-dir /path/to/chem_processed/videos \
    --output-dir /path/to/chem_raw \
    --split chem \
    --gpu-id 0 \
    --workers 4
```

`merge_to_raw.py` copies clips into the raw layout, normalizes utterances when WeTextProcessing is available, filters rows containing `+` or `-`, filters videos where face detection fails, and writes:

```text
chem_raw/
|-- metadata.csv
|-- no_normed.csv
|-- wer_stats.txt
|-- videos/
`-- audios/ost/
```

### Route B: already clipped videos -> CHEM raw

Use this when the downloaded CHEM data is already segmented into clips and does not need VideoClipper. `pipeline.py` accepts any of these scratch layouts:

```text
chem_clips_download/
|-- videos/*.mp4

chem_clips_download/
|-- <video_id>/cut-*.mp4

chem_clips_download/
`-- *.mp4
```

Run:

```bash
cd dataset/V2C/chem
python pipeline.py \
    --intervals-dir /path/to/chem_clips_download \
    --output-dir /path/to/chem_raw \
    --whisper-model large \
    --language en
```

This route treats each input clip as one sample, extracts 16 kHz mono audio, transcribes it with Whisper, copies videos into `videos/`, and writes `metadata.csv`. The transcript column is named `Transcription`; if you feed this output into a step that specifically requires `Utterance`, copy or rename `Transcription` to `Utterance` first.

## Long-Video Pipeline

`dataset/seperate/` is a vendored FunCineForge-style pipeline for building datasets from long videos. It can normalize and trim raw videos, separate vocals, segment videos, run speaker diarization, apply multimodal correction, and build final metadata. Start from [`seperate/README.md`](seperate/README.md) when using this path.

Starting from scratch, place downloaded videos in language-specific roots. The pipeline expects one directory per film/show, with episode videos inside. Optional `.wav`, `vocals/`, and `instrumental/` files can be present if you already have them; otherwise the pipeline can extract audio and run separation.

```text
datasets/raw_zh/
`-- film_name/
    |-- 01.mp4
    |-- 02.mp4
    |-- vocals/01.wav          # optional
    `-- instrumental/01.wav    # optional

datasets/raw_en/
`-- show_name/
    `-- episode01.mp4
```

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
