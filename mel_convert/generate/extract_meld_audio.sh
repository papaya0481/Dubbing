#!/usr/bin/env bash

set -euo pipefail

# 用法:
#   bash extract_meld_audio.sh <meld_raw_root> <output_dir> [sample_rate]
# 示例:
#   bash extract_meld_audio.sh /data2/ruixin/downloads/MELD-RAW/MELD.Raw /data2/ruixin/dataset/MELD_AUDIO 16000

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: bash $0 <meld_raw_root> <output_dir> [sample_rate]"
  exit 1
fi

MELD_ROOT="$1"
OUT_DIR="$2"
SAMPLE_RATE="${3:-16000}"

if [[ ! -d "$MELD_ROOT" ]]; then
  echo "[Error] MELD root does not exist: $MELD_ROOT"
  exit 1
fi

if [[ ! -d "$MELD_ROOT/dev" && -d "$MELD_ROOT/MELD.Raw/dev" ]]; then
  MELD_ROOT="$MELD_ROOT/MELD.Raw"
fi

MELD_ROOT="$(realpath -m "$MELD_ROOT")"
OUT_DIR="$(realpath -m "$OUT_DIR")"

if [[ ! -d "$MELD_ROOT/dev" && ! -d "$MELD_ROOT/test" && ! -d "$MELD_ROOT/train" ]]; then
  echo "[Error] Invalid MELD root (expect dev/test/train under it): $MELD_ROOT"
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[Error] ffmpeg not found in PATH"
  exit 1
fi

mkdir -p "$OUT_DIR"

total=0
ok=0
failed=0
skipped=0

extract_split() {
  local split="$1"
  local split_dir="$MELD_ROOT/$split"

  if [[ ! -d "$split_dir" ]]; then
    echo "[Warn] split directory not found, skip: $split_dir"
    return
  fi

  while IFS= read -r -d '' video; do
    video="${video//$'\r'/}"
    if [[ "$video" != /* ]]; then
      video="/$video"
    fi
    video="$(realpath -m "$video")"

    total=$((total + 1))

    local base
    base="$(basename "${video%.*}")"
    local out_wav="$OUT_DIR/${split}_${base}.wav"

    if [[ -f "$out_wav" ]]; then
      skipped=$((skipped + 1))
      echo "[Skip] exists: $out_wav"
      continue
    fi

    if [[ ! -f "$video" ]]; then
      failed=$((failed + 1))
      echo "[Fail] missing source: $video"
      continue
    fi

    if ffmpeg -hide_banner -loglevel error -y \
      -i "$video" \
      -vn \
      -ac 1 \
      -ar "$SAMPLE_RATE" \
      -c:a pcm_s16le \
      "$out_wav"; then
      ok=$((ok + 1))
      echo "[OK] $video -> $out_wav"
    else
      failed=$((failed + 1))
      echo "[Fail] $video"
    fi
  done < <(find "$split_dir" -type f \( -iname '*.mp4' -o -iname '*.mkv' -o -iname '*.avi' -o -iname '*.mov' -o -iname '*.webm' \) -print0)
}

extract_split dev
extract_split test
extract_split train

echo "========== Done =========="
echo "Output directory : $OUT_DIR"
echo "Total videos     : $total"
echo "Extracted        : $ok"
echo "Skipped(existing): $skipped"
echo "Failed           : $failed"

if [[ $failed -gt 0 ]]; then
  exit 2
fi
