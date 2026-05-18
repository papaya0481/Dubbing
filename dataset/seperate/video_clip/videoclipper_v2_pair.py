#!/usr/bin/env python3

import os
import sys
import csv
import ast
import time
import argparse
import shutil
import traceback
import multiprocessing as mp
from dataclasses import dataclass
from queue import Empty
from typing import Dict, List, Optional, Tuple

import librosa
from moviepy.video.io.VideoFileClip import VideoFileClip, AudioFileClip
from tqdm import tqdm

import videoclipper_v2 as base_v2
from utils.argparse_tools import ArgumentParser
from utils.subtitle_utils import generate_srt_clip


MIN_SINGLE_WORDS = 4
MAX_COMBINED_WORDS = 35


@dataclass
class PairTask:
    video_file: str
    state_root_dir: str
    base_output_dir: str
    lang: str
    pause_threshold: float
    force_restart: bool = False


def _count_words_like_original(sent: dict) -> int:
    """Count tokens for sentence-level constraints with alignment-first strategy."""
    ts = sent.get("timestamp") or []
    aligned = base_v2._build_aligned_tokens(sent, ts) if ts else None
    if aligned:
        return len(aligned)

    text_obj = sent.get("text")
    if isinstance(text_obj, list):
        return len(text_obj)
    if isinstance(text_obj, str):
        parts = [p for p in text_obj.split() if p]
        if parts:
            return len(parts)

    raw_text = sent.get("raw_text")
    if isinstance(raw_text, str):
        parts = [p for p in raw_text.split() if p]
        if parts:
            return len(parts)

    return 0


def _sentence_text(sent: dict) -> str:
    return base_v2._text_to_string(sent.get("text") or sent.get("raw_text") or "")


def _metadata_utterance_array(parts: List[str]) -> str:
    clean_parts = [str(p).strip() for p in parts if str(p).strip()]
    return str(clean_parts)


def _metadata_utterance_to_text(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""

    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        return raw

    if isinstance(parsed, list):
        return " ".join(str(x).strip() for x in parsed if str(x).strip())
    return raw


def _candidate_state_dirs(video_file: str, state_root: str) -> List[str]:
    """Return candidate stage-1 state directories for a video."""
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    parent_dir = os.path.dirname(video_file)
    parent_dir_name = os.path.basename(parent_dir)

    candidates = [
        os.path.join(state_root, parent_dir_name, video_name),
        os.path.join(state_root, video_name),
        state_root,
    ]

    unique: List[str] = []
    seen = set()
    for c in candidates:
        if c not in seen:
            unique.append(c)
            seen.add(c)
    return unique


def _resolve_state_dir(video_file: str, state_root: str) -> Optional[str]:
    for state_dir in _candidate_state_dirs(video_file, state_root):
        if os.path.isfile(os.path.join(state_dir, "sentences")):
            return state_dir
    return None


def _build_pair_candidates(sentences: List[dict], pause_threshold: Optional[float]) -> List[dict]:
    """Build valid 2-sentence adjacent combinations under constraints."""
    candidates: List[dict] = []
    for i in range(len(sentences) - 1):
        left = sentences[i]
        right = sentences[i + 1]

        left_wc = _count_words_like_original(left)
        right_wc = _count_words_like_original(right)
        if left_wc < MIN_SINGLE_WORDS or right_wc < MIN_SINGLE_WORDS:
            continue

        total_wc = left_wc + right_wc
        if total_wc > MAX_COMBINED_WORDS:
            continue

        pause_s = max(0.0, (right["start"] - left["end"]) / 1000.0)
        if pause_threshold is not None and pause_s > pause_threshold:
            continue

        left_text = _sentence_text(left)
        right_text = _sentence_text(right)
        utterance = f"{left_text} {right_text}".strip()
        utterance_parts = [left_text, right_text]

        spk = None
        if "spk" in left and "spk" in right and left.get("spk") == right.get("spk"):
            spk = left.get("spk")

        candidates.append({
            "left_idx": i,
            "right_idx": i + 1,
            "start": left["start"] / 1000.0,
            "end": right["end"] / 1000.0,
            "pause_seconds": pause_s,
            "utterance": utterance,
            "utterance_parts": utterance_parts,
            "left_wc": left_wc,
            "right_wc": right_wc,
            "total_wc": total_wc,
            "spk": spk,
        })

    return candidates


def _quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    q = min(max(q, 0.0), 1.0)
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _describe_pause_distribution(pauses: List[float]) -> Dict[str, float]:
    vals = sorted(pauses)
    mean_val = sum(vals) / len(vals)
    return {
        "count": float(len(vals)),
        "min": vals[0],
        "p10": _quantile(vals, 0.10),
        "p25": _quantile(vals, 0.25),
        "p50": _quantile(vals, 0.50),
        "p75": _quantile(vals, 0.75),
        "p90": _quantile(vals, 0.90),
        "p95": _quantile(vals, 0.95),
        "p99": _quantile(vals, 0.99),
        "max": vals[-1],
        "mean": mean_val,
    }


def _percentile_of_threshold(pauses: List[float], threshold: float) -> float:
    if not pauses:
        return 0.0
    covered = sum(1 for x in pauses if x <= threshold)
    return covered / len(pauses) * 100.0


def _scan_candidates_for_distribution(video_files: List[str], state_root: str) -> Tuple[List[float], Dict[str, int], Dict[str, object]]:
    """Scan all videos and collect pause gaps from valid 2-sentence combinations."""
    pauses: List[float] = []
    video_candidate_counts: Dict[str, int] = {}
    scan_stats = {
        "state_missing": 0,
        "state_load_error": 0,
        "missing_examples": [],
    }

    for video_file in video_files:
        state_dir = _resolve_state_dir(video_file, state_root)

        try:
            if state_dir is None:
                scan_stats["state_missing"] += 1
                missing_examples = scan_stats["missing_examples"]
                if isinstance(missing_examples, list) and len(missing_examples) < 5:
                    missing_examples.append({
                        "video": video_file,
                        "checked": _candidate_state_dirs(video_file, state_root),
                    })
                video_candidate_counts[video_file] = 0
                continue

            state = base_v2.load_state(state_dir)
            sentences = base_v2.split_sentences_by_terminal_punctuation(state.get("sentences", []))
            cands = _build_pair_candidates(sentences, pause_threshold=None)
            video_candidate_counts[video_file] = len(cands)
            pauses.extend([c["pause_seconds"] for c in cands])
        except Exception:
            scan_stats["state_load_error"] += 1
            video_candidate_counts[video_file] = 0

    return pauses, video_candidate_counts, scan_stats


def _write_pair_metadata(output_dir: str, rows: List[Dict]) -> str:
    metadata_csv_file = os.path.join(output_dir, "metadata.csv")
    with open(metadata_csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Clip_Name",
                "Utterance",
                "Duration_Seconds",
                "Pause_Seconds",
                "Total_Words",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return metadata_csv_file


def _clip_pair_segments(
    video_file: str,
    output_dir: str,
    sentences: List[dict],
    pair_candidates: List[dict],
    vocal_file: Optional[str],
    raw_audio_file: Optional[str],
    instrumental_file: Optional[str],
) -> int:
    """Clip paired segments and write metadata rows."""
    clipped_folder = os.path.join(output_dir, "clipped")
    vocal_folder = os.path.join(output_dir, "vocals")
    instrumental_folder = os.path.join(output_dir, "instrumental")
    os.makedirs(clipped_folder, exist_ok=True)
    os.makedirs(vocal_folder, exist_ok=True)
    os.makedirs(instrumental_folder, exist_ok=True)

    srt_index = 1
    time_acc_ost = 0.0
    metadata_rows: List[Dict] = []

    base_name = os.path.basename(video_file)
    video_name_without_ext, _ = os.path.splitext(base_name)

    for pair_idx, pair in enumerate(pair_candidates):
        start = pair["start"]
        end = pair["end"]

        srt_clip, subs, srt_index = generate_srt_clip(
            sentences,
            start,
            end,
            begin_index=srt_index - 1,
            time_acc_ost=time_acc_ost,
            normalize=False,
        )
        if not subs:
            continue

        start_hours = int(subs[0][0][0] // 3600)
        start_minutes = int((subs[0][0][0] % 3600) // 60)
        start_seconds = int(subs[0][0][0] % 60)
        start_milliseconds = int((subs[0][0][0] - int(subs[0][0][0])) * 100)

        if pair.get("spk") is not None:
            clip_filename = (
                f"{video_name_without_ext}_{start_hours:02}_{start_minutes:02}_{start_seconds:02}_"
                f"{start_milliseconds:02}_pair{pair_idx:05d}_spk{pair['spk']}"
            )
        else:
            clip_filename = (
                f"{video_name_without_ext}_{start_hours:02}_{start_minutes:02}_{start_seconds:02}_"
                f"{start_milliseconds:02}_pair{pair_idx:05d}"
            )

        clip_filepath = os.path.join(clipped_folder, clip_filename)
        video_clip = clip_filepath + ".mp4"
        audio_clip = clip_filepath + ".wav"
        clip_srt_file = clip_filepath + ".srt"
        vocal_clip = os.path.join(vocal_folder, clip_filename) + ".wav"
        instrumental_clip = os.path.join(instrumental_folder, clip_filename) + ".wav"

        if not (os.path.exists(video_clip) and os.path.exists(audio_clip) and os.path.exists(clip_srt_file)):
            if vocal_file is None:
                with VideoFileClip(video_file) as video:
                    seg_end = min(end, video.duration)
                    sub = video.subclipped(start, seg_end)
                    sub.write_videofile(video_clip, audio_codec="aac", logger=None)
                    sub.audio.write_audiofile(audio_clip, codec="pcm_s16le", logger=None)
                    sub.close()
                    del sub
            else:
                with VideoFileClip(video_file) as video:
                    seg_end = min(end, video.duration)
                    sub_v = video.subclipped(start, seg_end)
                    sub_v.write_videofile(video_clip, audio_codec="aac", logger=None)
                    sub_v.close()
                    del sub_v

                with AudioFileClip(vocal_file) as vocal:
                    seg_end = min(end, vocal.duration)
                    sub_vo = vocal.subclipped(start, seg_end)
                    sub_vo.write_audiofile(vocal_clip, codec="pcm_s16le", logger=None)
                    sub_vo.close()
                    del sub_vo

                if instrumental_file:
                    with AudioFileClip(instrumental_file) as instrumental:
                        seg_end = min(end, instrumental.duration)
                        sub_in = instrumental.subclipped(start, seg_end)
                        sub_in.write_audiofile(instrumental_clip, codec="pcm_s16le", logger=None)
                        sub_in.close()
                        del sub_in

                if raw_audio_file:
                    with AudioFileClip(raw_audio_file) as audio:
                        seg_end = min(end, audio.duration)
                        sub_a = audio.subclipped(start, seg_end)
                        sub_a.write_audiofile(audio_clip, codec="pcm_s16le", logger=None)
                        sub_a.close()
                        del sub_a

            with open(clip_srt_file, "w", encoding="utf-8") as fout:
                fout.write(srt_clip)

        metadata_rows.append({
            "Clip_Name": os.path.basename(video_clip),
            "Utterance": _metadata_utterance_array(pair.get("utterance_parts") or [pair["utterance"]]),
            "Duration_Seconds": round(max(0.0, end - start), 3),
            "Pause_Seconds": round(pair["pause_seconds"], 3),
            "Total_Words": int(pair["total_wc"]),
        })

        time_acc_ost += max(0.0, end - start)

    _write_pair_metadata(output_dir, metadata_rows)
    return len(metadata_rows)


def process_single_video(task: PairTask, device: str) -> Tuple[bool, str, Optional[str], Optional[Dict]]:
    timing = {}
    start_time = time.time()

    try:
        video_file = task.video_file
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        parent_dir = os.path.dirname(video_file)
        parent_dir_name = os.path.basename(parent_dir)
        state_dir = _resolve_state_dir(video_file, task.state_root_dir)
        if state_dir is None:
            checked = _candidate_state_dirs(video_file, task.state_root_dir)
            checked_str = " | ".join(checked)
            raise FileNotFoundError(
                f"Stage-1 state not found for video: {video_file}. Checked: {checked_str}"
            )
        output_dir = os.path.join(task.base_output_dir, parent_dir_name, video_name)
        os.makedirs(output_dir, exist_ok=True)

        vocal_file = os.path.join(parent_dir, "vocals", f"{video_name}.wav")
        instrumental_file = os.path.join(parent_dir, "instrumental", f"{video_name}.wav")
        raw_audio_file = os.path.join(parent_dir, f"{video_name}.wav")

        if not os.path.exists(vocal_file):
            vocal_file = None
        if not os.path.exists(instrumental_file):
            instrumental_file = None
        if not os.path.exists(raw_audio_file):
            raw_audio_file = None

        if task.force_restart:
            clipped_folder = os.path.join(output_dir, "clipped")
            if os.path.isdir(clipped_folder):
                shutil.rmtree(clipped_folder)
            metadata_csv_file = os.path.join(output_dir, "metadata.csv")
            if os.path.isfile(metadata_csv_file):
                os.remove(metadata_csv_file)

        t1 = time.time()
        state = base_v2.load_state(state_dir)
        sentences = base_v2.split_sentences_by_terminal_punctuation(state.get("sentences", []))
        pair_candidates = _build_pair_candidates(sentences, pause_threshold=task.pause_threshold)
        timing["candidate_scan"] = time.time() - t1

        t2 = time.time()
        created = _clip_pair_segments(
            video_file=video_file,
            output_dir=output_dir,
            sentences=sentences,
            pair_candidates=pair_candidates,
            vocal_file=vocal_file,
            raw_audio_file=raw_audio_file,
            instrumental_file=instrumental_file,
        )
        timing["clip"] = time.time() - t2

        t3 = time.time()
        clip_wer_rows = []
        scored_wers: List[float] = []

        model = base_v2._worker_model
        if model is None:
            raise RuntimeError("ASR model is not initialized for stage 2 WER calculation")

        clipper = base_v2.VideoClipper(model=model, lang=task.lang, text_normalize=False)
        metadata_csv_path = os.path.join(output_dir, "metadata.csv")
        if os.path.isfile(metadata_csv_path):
            with open(metadata_csv_path, "r", encoding="utf-8") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    clip_name = (row.get("Clip_Name") or "").strip()
                    reference_text = _metadata_utterance_to_text(row.get("Utterance") or "")
                    if not clip_name:
                        continue

                    clip_stem, _ = os.path.splitext(clip_name)
                    clip_audio = os.path.join(output_dir, "clipped", clip_stem + ".wav")
                    if not os.path.isfile(clip_audio):
                        clip_audio = os.path.join(output_dir, "vocals", clip_stem + ".wav")
                    if not os.path.isfile(clip_audio):
                        continue

                    wav, sr = librosa.load(clip_audio, sr=16000)
                    _, clip_state = clipper.recog((sr, wav), clip_audio, "no", {"audio_filename": clip_audio})
                    hypothesis_text = base_v2._text_to_string(clip_state.get("recog_res_raw", ""))

                    wer = base_v2.compute_wer(reference_text, hypothesis_text)
                    clip_wer_rows.append({
                        "Clip_Name": clip_name,
                        "Reference": reference_text,
                        "Hypothesis": hypothesis_text,
                        "WER": "" if wer is None else f"{wer:.6f}",
                    })
                    if wer is not None:
                        scored_wers.append(wer)

        clip_wer_csv = os.path.join(output_dir, "clip_wer.csv")
        with open(clip_wer_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Clip_Name", "Reference", "Hypothesis", "WER"])
            writer.writeheader()
            writer.writerows(clip_wer_rows)

        avg_wer = (sum(scored_wers) / len(scored_wers)) if scored_wers else None
        timing["wer"] = time.time() - t3
        timing["pair_created"] = created
        timing["wer_summary"] = {
            "parent_folder": parent_dir_name,
            "video_name": video_name,
            "video_output_dir": output_dir,
            "average_wer": avg_wer,
            "scored_clips": len(scored_wers),
            "total_clips": len(clip_wer_rows),
        }

        timing["total"] = time.time() - start_time
        return True, video_file, None, timing

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        timing["total"] = time.time() - start_time
        return False, task.video_file, error_msg, timing


def worker_process(gpu_id: int, task_queue: mp.Queue, result_queue: mp.Queue, lang: str, device: str):
    try:
        base_v2._init_worker_process(gpu_id, lang, device)

        while True:
            try:
                task = task_queue.get(timeout=1)
                if task is None:
                    break

                success, video_file, error, timing = process_single_video(task, device)
                result_queue.put((success, video_file, error, timing))
            except Empty:
                continue
            except Exception as e:
                error_msg = f"Worker error: {e}\n{traceback.format_exc()}"
                result_queue.put((False, "unknown", error_msg, None))
    except Exception as e:
        print(f"[Worker-{os.getpid()}] Fatal error: {e}")
        traceback.print_exc()
    finally:
        result_queue.put(None)


def _prompt_threshold(pauses: List[float], preset_threshold: Optional[float], auto_confirm: bool) -> Optional[float]:
    if not pauses:
        print("[Info] No valid 2-sentence combinations found across videos.")
        return None

    stats = _describe_pause_distribution(pauses)
    print("\n" + "=" * 60)
    print("Valid pair pause distribution (seconds)")
    print("=" * 60)
    print(f"count={int(stats['count'])}")
    print(f"min={stats['min']:.3f}, mean={stats['mean']:.3f}, max={stats['max']:.3f}")
    print(
        "quantiles: "
        f"p10={stats['p10']:.3f}, p25={stats['p25']:.3f}, p50={stats['p50']:.3f}, "
        f"p75={stats['p75']:.3f}, p90={stats['p90']:.3f}, p95={stats['p95']:.3f}, p99={stats['p99']:.3f}"
    )
    print("=" * 60)

    if preset_threshold is None:
        while True:
            val = input("请输入停顿阈值秒数 (e.g. 0.6): ").strip()
            try:
                threshold = float(val)
                if threshold < 0:
                    print("阈值必须 >= 0，请重试。")
                    continue
                break
            except ValueError:
                print("输入无效，请输入数字。")
    else:
        threshold = float(preset_threshold)
        print(f"[Non-interactive] pause_threshold = {threshold:.3f}s")

    pct = _percentile_of_threshold(pauses, threshold)
    covered = sum(1 for x in pauses if x <= threshold)
    print(f"\n阈值 {threshold:.3f}s 对应分位值约 P{pct:.2f}，覆盖 {covered}/{len(pauses)} 个候选组合。")

    if auto_confirm:
        print("[Non-interactive] auto confirm enabled, continue.")
        return threshold

    answer = input("确认继续切割与WER统计？(y/n): ").strip().lower()
    if answer not in {"y", "yes"}:
        print("用户未确认，程序终止。")
        return None

    return threshold


def get_parser():
    parser = ArgumentParser(
        description="Video Clipper V2 Pair Stage2 (2-sentence combine only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Input video file or folder. If folder, videos are discovered recursively.",
    )
    parser.add_argument(
        "--state_dir",
        type=str,
        default=None,
        help=(
            "Stage-1 state root directory (default: output_dir). Supported layouts: "
            "<state_root>/<parent_folder>/<video_name>/sentences OR "
            "<state_root>/<video_name>/sentences"
        ),
    )
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--skip_processed", action="store_true", help="Skip already processed videos")
    parser.add_argument("--force_restart", action="store_true", help="Delete clipped outputs and restart")
    parser.add_argument("--lang", type=str, default="en", help="Language: zh or en")
    parser.add_argument("--device", type=str, default="gpu", choices=["cpu", "cuda", "gpu"], help="Device to use")
    parser.add_argument("--num_samples", type=int, default=None, help="Limit number of videos to process")

    # Interactive flow helpers
    parser.add_argument("--pause_threshold", type=float, default=None, help="Optional pause threshold seconds; if omitted, prompt")
    parser.add_argument("--yes", action="store_true", help="Auto confirm interactive prompt")
    return parser


def main(cmd=None):
    parser = get_parser()
    args = parser.parse_args(cmd)

    file_or_folder = args.file
    state_dir = args.state_dir
    output_dir = args.output_dir
    skip_processed = args.skip_processed
    force_restart = args.force_restart
    lang = args.lang
    device = "cuda" if args.device == "gpu" else args.device
    num_samples = args.num_samples

    use_cuda = device == "cuda" and base_v2.torch.cuda.is_available()
    num_workers = base_v2.torch.cuda.device_count() if use_cuda else 1

    print("=" * 60)
    print("Video Clipper V2 Pair (Stage 2 only)")
    print(f"Language: {lang}, Device: {device}")
    if use_cuda:
        print(f"Using {num_workers} GPU(s)")
    else:
        print("Using CPU")
    print("Rules: 2 adjacent sentences, each >3 words, combined <=35 words")
    print(f"State root: {state_dir or output_dir}")
    print(f"Output root: {output_dir}")
    print("=" * 60 + "\n")

    state_root = state_dir or output_dir

    if not os.path.isdir(file_or_folder):
        all_videos = [file_or_folder]
    else:
        all_videos = base_v2.find_all_videos(file_or_folder, output_dir, stage=2, skip_processed=skip_processed)

    print(f"Found {len(all_videos)} videos to process")
    if num_samples is not None:
        all_videos = all_videos[:num_samples]
        print(f"Limited to {num_samples} samples for testing")
    print()

    if not all_videos:
        print("No videos to process")
        return

    pauses, video_candidate_counts, scan_stats = _scan_candidates_for_distribution(all_videos, state_root)
    total_candidates = sum(video_candidate_counts.values())
    print(f"Scanned candidate pairs (before pause threshold): {total_candidates}")
    if scan_stats["state_missing"] > 0 or scan_stats["state_load_error"] > 0:
        print(
            f"Scan diagnostics: state_missing={scan_stats['state_missing']}, "
            f"state_load_error={scan_stats['state_load_error']}"
        )
        if scan_stats["state_missing"] == len(all_videos):
            print("[Hint] Stage-1 states were not found for all videos.")
            print("[Hint] Supported state layouts:")
            print("       1) <state_root>/<parent_folder>/<video_name>/sentences")
            print("       2) <state_root>/<video_name>/sentences")
            missing_examples = scan_stats.get("missing_examples", [])
            if isinstance(missing_examples, list) and missing_examples:
                print("[Hint] Checked examples:")
                for ex in missing_examples[:3]:
                    video = ex.get("video", "")
                    checked = ex.get("checked", [])
                    print(f"       video: {video}")
                    for c in checked:
                        print(f"         - {c}")

    threshold = _prompt_threshold(pauses, preset_threshold=args.pause_threshold, auto_confirm=args.yes)
    if threshold is None:
        return

    tasks = [
        PairTask(
            video_file=video,
            state_root_dir=state_root,
            base_output_dir=output_dir,
            lang=lang,
            pause_threshold=threshold,
            force_restart=force_restart,
        )
        for video in all_videos
    ]

    task_queue = mp.Queue()
    result_queue = mp.Queue()
    for task in tasks:
        task_queue.put(task)
    for _ in range(num_workers):
        task_queue.put(None)

    processes = []
    for gpu_id in range(num_workers):
        p = mp.Process(target=worker_process, args=(gpu_id, task_queue, result_queue, lang, device))
        p.start()
        processes.append(p)
        print(f"Started worker process {p.pid} on GPU {gpu_id}")
    print()

    results = []
    timings = []
    finished_workers = 0
    with tqdm(total=len(tasks), desc="Processing", unit="video") as pbar:
        while finished_workers < num_workers:
            try:
                result = result_queue.get(timeout=1)
                if result is None:
                    finished_workers += 1
                    continue

                success, video_file, error, timing = result
                results.append((success, video_file, error, timing))
                if timing:
                    timings.append(timing)
                if timings:
                    avg_time = sum(t.get("total", 0) for t in timings) / len(timings)
                    pbar.set_postfix({"avg_time": f"{avg_time:.1f}s"})
                pbar.update(1)
            except Empty:
                continue
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Terminating workers...")
                for p in processes:
                    p.terminate()
                for p in processes:
                    p.join(timeout=5)
                sys.exit(1)

    for p in processes:
        p.join()

    success_count = sum(1 for success, _, _, _ in results if success)
    failed_count = len(results) - success_count

    print("\n" + "=" * 60)
    print("Processing Complete")
    print(f"Total: {len(results)}, Success: {success_count}, Failed: {failed_count}")

    wer_rows = []
    for success, _, _, timing in results:
        if not success or not timing:
            continue
        summary = timing.get("wer_summary")
        if not summary:
            continue
        wer_rows.append({
            "Parent_Folder": summary["parent_folder"],
            "Video_Name": summary["video_name"],
            "Video_Output_Dir": summary["video_output_dir"],
            "Average_WER": "" if summary["average_wer"] is None else f"{summary['average_wer']:.6f}",
            "Scored_Clips": summary["scored_clips"],
            "Total_Clips": summary["total_clips"],
        })

    if wer_rows:
        csv_path = base_v2.write_wer_csv(output_dir, wer_rows)
        print(f"WER summary saved to: {csv_path}")

    if timings:
        avg_total = sum(t.get("total", 0) for t in timings) / len(timings)
        avg_candidates = sum(t.get("pair_created", 0) for t in timings) / len(timings)
        print("\nPerformance Statistics:")
        print(f"  Average total time: {avg_total:.2f}s")
        print(f"  Average created pair clips: {avg_candidates:.2f}")

    print("=" * 60 + "\n")

    if failed_count > 0:
        print("Failed videos:")
        for success, video, error, _ in results:
            if not success:
                print(f"\n❌ {video}")
                if error:
                    error_lines = error.split("\n")[:10]
                    print(f"   {chr(10).join(error_lines)}")
                    if len(error.split("\n")) > 10:
                        print("   ... (truncated)")


if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]

    mp.set_start_method("spawn", force=True)
    main()
