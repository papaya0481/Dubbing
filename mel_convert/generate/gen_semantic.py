"""
多进程并行语音合成脚本（基于 dubbing/indextts/infer_v2.py）

- Prompt 选择：优先同 speaker + 同 emotion；若不存在则同 speaker + neutral
- 二级回退：当使用 neutral prompt 时，启用 use_emo_text=True 并传入 emotion 列文本辅助情感控制
- 输出命名：与输入音频文件同名（.wav / .pt）
- 多 GPU 多进程：--gpus 0,1  --num-process 4  → 每卡 2 进程，样本均匀分配
- 自动跳过已生成的文件（断点续跑）

输入 CSV 格式（至少含以下两列）：
  audio_path   : 参考/speaker prompt 音频的完整路径
  text         : 待合成文本（"|" 分割多段）
"""

import argparse
import csv
import os
import sys
import threading
import time
import traceback
import unicodedata
from collections import defaultdict
from pathlib import Path

import torch
import torch.multiprocessing as mp


_UNICODE_TO_ASCII: dict[int, str] = {
    0x2018: "'",  # ‘
    0x2019: "'",  # ’
    0x02BC: "'",  # ʼ
    0x201C: '"',  # “
    0x201D: '"',  # ”
    0x2013: "-",  # en dash
    0x2014: "-",  # em dash
    0x2012: "-",  # figure dash
    0x00A0: " ",
    0x202F: " ",
    0x3000: " ",
    0x0091: "'",  # CP-1252 left single quotation mark
    0x0092: "'",  # CP-1252 right single quotation mark / apostrophe
    0x0093: '"',  # CP-1252 left double quotation mark
    0x0094: '"',  # CP-1252 right double quotation mark
    0x0096: "-",  # CP-1252 en dash
    0x0097: "-",  # CP-1252 em dash
}
_ZERO_WIDTH = {0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0x2060, 0xFEFF, 0x00AD}


def sanitize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    chars: list[str] = []
    for ch in text:
        cp = ord(ch)
        if cp in _ZERO_WIDTH:
            continue
        replacement = _UNICODE_TO_ASCII.get(cp)
        chars.append(replacement if replacement is not None else ch)
    text = "".join(chars)
    return text.encode("ascii", errors="ignore").decode("ascii")


# --------------------------------------------------------------------------- #
#  命令行参数                                                                   #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多进程批量语音合成（infer_v2 / IndexTTS2）")
    parser.add_argument("--csv", type=str, required=True,
                        help="输入 CSV，必须含 audio_path 和 text 两列")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="输出目录（wav + pt 都存这里）")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="IndexTTS2 模型目录（含 config.yaml）")
    parser.add_argument("--index-tts-root", type=str, default="./dubbing",
                        help="dubbing/ 目录的路径（含 indextts 包）")
    parser.add_argument("--gpus", type=str, default="0",
                        help="逗号分隔的 GPU 编号，如 '0,1'")
    parser.add_argument("--num-process", type=int, default=2,
                        help="总进程数；建议为 GPU 数量的整数倍")
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--max-text-tokens-per-sentence", type=int, default=200)
    parser.add_argument("--max-mel-tokens", type=int, default=2000)
    parser.add_argument("--is-fp16", action="store_true", default=False)
    parser.add_argument("--audio-col", type=str, default="Audio_Path",
                        help="CSV 中音频路径的列名（默认 audio_path）")
    parser.add_argument("--text-col", type=str, default="Utterance",
                        help="CSV 中文本的列名（默认 text）")
    parser.add_argument("--speaker-col", type=str, default="Speaker",
                        help="CSV 中 speaker 的列名（默认 Speaker）")
    parser.add_argument("--emotion-col", type=str, default="Emotion",
                        help="CSV 中 emotion 的列名（默认 Emotion）")
    parser.add_argument("--drop-cols", type=str, default="Video_Filename,Video_Path",
                        help="逗号分隔的要从输出 CSV 中删除的列名（默认去掉 video 列）")
    parser.add_argument("--test", action="store_true", default=False,
                        help="测试模式：只生成前 20 个样本")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
#  数据加载                                                                     #
# --------------------------------------------------------------------------- #

_NEUTRAL_ALIASES = {"neutral", "calm", "natural"}


def normalize_emotion(emotion: str) -> str:
    emo = str(emotion).strip().lower()
    if emo in _NEUTRAL_ALIASES:
        return "neutral"
    return emo


def choose_prompt_audio(
    audio_path: str,
    speaker: str,
    emotion: str,
    prompt_index: dict[tuple[str, str], list[str]],
) -> tuple[str | None, str]:
    """选择 prompt 音频。

    返回:
      (prompt_path, strategy)
      strategy in {"same_speaker_same_emotion", "same_speaker_neutral_text_fallback", "not_found"}
    """
    target = str(Path(audio_path).resolve())
    speaker_key = str(speaker).strip().lower()
    emotion_key = normalize_emotion(emotion)

    same_emo_candidates = [p for p in prompt_index.get((speaker_key, emotion_key), []) if p != target]
    if same_emo_candidates:
        return same_emo_candidates[0], "same_speaker_same_emotion"

    neutral_candidates = [p for p in prompt_index.get((speaker_key, "neutral"), []) if p != target]
    if neutral_candidates:
        return neutral_candidates[0], "same_speaker_neutral_text_fallback"

    return None, "not_found"

def save_metadata_csv(csv_path: str, output_dir: Path, audio_col: str, drop_cols: set,
                      results: dict | None = None) -> None:
    """将输入 CSV 增强后保存到 output_dir/metadata.csv

    results: stem -> {
        "prompt_audio_path": str,
        "prompt_strategy": str,
        "use_emo_text": bool,
        "out_wav": str,
        "out_pt": str,
        "gen_error": str,
    }
             None 表示尚未生成，out_wav/out_pt/gen_error 均留空
    """
    csv_base     = Path(csv_path).parent
    out_csv      = output_dir / "metadata.csv"

    rows: list[dict] = []
    fieldnames: list[str] = []

    def _to_rel_under_output(path_str: str) -> str:
        """Normalize to path relative to output_dir when possible."""
        if not path_str:
            return ""
        p = Path(path_str)
        if not p.is_absolute():
            return path_str.replace("\\", "/")
        try:
            rel = p.relative_to(output_dir)
            return rel.as_posix()
        except Exception:
            return str(p)

    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        orig_fields = [c for c in (reader.fieldnames or []) if c not in drop_cols]
        fieldnames = orig_fields + [
            "prompt_audio_path",
            "prompt_strategy",
            "use_emo_text",
            "out_wav",
            "out_pt",
            "gen_error",
        ]
        for row in reader:
            audio_path = str(row.get(audio_col, "")).strip()
            if audio_path and not Path(audio_path).is_absolute():
                audio_path = str((csv_base / audio_path).resolve())
            stem = Path(audio_path).stem if audio_path else ""
            new_row = {k: v for k, v in row.items() if k not in drop_cols}
            new_row["prompt_audio_path"] = ""
            new_row["prompt_strategy"] = ""
            new_row["use_emo_text"] = ""
            if results is not None and stem in results:
                r = results[stem]
                new_row["prompt_audio_path"] = _to_rel_under_output(r.get("prompt_audio_path", ""))
                new_row["prompt_strategy"] = r.get("prompt_strategy", "")
                new_row["use_emo_text"] = "1" if r.get("use_emo_text", False) else "0"
                new_row["out_wav"]   = _to_rel_under_output(r.get("out_wav", ""))
                new_row["out_pt"]    = _to_rel_under_output(r.get("out_pt", ""))
                new_row["gen_error"] = r.get("gen_error", "")
            else:
                new_row["out_wav"]   = ""
                new_row["out_pt"]    = ""
                new_row["gen_error"] = ""
            rows.append(new_row)

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[CSV] metadata saved → {out_csv}  ({len(rows)} rows)")


def load_work_items(
    csv_path: str,
    output_dir: Path,
    audio_col: str = "audio_path",
    text_col: str = "text",
    speaker_col: str = "Speaker",
    emotion_col: str = "Emotion",
) -> list[dict]:
    csv_base = Path(csv_path).parent
    audios_dir   = output_dir / "audios" / "ost"
    semantic_dir = output_dir / "semantic"
    audios_dir.mkdir(parents=True, exist_ok=True)
    semantic_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    prompt_index: dict[tuple[str, str], list[str]] = defaultdict(list)

    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required_cols = [audio_col, text_col, speaker_col, emotion_col]
        missing_cols = [c for c in required_cols if c not in fieldnames]
        if missing_cols:
            raise ValueError(f"CSV 缺少必要列: {missing_cols}")

        for row in reader:
            audio_path = str(row.get(audio_col, "")).strip()
            text = str(row.get(text_col, "")).strip()
            speaker = str(row.get(speaker_col, "")).strip()
            emotion = str(row.get(emotion_col, "")).strip()

            if audio_path and not Path(audio_path).is_absolute():
                audio_path = str((csv_base / audio_path).resolve())

            if not audio_path or not text or not speaker or not emotion:
                continue

            resolved_audio_path = str(Path(audio_path).resolve())
            if not Path(resolved_audio_path).exists():
                continue

            rows.append({
                "audio_path": resolved_audio_path,
                "text": sanitize_text(text),
                "speaker": speaker,
                "emotion": emotion,
                "emo_text": sanitize_text(emotion),
            })

            speaker_key = speaker.strip().lower()
            emotion_key = normalize_emotion(emotion)
            prompt_index[(speaker_key, emotion_key)].append(resolved_audio_path)

    for key in prompt_index:
        prompt_index[key] = sorted(set(prompt_index[key]))

    items: list[dict] = []

    same_emo_count = 0
    neutral_fallback_count = 0
    no_prompt_count = 0

    for row in rows:
        prompt_audio_path, prompt_strategy = choose_prompt_audio(
            audio_path=row["audio_path"],
            speaker=row["speaker"],
            emotion=row["emotion"],
            prompt_index=prompt_index,
        )

        if prompt_strategy == "same_speaker_same_emotion":
            same_emo_count += 1
        elif prompt_strategy == "same_speaker_neutral_text_fallback":
            neutral_fallback_count += 1
        else:
            no_prompt_count += 1
            continue

        stem = Path(row["audio_path"]).stem
        out_wav = audios_dir   / f"{stem}.wav"
        out_txt = audios_dir   / f"{stem}.txt"
        out_pt  = semantic_dir / f"{stem}.pt"
        if out_wav.exists() and out_txt.exists() and out_pt.exists():
            continue  # 三者都已生成则跳过

        use_emo_text = prompt_strategy == "same_speaker_neutral_text_fallback"
        items.append({
            "audio_path": row["audio_path"],
            "prompt_audio_path": prompt_audio_path,
            "prompt_strategy": prompt_strategy,
            "use_emo_text": use_emo_text,
            "text": row["text"],
            "emo_text": row["emo_text"],
            "stem": stem,
            "out_wav": str(out_wav),
            "out_txt": str(out_txt),
            "out_pt": str(out_pt),
        })

    print(
        "[PromptSelect] "
        f"same_emotion={same_emo_count}, "
        f"neutral_text_fallback={neutral_fallback_count}, "
        f"no_prompt={no_prompt_count}"
    )
    return items


# --------------------------------------------------------------------------- #
#  单进程 worker                                                                #
# --------------------------------------------------------------------------- #

def worker(rank: int, gpu_id: int, items: list[dict], args: argparse.Namespace,
           result_queue: mp.Queue) -> None:
    if not items:
        return

    # 设置 HuggingFace 缓存目录（与原 infer_v2.py 保持一致）
    os.environ["HF_HUB_CACHE"] = str(Path(args.model_dir) / "hf_cache")

    sys.path.insert(0, str(Path(args.index_tts_root)))
    import importlib
    infer_mod = importlib.import_module("indextts.infer_v2")
    IndexTTS2 = getattr(infer_mod, "IndexTTS2")

    device = f"cuda:{gpu_id}"
    cfg_path = str(Path(args.model_dir) / "config.yaml")

    tts = IndexTTS2(
        cfg_path=cfg_path,
        model_dir=args.model_dir,
        is_fp16=args.is_fp16,
        device=device,
        use_cuda_kernel=False,
    )

    total = len(items)
    for idx, item in enumerate(items, start=1):
        gen_error = ""
        out_wav_result = ""
        out_pt_result  = ""
        try:
            if not Path(item["out_wav"]).exists() or not Path(item["out_pt"]).exists():
                result = tts.infer(
                    spk_audio_prompt=item["prompt_audio_path"],
                    text=item["text"],
                    output_path=item["out_wav"],
                    emo_audio_prompt=None,
                    emo_vector=None,
                    use_emo_text=item["use_emo_text"],
                    emo_text=item["emo_text"] if item["use_emo_text"] else None,
                    verbose=False,
                    num_beams=args.num_beams,
                    method="hmm",
                    max_text_tokens_per_sentence=args.max_text_tokens_per_sentence,
                    max_mel_tokens=args.max_mel_tokens,
                    return_stats=True,
                )
                if result is not None:
                    _, _, _, inference_stats = result
                    s_infer = inference_stats.get("S_infer")
                    if s_infer is not None:
                        torch.save(s_infer, item["out_pt"])
            # 写入 transcript txt
            Path(item["out_txt"]).write_text(item["text"], encoding="utf-8")
            out_wav_result = item["out_wav"]
            out_pt_result  = item["out_pt"]
            print(f"[rank {rank}][{idx}/{total}] {item['stem']} -> ok")
        except Exception as exc:
            traceback.print_exc()
            gen_error = str(exc)
            print(f"[rank {rank}][{idx}/{total}] {item['stem']} -> failed: {exc}")

        result_queue.put({
            "stem":      item["stem"],
            "prompt_audio_path": item["prompt_audio_path"],
            "prompt_strategy": item["prompt_strategy"],
            "use_emo_text": item["use_emo_text"],
            "out_wav":   out_wav_result,
            "out_pt":    out_pt_result,
            "gen_error": gen_error,
        })


# --------------------------------------------------------------------------- #
#  进度监控                                                                     #
# --------------------------------------------------------------------------- #

def _fmt_seconds(s: float) -> str:
    """将秒数格式化为 HH:MM:SS 字符串。"""
    s = max(0.0, s)
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _progress_monitor(
    q: "mp.Queue",
    total_items: int,
    out_results: dict,
    stop_event: threading.Event,
) -> None:
    """运行在独立线程中，持续从 result_queue 读取结果并打印进度。"""
    completed = 0
    start = time.time()
    while completed < total_items and not stop_event.is_set():
        try:
            r = q.get(timeout=1.0)
            out_results[r["stem"]] = r
            completed += 1
            elapsed = time.time() - start
            avg = elapsed / completed
            remaining = avg * (total_items - completed)
            status = "ok" if not r["gen_error"] else "FAIL"
            print(
                f"\r[进度] {completed}/{total_items} "
                f"({100.0 * completed / total_items:.1f}%) "
                f"| 已用时 {_fmt_seconds(elapsed)} "
                f"| 预计剩余 {_fmt_seconds(remaining)} "
                f"| 最新: {r['stem']} [{status}]   ",
                end="",
                flush=True,
            )
        except Exception:
            pass  # queue.get 超时，继续等待
    if completed >= total_items:
        print()  # 全部完成后换行


# --------------------------------------------------------------------------- #
#  主入口                                                                       #
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # audios/ 和 semantic/ 子目录由 load_work_items 创建

    items = load_work_items(
        args.csv,
        output_dir,
        audio_col=args.audio_col,
        text_col=args.text_col,
        speaker_col=args.speaker_col,
        emotion_col=args.emotion_col,
    )
    drop_cols = {c.strip() for c in args.drop_cols.split(",") if c.strip()}
    if args.test:
        items = items[:20]
        print(f"[Test] 测试模式，截取前 20 个样本")
    print(f"[Plan] {len(items)} items to generate (skipped existing)")
    if not items:
        print("Nothing to do.")
        return

    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    num_process = args.num_process

    # 按顺序分块（不打乱）；每块分配给对应进程
    chunks = [items[i::num_process] for i in range(num_process)]

    manager = mp.Manager()
    result_queue = manager.Queue()

    processes: list[mp.Process] = []
    for rank, chunk in enumerate(chunks):
        gpu_id = gpu_ids[rank % len(gpu_ids)]
        p = mp.Process(target=worker, args=(rank, gpu_id, chunk, args, result_queue))
        p.start()
        processes.append(p)

    # 启动进度监控线程（同时负责收集结果）
    results: dict[str, dict] = {}
    stop_monitor = threading.Event()
    monitor_thread = threading.Thread(
        target=_progress_monitor,
        args=(result_queue, len(items), results, stop_monitor),
        daemon=True,
    )
    monitor_thread.start()

    for p in processes:
        p.join()

    # 等待监控线程处理完所有队列项（最多 30 s）
    stop_monitor.set()
    monitor_thread.join(timeout=30)
    if monitor_thread.is_alive():
        print()  # 确保换行
        # 兜底：手动排空剩余队列
        while not result_queue.empty():
            try:
                r = result_queue.get_nowait()
                results[r["stem"]] = r
            except Exception:
                break

    save_metadata_csv(args.csv, output_dir, args.audio_col, drop_cols, results)

    ok  = sum(1 for r in results.values() if not r["gen_error"])
    err = sum(1 for r in results.values() if r["gen_error"])
    print(f"Done. ok={ok}  failed={err}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
