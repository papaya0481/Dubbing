"""
多进程并行语音合成脚本（基于 dubbing/indextts/infer_v2.py）

- 情感来源：直接使用 speaker prompt 音频（不依赖文本标注）
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
import traceback
from pathlib import Path

import torch
import torch.multiprocessing as mp


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
    parser.add_argument("--index-tts-root", type=str, required=True,
                        help="dubbing/ 目录的路径（含 indextts 包）")
    parser.add_argument("--gpus", type=str, default="0",
                        help="逗号分隔的 GPU 编号，如 '0,1'")
    parser.add_argument("--num-process", type=int, default=2,
                        help="总进程数；建议为 GPU 数量的整数倍")
    parser.add_argument("--num-beams", type=int, default=3)
    parser.add_argument("--max-text-tokens-per-sentence", type=int, default=200)
    parser.add_argument("--max-mel-tokens", type=int, default=2000)
    parser.add_argument("--is-fp16", action="store_true", default=False)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
#  数据加载                                                                     #
# --------------------------------------------------------------------------- #

def load_work_items(csv_path: str, output_dir: Path) -> list[dict]:
    audios_dir   = output_dir / "audios"
    semantic_dir = output_dir / "semantic"
    audios_dir.mkdir(parents=True, exist_ok=True)
    semantic_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_path = str(row.get("audio_path", "")).strip()
            text = str(row.get("text", "")).strip()
            if not audio_path or not text:
                continue
            stem = Path(audio_path).stem
            out_wav = audios_dir   / f"{stem}.wav"
            out_pt  = semantic_dir / f"{stem}.pt"
            if out_wav.exists() and out_pt.exists():
                continue  # 两者都已生成则跳过
            items.append({
                "audio_path": audio_path,
                "text":       text,
                "stem":       stem,
                "out_wav":    str(out_wav),
                "out_pt":     str(out_pt),
            })
    return items


# --------------------------------------------------------------------------- #
#  单进程 worker                                                                #
# --------------------------------------------------------------------------- #

def worker(rank: int, gpu_id: int, items: list[dict], args: argparse.Namespace) -> None:
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
        try:
            result = tts.infer(
                spk_audio_prompt=item["audio_path"],
                text=item["text"],
                output_path=item["out_wav"],
                emo_audio_prompt=None,   # 默认等同 spk_audio_prompt → 以音频作情感源
                emo_vectors=None,        # 不使用文本情感向量
                verbose=False,
                num_beams=args.num_beams,
                max_text_tokens_per_sentence=args.max_text_tokens_per_sentence,
                max_mel_tokens=args.max_mel_tokens,
                return_stats=True,
            )
            if result is not None:
                _, _, _, inference_stats = result
                s_infer = inference_stats.get("S_infer")
                if s_infer is not None:
                    torch.save(s_infer, item["out_pt"])
            print(f"[rank {rank}][{idx}/{total}] {item['stem']} -> ok")
        except Exception as exc:
            traceback.print_exc()
            print(f"[rank {rank}][{idx}/{total}] {item['stem']} -> failed: {exc}")


# --------------------------------------------------------------------------- #
#  主入口                                                                       #
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # audios/ 和 semantic/ 子目录由 load_work_items 创建

    items = load_work_items(args.csv, output_dir)
    print(f"[Plan] {len(items)} items to generate (skipped existing)")
    if not items:
        print("Nothing to do.")
        return

    gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    num_process = args.num_process

    # 按顺序分块（不打乱）；每块分配给对应进程
    chunks = [items[i::num_process] for i in range(num_process)]

    processes: list[mp.Process] = []
    for rank, chunk in enumerate(chunks):
        gpu_id = gpu_ids[rank % len(gpu_ids)]
        p = mp.Process(target=worker, args=(rank, gpu_id, chunk, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("Done.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
