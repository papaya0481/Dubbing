import argparse
import json
import logging
import os
import random
import time
from itertools import cycle
from pathlib import Path

import torch

from indextts.infer_v2 import IndexTTS2

RANDOM_SEED = 42
SAMPLE_SIZE = 200

EMOTION_DIMENSIONS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]


def get_emotion_vector(emotion: str) -> tuple[list[float], str]:
    emotion = emotion.lower().strip()
    if emotion in ["fear", "fearful"]:
        emotion = "sad"
    if emotion in ["disgust", "disgusted"]:
        emotion = "angry"
    if emotion in ["surprise", "surprised"]:
        emotion = "surprised"
    if emotion in ["neutral"]:
        emotion = "calm"
    if emotion not in EMOTION_DIMENSIONS:
        raise ValueError(f"Unknown emotion: {emotion}")

    vec = [0.0] * len(EMOTION_DIMENSIONS)
    idx = EMOTION_DIMENSIONS.index(emotion)
    vec[idx] = 1.0
    return vec, emotion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference with timing/RTF statistics")
    parser.add_argument("--data_file", type=str, required=True, help="Path to input json")
    parser.add_argument(
        "--selected_data_file",
        type=str,
        default=None,
        help="Path to save randomly selected samples as a new data_file json",
    )
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to ESD dataset root")
    parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
    parser.add_argument("--save_dir", type=str, default="index-tts2/results/generated_timing", help="Directory to save wavs and json")
    parser.add_argument("--device", type=int, default=0, help="CUDA/MPS device index (fallback to CPU)")
    parser.add_argument("--is_fp16", action="store_true", default=False, help="Use fp16 inference if supported")
    parser.add_argument("--use_random", action="store_true", default=False, help="Use random sampling for emotion matrix")
    parser.add_argument("--verbose", action="store_true", default=False, help="Verbose logs from IndexTTS2")
    return parser.parse_args()


def detect_language(text: str) -> str:
    chinese_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return "chinese" if chinese_chars > len(text) * 0.3 else "english"


def list_wavs(emotion_dir: Path) -> list[Path]:
    return sorted([p for p in emotion_dir.glob("*.wav") if p.is_file()])


def build_reference_getter(dataset_dir: Path):
    cache: dict[tuple[str, str], list[Path]] = {}

    def get_wav(speaker_id: str) -> Path:
        target_emotion = "Neutral"
        cache_key = (speaker_id, target_emotion)
        if cache_key not in cache:
            emotion_dir = dataset_dir / speaker_id / target_emotion
            wavs = list_wavs(emotion_dir)
            if not wavs:
                raise FileNotFoundError(f"No wavs under {emotion_dir}")
            cache[cache_key] = wavs
        return cache[cache_key][0]

    return get_wav


def load_transcripts(dataset_dir: Path) -> dict[str, dict[str, str]]:
    transcripts: dict[str, dict[str, str]] = {}
    for speaker_dir in dataset_dir.iterdir():
        if not speaker_dir.is_dir():
            continue
        txt_path = speaker_dir / f"{speaker_dir.name}.txt"
        if not txt_path.exists():
            continue
        speaker_map: dict[str, str] = {}
        with txt_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    utt_id, transcript = parts[0], parts[1]
                    speaker_map[utt_id] = transcript
        transcripts[speaker_dir.name] = speaker_map
    return transcripts


def pick_device(idx: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{idx}")
    if torch.backends.mps.is_available():
        return torch.device(f"mps:{idx}")
    return torch.device("cpu")


def load_data(path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(f"data_file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_save_dir(save_dir: Path) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def initialize_tts(model_dir: Path, is_fp16: bool, device: torch.device) -> IndexTTS2:
    cfg_path = model_dir / "config.yaml"
    tts = IndexTTS2(model_dir=str(model_dir), cfg_path=str(cfg_path), is_fp16=is_fp16, use_cuda_kernel=True)
    if hasattr(tts, "logger") and hasattr(tts.logger, "set_level"):
        tts.logger.set_level("DEBUG")
    return tts


def build_input_text(segments: list[dict]) -> str:
    return "|".join(seg.get("lines_seg", "") for seg in segments)


def select_random_samples(data: list, sample_size: int = SAMPLE_SIZE, seed: int = RANDOM_SEED) -> list[tuple[int, dict]]:
    rng = random.Random(seed)
    if len(data) <= sample_size:
        indices = list(range(len(data)))
        rng.shuffle(indices)
    else:
        indices = rng.sample(range(len(data)), sample_size)
    return [(idx, data[idx]) for idx in indices]


def save_selected_data_file(selected: list[tuple[int, dict]], output_path: Path) -> None:
    selected_items = []
    for sent_idx, item in selected:
        item_copy = dict(item)
        item_copy["selected_sent_idx"] = sent_idx
        selected_items.append(item_copy)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(selected_items, f, ensure_ascii=False, indent=2)


def main() -> None:
    random.seed(RANDOM_SEED)
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    data_file = Path(args.data_file)
    dataset_dir = Path(args.dataset_dir)
    save_dir = ensure_save_dir(Path(args.save_dir))

    data = load_data(data_file)
    selected = select_random_samples(data, sample_size=SAMPLE_SIZE, seed=RANDOM_SEED)

    selected_data_file = (
        Path(args.selected_data_file)
        if args.selected_data_file
        else save_dir / f"selected_data_seed{RANDOM_SEED}_n{len(selected)}.json"
    )
    save_selected_data_file(selected, selected_data_file)

    logging.info("Loaded %d samples from %s", len(data), data_file)
    logging.info("Randomly selected %d samples with fixed seed=%d", len(selected), RANDOM_SEED)
    logging.info("Selected samples data_file saved to %s", selected_data_file)

    speaker_pools = {
        "chinese": cycle([f"{i:04d}" for i in range(1, 11)]),
        "english": cycle([f"{i:04d}" for i in range(11, 21)]),
    }

    prompt_getter = build_reference_getter(dataset_dir)
    transcripts = load_transcripts(dataset_dir)

    device = pick_device(args.device)
    logging.info("Using device: %s", device)
    tts = initialize_tts(Path(args.model_dir), args.is_fp16, device)

    batch_start = time.perf_counter()
    processed_items: list[dict] = []

    total_inference_time_sum = 0.0
    total_token_generation_time_sum = 0.0
    total_wav_seconds_sum = 0.0

    for local_idx, (sent_idx, item) in enumerate(selected):
        input_text_raw = item.get("input", {}).get("text", "")
        language = detect_language(input_text_raw)
        speaker_id = next(speaker_pools[language])

        output_obj = item.get("output", {})
        segments = output_obj.get("segments", [])
        if not segments:
            logging.warning("Skip sent_idx=%s: empty segments", sent_idx)
            continue
        if any(not seg.get("emotion") for seg in segments):
            logging.warning("Skip sent_idx=%s: missing emotion", sent_idx)
            continue

        input_text = build_input_text(segments)

        emo_vectors = []
        for seg in segments:
            emo = seg.get("emotion", "neutral")
            emo_vec, mapped_emo = get_emotion_vector(emo)
            emo_vectors.append(emo_vec)
            seg["emotion_mapped"] = mapped_emo

        try:
            prompt_paths = []
            for seg in segments:
                prompt_wav_path = prompt_getter(speaker_id)
                prompt_utt_id = prompt_wav_path.stem
                prompt_text = transcripts.get(speaker_id, {}).get(prompt_utt_id)

                seg["prompt_wav"] = os.path.relpath(str(prompt_wav_path), str(dataset_dir))
                seg["prompt_utt_id"] = prompt_utt_id
                seg["prompt_text"] = prompt_text
                prompt_paths.append(prompt_wav_path)

            output_wav_path = save_dir / f"{language}_{sent_idx:04d}.wav"
            spk_prompt = str(prompt_paths[0]) if prompt_paths else None

            out_path, seg_lens, wav_secs, stats = tts.infer(
                spk_audio_prompt=spk_prompt,
                text=input_text,
                output_path=str(output_wav_path),
                emo_audio_prompt=None,
                emo_alpha=0,
                use_emo_text=False,
                emo_vector=emo_vectors,
                target_duration_tokens=None,
                use_random=args.use_random,
                verbose=args.verbose,
                max_text_tokens_per_sentence=200,
                do_sample=True,
                top_p=0.8,
                top_k=30,
                temperature=0.8,
                length_penalty=0,
                num_beams=3,
                repetition_penalty=1.2,
                max_mel_tokens=2000,
                method="hmm",
                return_stats=True,
            )

            rel_out_path = os.path.relpath(str(output_wav_path), str(save_dir))
            for seg in segments:
                seg["generated_wav"] = rel_out_path

            total_inference_time = float(stats.get("total_inference_time", 0.0))
            rtf = float(stats.get("rtf", 0.0))
            token_generation_time = float(stats.get("token_generation_time", 0.0))

            total_inference_time_sum += total_inference_time
            total_token_generation_time_sum += token_generation_time
            total_wav_seconds_sum += float(wav_secs)

            processed_items.append(
                {
                    "sent_idx": sent_idx,
                    "language": language,
                    "speaker_id": speaker_id,
                    "merged_wav": rel_out_path,
                    "wav_secs": float(wav_secs),
                    "seg_lens": seg_lens,
                    "timing": {
                        "generation_time_sec": total_inference_time,
                        "rtf": rtf,
                        "token_generation_time_sec": token_generation_time,
                    },
                    "sample": item,
                }
            )

            logging.info(
                "[%d/%d] sent_idx=%s done | gen_time=%.3fs | token_gen=%.3fs | wav=%.3fs | RTF=%.4f",
                local_idx + 1,
                len(selected),
                sent_idx,
                total_inference_time,
                token_generation_time,
                float(wav_secs),
                rtf,
            )

        except Exception as exc:  # pragma: no cover
            logging.exception("Generation failed for sent_idx=%s: %s", sent_idx, exc)
            processed_items.append(
                {
                    "sent_idx": sent_idx,
                    "status": "failed",
                    "error": str(exc),
                    "sample": item,
                }
            )

    batch_end = time.perf_counter()
    overall_time_sec = batch_end - batch_start
    overall_rtf = (overall_time_sec / total_wav_seconds_sum) if total_wav_seconds_sum > 0 else 0.0

    result_obj = {
        "meta": {
            "seed": RANDOM_SEED,
            "requested_sample_size": SAMPLE_SIZE,
            "selected_sample_size": len(selected),
            "processed_count": len(processed_items),
        },
        "summary": {
            "overall_time_sec": overall_time_sec,
            "total_generated_wav_sec": total_wav_seconds_sum,
            "overall_rtf": overall_rtf,
            "sum_generation_time_sec": total_inference_time_sum,
            "sum_token_generation_time_sec": total_token_generation_time_sum,
        },
        "results": processed_items,
    }

    output_json_path = save_dir / "inference_results_timing.json"
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(result_obj, f, ensure_ascii=False, indent=2)

    logging.info("All done. Output written to %s", output_json_path)
    logging.info(
        "Summary | overall_time=%.3fs | total_wav=%.3fs | overall_RTF=%.4f",
        overall_time_sec,
        total_wav_seconds_sum,
        overall_rtf,
    )


if __name__ == "__main__":
    main()
