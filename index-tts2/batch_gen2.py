import argparse
import json
import logging
import os
import random
from itertools import cycle
from pathlib import Path

import torch

from indextts.infer_v2 import IndexTTS2

random.seed(42)

EMOTION_MAP = {
    "angry": "Angry",
    "happy": "Happy",
    "neutral": "Neutral",
    "sad": "Sad",
    "surprise": "Surprise",
    "surprised": "Surprise",
    "fear": "Sad",  # dataset lacks Fearful, fallback to Sad
    "fearful": "Sad",
    "disgust": "Angry",  # dataset lacks Disgusted, fallback to Angry
    "disgusted": "Angry",
}
EMOTION_DIMENSIONS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]

def get_emotion_vector(emotion: str) -> tuple[list[float], str]:
    emotion = emotion.lower().strip()
    if emotion in ["fear", "fearful"]: emotion = "sad"
    if emotion in ["disgust", "disgusted"]: emotion = "angry"
    if emotion in ["surprise", "surprised"]: emotion = "surprised"
    if emotion in ["neutral"]: emotion = "calm"
    if emotion not in EMOTION_DIMENSIONS:
        raise ValueError(f"Unknown emotion: {emotion}")
    
    vec = [0.0] * len(EMOTION_DIMENSIONS)
    if emotion in EMOTION_DIMENSIONS:
        idx = EMOTION_DIMENSIONS.index(emotion)
        vec[idx] = 1.0
    else:
        # Fallback to calm
        vec[EMOTION_DIMENSIONS.index("calm")] = 1.0
        emotion = "calm"
    return vec, emotion

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run IndexTTS2 inference to match example_output.json format")
    parser.add_argument("--data_file", type=str, required=True, help="Path to input json (same schema as example_output.json)")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to ESD dataset root")
    parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model checkpoints directory")
    parser.add_argument("--save_dir", type=str, default="index-tts2/results/generated", help="Directory to save generated wavs and updated json")
    parser.add_argument("--device", type=int, default=0, help="CUDA/MPS device index (fallback to CPU)")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap on processed samples")
    parser.add_argument("--is_fp16", action="store_true", default=False, help="Use fp16 inference if supported")
    parser.add_argument("--use_random", action="store_true", default=False, help="Use random sampling for emotion matrix")
    parser.add_argument("--verbose", action="store_true", default=True, help="Print verbose logs from IndexTTS2")
    return parser.parse_args()


def detect_language(text: str) -> str:
    chinese_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return "chinese" if chinese_chars > len(text) * 0.3 else "english"


def normalize_emotion(emotion: str) -> str:
    key = emotion.strip().lower()
    logging.warning("Unknown emotion '%s', fallback to Neutral", emotion)
    return "Neutral"


def list_wavs(emotion_dir: Path) -> list[Path]:
    return sorted([p for p in emotion_dir.glob("*.wav") if p.is_file()])


def build_reference_getter(dataset_dir: Path):
    cache: dict[tuple[str, str], list[Path]] = {}

    def get_wav(speaker_id: str, emotion: str = None) -> Path:
        # Always use Neutral emotion for reference as requested
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
    tts = IndexTTS2(model_dir=str(model_dir), cfg_path=str(cfg_path), is_fp16=is_fp16, use_cuda_kernel=False)
    # tts.device = device
    return tts


def build_emo_text(segments: list[dict]) -> str:
    parts = []
    for seg in segments:
        emo = seg.get("emotion", None)
        desc = seg.get("emotion_description", "")
        parts.append(f"{emo}: {desc}")
    return "|".join(parts)

def build_input_text(segments: list[dict]) -> str:
    return "|".join(seg.get("lines_seg", None) for seg in segments)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    data_file = Path(args.data_file)
    dataset_dir = Path(args.dataset_dir)
    save_dir = ensure_save_dir(Path(args.save_dir))

    data = load_data(data_file)
    language_hint = detect_language(data[0]["input"]["text"]) if data else "english"
    logging.info("Detected language hint: %s", language_hint)

    speaker_pools = {
        "chinese": cycle([f"{i:04d}" for i in range(1, 11)]),
        "english": cycle([f"{i:04d}" for i in range(11, 21)]),
    }

    prompt_getter = build_reference_getter(dataset_dir)
    transcripts = load_transcripts(dataset_dir)

    device = pick_device(args.device)
    logging.info("Using device: %s", device)
    tts = initialize_tts(Path(args.model_dir), args.is_fp16, device)

    logging.info("Starting inference on %d samples from %s", len(data), data_file)
    results: list = []
    output_json_path = save_dir / "inference_results_with_wav.json"
    if output_json_path.exists():
        try:
            results = load_data(output_json_path)
            logging.info("Loaded existing results from %s", output_json_path)
        except Exception:
            logging.warning("Failed to load existing results, will overwrite")

    processed = 0
    for sent_idx, item in enumerate(data):
        language = detect_language(item["input"]["text"])
        speaker_id = next(speaker_pools[language])
        if args.max_samples is not None and processed >= args.max_samples:
            break
        if any((r.get("output", {}).get("sent_idx") == sent_idx) for r in results):
            logging.info("Skipping already processed sample %s", sent_idx)
            continue

        segments = item.get("output", {}).get("segments", [])
        if any(not seg.get("emotion") for seg in segments):
            logging.warning("Skip sentence %s: missing emotion", sent_idx)
            continue

        item.setdefault("output", {})
        item["output"]["language"] = language
        item["output"]["speaker_id"] = speaker_id
        item["output"]["sent_idx"] = sent_idx

        # emo_text = build_emo_text(segments)
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

            output = tts.infer(
                spk_audio_prompt=spk_prompt,
                text=input_text,
                output_path=str(output_wav_path),
                emo_audio_prompt=None,
                emo_alpha=0,
                use_emo_text=False,
                emo_vector=emo_vectors,
                target_duration_tokens=None,
                use_random=False,
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
            )

            out_path, seg_lens, wav_secs = output
            rel_out_path = os.path.relpath(str(output_wav_path), str(save_dir))

            for seg in segments:
                seg["generated_wav"] = rel_out_path

            item["output"]["merged_wav"] = rel_out_path
            item["output"]["wav_secs"] = wav_secs
            item["output"]["seg_lens"] = seg_lens
            item["output"].pop("status", None)
            item["output"].pop("error", None)

            results.append(item)
            processed += 1

            if processed % 5 == 0:
                with output_json_path.open("w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logging.info("Checkpoint saved to %s", output_json_path)

        except Exception as exc:  # pragma: no cover
            logging.exception("Generation failed for sentence %s: %s", sent_idx, exc)
            item.setdefault("output", {})
            item["output"]["status"] = "failed"
            item["output"]["error"] = str(exc)
            results.append(item)

    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logging.info("All done. Results written to %s", output_json_path)


if __name__ == "__main__":
    main()


