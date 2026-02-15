import argparse
import ast
import csv
import importlib
import json
import random
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import textgrid


def seconds_to_mel_tokens(seconds, mel_to_sec_ratio=0.02):
	if isinstance(seconds, (list, tuple)):
		return [int(s / mel_to_sec_ratio) for s in seconds]
	return int(seconds / mel_to_sec_ratio)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="随机抽样 600 条 MELD 样本并按对齐时长生成语音")
	parser.add_argument("--dataset-root", type=str, default="/data2/ruixin/datasets/MELD_clips")
	parser.add_argument("--metadata", type=str, default="/data2/ruixin/datasets/MELD_clips/metadata.csv")
	parser.add_argument("--aligned-dir", type=str, default="/data2/ruixin/datasets/MELD_clips/audios/aligned")
	parser.add_argument("--model-dir", type=str, default="/data2/ruixin/index-tts2/checkpoints")
	parser.add_argument("--index-tts-root", type=str, default="/home/ruixin/Dubbing/index-tts2")
	parser.add_argument("--output-dir", type=str, default="/data2/ruixin/Dubbing/index-tts2/results/meld600_seed42_beam2_repeat3")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--sample-size", type=int, default=600)
	parser.add_argument("--repeat", type=int, default=3)
	parser.add_argument("--num-beams", type=int, default=2)
	parser.add_argument("--max-text-tokens-per-sentence", type=int, default=200)
	parser.add_argument("--max-mel-tokens", type=int, default=2000)
	parser.add_argument("--is-fp16", action="store_true", default=False)
	parser.add_argument("--dry-run", action="store_true", default=False)
	return parser.parse_args()


def normalize_token(token: str) -> str:
	token = token.lower().strip()
	token = re.sub(r"[^a-z0-9']+", "", token)
	return token


def tokenize_text(text: str) -> list[str]:
	raw = re.findall(r"[A-Za-z0-9']+", text)
	return [normalize_token(t) for t in raw if normalize_token(t)]


def parse_list_field(value: str) -> list[str]:
	try:
		parsed = ast.literal_eval(value)
		if isinstance(parsed, list):
			return [str(item) for item in parsed]
	except Exception:
		pass

	text = str(value).strip()
	if not (text.startswith("[") and text.endswith("]")):
		raise ValueError(f"字段不是 list: {value}")

	body = text[1:-1].strip()
	if not body:
		return []

	pattern = r"(['\"])(.*?)\1(?=\s*,|\s*$)"
	items: list[str] = []
	pos = 0
	for match in re.finditer(pattern, body):
		sep = body[pos:match.start()]
		if sep and sep.strip(" ,\t"):
			raise ValueError(f"list item 分隔符异常: {value}")
		items.append(match.group(2))
		pos = match.end()

	tail = body[pos:]
	if tail and tail.strip(" ,\t"):
		raise ValueError(f"list 末尾格式异常: {value}")
	if not items:
		raise ValueError(f"list 解析失败: {value}")
	return items


def get_textgrid_path(aligned_dir: Path, clip_filename: str) -> Path:
	base = Path(clip_filename).stem
	return aligned_dir / f"{base}_vocals.TextGrid"


def read_word_intervals(textgrid_path: Path) -> list[tuple[float, float, str]]:
	tg = textgrid.TextGrid.fromFile(str(textgrid_path))
	tier = None
	for tier_name in ["words", "word", "Words", "Word"]:
		try:
			tier = tg.getFirst(tier_name)
			break
		except Exception:
			continue
	if tier is None:
		for candidate in tg.tiers:
			if "word" in candidate.name.lower():
				tier = candidate
				break
	if tier is None:
		raise ValueError(f"TextGrid 中未找到 words tier: {textgrid_path}")

	intervals: list[tuple[float, float, str]] = []
	for interval in tier:
		mark = str(interval.mark).strip()
		if mark:
			intervals.append((float(interval.minTime), float(interval.maxTime), mark))
	if not intervals:
		raise ValueError(f"words tier 为空: {textgrid_path}")
	return intervals


def find_subsequence(tokens: list[str], sub: list[str], start_idx: int) -> tuple[int, int] | None:
	if not sub:
		return None
	max_i = len(tokens) - len(sub)
	for i in range(start_idx, max_i + 1):
		if tokens[i : i + len(sub)] == sub:
			return i, i + len(sub)
	return None


def estimate_segment_seconds(utterances: list[str], word_intervals: list[tuple[float, float, str]]) -> list[float]:
	word_marks = [normalize_token(w[2]) for w in word_intervals]
	word_marks = [w for w in word_marks if w]
	if not word_marks:
		total = word_intervals[-1][1] - word_intervals[0][0]
		split = total / max(1, len(utterances))
		return [round(split, 3) for _ in utterances]

	non_empty_intervals = [(s, e, normalize_token(m)) for s, e, m in word_intervals if normalize_token(m)]
	token_to_interval_idx = list(range(len(non_empty_intervals)))

	cursor = 0
	durations: list[float] = []
	remaining_total = non_empty_intervals[-1][1] - non_empty_intervals[0][0]
	remaining_utt = len(utterances)

	for utt in utterances:
		utt_tokens = tokenize_text(utt)
		span = find_subsequence(word_marks, utt_tokens, cursor) if utt_tokens else None

		if span is not None:
			begin_idx, end_idx = span
			seg_start = non_empty_intervals[token_to_interval_idx[begin_idx]][0]
			seg_end = non_empty_intervals[token_to_interval_idx[end_idx - 1]][1]
			duration = max(0.05, seg_end - seg_start)
			cursor = end_idx
		else:
			if remaining_utt <= 0:
				duration = max(0.05, remaining_total)
			else:
				guessed = remaining_total / remaining_utt
				duration = max(0.05, guessed)
		durations.append(round(duration, 3))
		remaining_total = max(0.0, remaining_total - durations[-1])
		remaining_utt -= 1

	return durations


def load_candidates(metadata_path: Path, dataset_root: Path, aligned_dir: Path) -> list[dict[str, Any]]:
	candidates: list[dict[str, Any]] = []
	total_rows = 0
	excluded_reasons: dict[str, int] = {}
	excluded_examples: dict[str, list[str]] = {}

	def mark_excluded(reason: str, clip_name: str = "", detail: str = "") -> None:
		excluded_reasons[reason] = excluded_reasons.get(reason, 0) + 1
		if reason not in excluded_examples:
			excluded_examples[reason] = []
		if len(excluded_examples[reason]) < 3:
			msg = clip_name or "<unknown>"
			if detail:
				msg = f"{msg} ({detail})"
			excluded_examples[reason].append(msg)

	with metadata_path.open("r", encoding="utf-8-sig") as f:
		reader = csv.DictReader(f)
		for row in reader:
			total_rows += 1
			clip_filename = row.get("Clip_Filename", "")
			clip_stem = Path(clip_filename).stem if clip_filename else ""

			sample_id_raw = row.get("Sample_ID") or row.get("\ufeffSample_ID")
			if not sample_id_raw:
				mark_excluded("missing_sample_id", clip_stem)
				continue

			if not clip_filename:
				mark_excluded("missing_clip_filename", clip_stem)
				continue

			try:
				utterances = parse_list_field(row["Utterances"])
				emotions = parse_list_field(row["Emotions"])
			except Exception as exc:
				mark_excluded("invalid_utterances_or_emotions", clip_stem, str(exc))
				continue

			if len(utterances) != len(emotions) or len(utterances) == 0:
				mark_excluded("utterance_emotion_length_mismatch", clip_stem)
				continue

			textgrid_path = get_textgrid_path(aligned_dir, clip_filename)
			if not textgrid_path.exists():
				mark_excluded("missing_textgrid", clip_stem)
				continue

			vocals_path = dataset_root / "audios" / "vocals" / f"{clip_stem}_vocals.wav"
			if not vocals_path.exists():
				mark_excluded("missing_vocals_wav", clip_stem)
				continue

			try:
				intervals = read_word_intervals(textgrid_path)
			except Exception as exc:
				mark_excluded("textgrid_read_error", clip_stem, str(exc))
				continue

			try:
				target_seconds = estimate_segment_seconds(utterances, intervals)
			except Exception as exc:
				mark_excluded("duration_estimation_error", clip_stem, str(exc))
				continue

			if len(target_seconds) != len(utterances):
				mark_excluded("target_duration_count_mismatch", clip_stem)
				continue

			candidates.append(
				{
					"sample_id": int(sample_id_raw),
					"split": row["Split"],
					"clip_filename": clip_filename,
					"vocals_path": str(vocals_path),
					"utterances": utterances,
					"emotions": emotions,
					"target_seconds": target_seconds,
					"textgrid_path": str(textgrid_path),
				}
			)
	print(f"[Load] total_rows={total_rows}, valid_candidates={len(candidates)}")
	if excluded_reasons:
		print("[Load] excluded reasons:")
		for reason, count in sorted(excluded_reasons.items(), key=lambda x: (-x[1], x[0])):
			examples = ", ".join(excluded_examples.get(reason, []))
			print(f"  - {reason}: {count} | examples: {examples}")
	return candidates


def build_text_and_emotion(sample: dict[str, Any]) -> tuple[str, str]:
	text = "|".join(sample["utterances"])
	emo_text = "|".join([str(e).lower().strip() for e in sample["emotions"]])
	return text, emo_text


def main() -> None:
	args = parse_args()

	dataset_root = Path(args.dataset_root)
	metadata_path = Path(args.metadata)
	aligned_dir = Path(args.aligned_dir)
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	candidates = load_candidates(metadata_path, dataset_root, aligned_dir)
	if len(candidates) < args.sample_size:
		raise RuntimeError(f"可用样本不足: {len(candidates)} < {args.sample_size}")

	random.seed(args.seed)
	selected = random.sample(candidates, args.sample_size)

	selected_json_path = output_dir / "selected_600_samples.json"
	with selected_json_path.open("w", encoding="utf-8") as f:
		json.dump(selected, f, ensure_ascii=False, indent=2)

	if args.dry_run:
		print(f"[DRY RUN] selected {len(selected)} samples -> {selected_json_path}")
		return

	sys.path.insert(0, str(Path(args.index_tts_root)))
	infer_mod = importlib.import_module("indextts.infer_v2")
	IndexTTS2 = getattr(infer_mod, "IndexTTS2")

	tts = IndexTTS2(
		model_dir=args.model_dir,
		cfg_path=str(Path(args.model_dir) / "config.yaml"),
		is_fp16=args.is_fp16,
		use_cuda_kernel=False,
	)

	manifest_path = output_dir / "generation_manifest.jsonl"
	total = len(selected) * args.repeat
	current = 0
	skipped_existing_count = 0

	with manifest_path.open("w", encoding="utf-8") as manifest:
		for sample in selected:
			text, emo_text = build_text_and_emotion(sample)
			target_duration_tokens = seconds_to_mel_tokens(sample["target_seconds"])
			sample_key = Path(sample["clip_filename"]).stem
			print("\n" + "=" * 100)
			print(f"[Sample] sample_id={sample['sample_id']} split={sample['split']} clip={sample['clip_filename']}")
			print(f"[Input ] vocals={sample['vocals_path']}")
			print(f"[Input ] text={text}")
			print(f"[Input ] emo_text={emo_text}")
			print(f"[Input ] target_seconds={sample['target_seconds']}")
			print(f"[Input ] target_duration_tokens={target_duration_tokens}")
			print("=" * 100)

			for rep in range(1, args.repeat + 1):
				current += 1
				out_name = f"{sample_key}_r{rep}.wav"
				out_path = output_dir / out_name
				out_txt_path = out_path.with_suffix(".txt")
				transcript_for_mfa = " ".join(sample["utterances"]).strip()

				if out_path.exists():
					skipped_existing_count += 1
					record = {
						"sample_id": sample["sample_id"],
						"split": sample["split"],
						"clip_filename": sample["clip_filename"],
						"repeat_idx": rep,
						"output_wav": str(out_path),
						"output_txt": str(out_txt_path),
						"source_vocals": sample["vocals_path"],
						"source_textgrid": sample["textgrid_path"],
						"text": text,
						"emo_text": emo_text,
						"target_seconds": sample["target_seconds"],
						"target_duration_tokens": target_duration_tokens,
						"num_beams": args.num_beams,
						"seg_lens": None,
						"wav_secs": None,
						"status": "skipped_existing",
						"error": None,
					}
					manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
					manifest.flush()
					print(f"[{current}/{total}] {sample_key} rep={rep} -> skipped_existing")
					continue

				status = "ok"
				error_msg = None
				seg_lens = None
				wav_secs = None

				try:
					_, seg_lens, wav_secs = tts.infer(
						spk_audio_prompt=sample["vocals_path"],
						text=text,
						output_path=str(out_path),
						style_prompt=None,
						emo_audio_prompt=None,
						emo_alpha=0,
						use_emo_text=True,
						emo_text=emo_text,
						use_random=False,
						verbose=True,
						emo_vector=None,
						target_duration_tokens=target_duration_tokens,
						method="hmm",
						save_attention_maps=False,
						max_text_tokens_per_sentence=args.max_text_tokens_per_sentence,
						do_sample=True,
						top_p=0.8,
						top_k=30,
						temperature=0.8,
						length_penalty=0,
						num_beams=args.num_beams,
						repetition_penalty=10.0,
						max_mel_tokens=args.max_mel_tokens,
					)
				except Exception as exc:
					status = "failed"
					error_msg = str(exc)
					print(
						f"[ERROR] sample_id={sample['sample_id']} clip={sample['clip_filename']} "
						f"rep={rep} failed: {error_msg}"
					)
					traceback.print_exc()

				if status == "ok":
					out_txt_path.write_text(transcript_for_mfa + "\n", encoding="utf-8")

				record = {
					"sample_id": sample["sample_id"],
					"split": sample["split"],
					"clip_filename": sample["clip_filename"],
					"repeat_idx": rep,
					"output_wav": str(out_path),
					"output_txt": str(out_txt_path),
					"source_vocals": sample["vocals_path"],
					"source_textgrid": sample["textgrid_path"],
					"text": text,
					"emo_text": emo_text,
					"target_seconds": sample["target_seconds"],
					"target_duration_tokens": target_duration_tokens,
					"num_beams": args.num_beams,
					"seg_lens": seg_lens,
					"wav_secs": wav_secs,
					"status": status,
					"error": error_msg,
				}
				manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
				manifest.flush()
				if status == "failed":
					print(f"[Record] failed sample written to manifest: {sample_key} rep={rep}")
				print(f"[{current}/{total}] {sample_key} rep={rep} -> {status}")

	print(f"Done. Manifest saved to: {manifest_path}")
	if skipped_existing_count > 0:
		print(f"[Summary] skipped existing samples: {skipped_existing_count}")


if __name__ == "__main__":
	main()
