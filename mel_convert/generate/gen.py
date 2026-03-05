import argparse
import ast
import csv
import importlib
import random
import sys
import traceback
import unicodedata
from pathlib import Path
from typing import Any

import torch

EMOTION_DIMENSIONS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]

_EMOTION_ALIASES: dict[str, str] = {
    # MELD labels
    "joy": "happy",
    "anger": "angry",
    "sadness": "sad",
    "fear": "afraid",
    "fearful": "afraid",
    "disgust": "disgusted",
    "surprise": "surprised",
    "neutral": "calm",
    # already-canonical passthrough handled below
}


def get_emotion_vector(emotion: str) -> list[float]:
    emotion = emotion.lower().strip()
    emotion = _EMOTION_ALIASES.get(emotion, emotion)
    if emotion not in EMOTION_DIMENSIONS:
        emotion = "calm"  # fallback
    vec = [0.0] * len(EMOTION_DIMENSIONS)
    vec[EMOTION_DIMENSIONS.index(emotion)] = 1.0
    return vec

# 将常见 Unicode 标点替换为 ASCII 等价字符，并去除零宽不可见字符
_UNICODE_TO_ASCII: dict[int, str] = {
    # 弯单引号 / 撇号
    0x2018: "'",  # '
    0x2019: "'",  # '
    0x02BC: "'",  # ʼ (modifier letter apostrophe)
    # 弯双引号
    0x201C: '"',  # "
    0x201D: '"',  # "
    # 破折号
    0x2013: "-",  # en dash
    0x2014: "-",  # em dash
    0x2012: "-",  # figure dash
    # 不间断空格 → 普通空格
    0x00A0: " ",
    0x202F: " ",
    0x3000: " ",
    # CP-1252 字符被错误解码为 UTF-8 时产生的 C1 控制字符
    # 例如 CSV 以 Windows-1252 存储但按 UTF-8 读取：\x92 → U+0092
    0x0091: "'",  # CP-1252 left single quotation mark
    0x0092: "'",  # CP-1252 right single quotation mark / apostrophe
    0x0093: '"',  # CP-1252 left double quotation mark
    0x0094: '"',  # CP-1252 right double quotation mark
    0x0096: "-",  # CP-1252 en dash
    0x0097: "-",  # CP-1252 em dash
}
# 零宽字符 → 删除
_ZERO_WIDTH = {
    0x200B,  # zero-width space
    0x200C,  # zero-width non-joiner
    0x200D,  # zero-width joiner
    0x200E,  # left-to-right mark
    0x200F,  # right-to-left mark
    0x2060,  # word joiner
    0xFEFF,  # zero-width no-break space / BOM
    0x00AD,  # soft hyphen
}


def sanitize_text(text: str) -> str:
    """将文本规范化为干净的 ASCII 可打印字符串。
    - 替换常见 Unicode 标点为 ASCII 等价字符（弯引号、破折号等）
    - 删除零宽不可见字符
    - NFKC 归一化后丢弃无法转为 ASCII 的字符
    """
    # 先做 NFKC 归一化（合并兼容字符）
    text = unicodedata.normalize("NFKC", text)
    # 逐字符替换
    chars: list[str] = []
    for ch in text:
        cp = ord(ch)
        if cp in _ZERO_WIDTH:
            continue
        replacement = _UNICODE_TO_ASCII.get(cp)
        if replacement is not None:
            chars.append(replacement)
        else:
            chars.append(ch)
    text = "".join(chars)
    # 最终只保留 ASCII 可打印字符及常规空白
    text = text.encode("ascii", errors="ignore").decode("ascii")
    return text


def seconds_to_mel_tokens(seconds: list[float], mel_to_sec_ratio: float = 0.02) -> list[int]:
	return [max(1, int(float(s) / mel_to_sec_ratio)) for s in seconds]


def parse_list_field(value: Any) -> list[Any]:
	if isinstance(value, list):
		return value
	if value is None:
		return []
	text = str(value).strip()
	if not text:
		return []
	try:
		parsed = ast.literal_eval(text)
		if isinstance(parsed, list):
			return parsed
		return [parsed]
	except Exception:
		return [text]


def parse_time_to_seconds(value: Any) -> float | None:
	if value is None:
		return None
	text = str(value).strip()
	if not text:
		return None
	text = text.replace(",", ".")
	parts = text.split(":")
	try:
		if len(parts) == 3:
			h, m, s = parts
			return float(h) * 3600 + float(m) * 60 + float(s)
		if len(parts) == 2:
			m, s = parts
			return float(m) * 60 + float(s)
		return float(parts[0])
	except Exception:
		return None


def sanitize_row_keys(row: dict[str, Any]) -> dict[str, Any]:
	cleaned: dict[str, Any] = {}
	for key, value in row.items():
		if key is None:
			continue
		cleaned[str(key).strip()] = value
	return cleaned


def load_sent_emo_rows(csv_path: Path, split: str, start_idx: int) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
		reader = csv.DictReader(f)
		for local_idx, raw_row in enumerate(reader, start=1):
			row = sanitize_row_keys(raw_row)
			utterance = sanitize_text(str(row.get("Utterance", "")).strip())
			emotion = str(row.get("Emotion", "neutral")).strip().lower()
			if not utterance:
				continue

			dialogue_id = str(row.get("Dialogue_ID", "")).strip() or "na"
			utterance_id = str(row.get("Utterance_ID", "")).strip() or str(local_idx)
			sample_key = f"{split}_dia{dialogue_id}_utt{utterance_id}"

			start_sec = parse_time_to_seconds(row.get("StartTime"))
			end_sec = parse_time_to_seconds(row.get("EndTime"))
			target_seconds: list[float] | None = None
			if start_sec is not None and end_sec is not None and end_sec > start_sec:
				target_seconds = [round(end_sec - start_sec, 3)]

			rows.append(
				{
					"global_idx": start_idx + len(rows),
					"split": split,
					"source_file": csv_path.name,
					"sample_key": sample_key,
					"utterances": [utterance],
					"emotions": [emotion],
					"target_seconds": target_seconds,
					"dialogue_id": dialogue_id,
					"utterance_ids": [utterance_id],
				}
			)
	return rows


def load_dialog_rows(csv_path: Path, split: str, start_idx: int) -> list[dict[str, Any]]:
	rows: list[dict[str, Any]] = []
	with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
		reader = csv.DictReader(f)
		for local_idx, raw_row in enumerate(reader, start=1):
			row = sanitize_row_keys(raw_row)
			utterances = [sanitize_text(str(x).strip()) for x in parse_list_field(row.get("Utterances")) if str(x).strip()]
			emotions = [str(x).strip().lower() for x in parse_list_field(row.get("Emotions")) if str(x).strip()]

			if not utterances or len(utterances) != len(emotions):
				continue

			dialogue_id = str(row.get("Dialogue_ID", "")).strip() or "na"
			utterance_ids = [str(x).strip() for x in parse_list_field(row.get("Utterance_IDs")) if str(x).strip()]
			sample_key = f"{split}_dia{dialogue_id}_row{local_idx}"

			start_times = parse_list_field(row.get("Start_Time"))
			end_times = parse_list_field(row.get("End_Time"))
			target_seconds: list[float] | None = None
			if len(start_times) == len(end_times) == len(utterances):
				durations: list[float] = []
				ok = True
				for st, ed in zip(start_times, end_times):
					s = parse_time_to_seconds(st)
					e = parse_time_to_seconds(ed)
					if s is None or e is None or e <= s:
						ok = False
						break
					durations.append(round(e - s, 3))
				if ok:
					target_seconds = durations

			rows.append(
				{
					"global_idx": start_idx + len(rows),
					"split": split,
					"source_file": csv_path.name,
					"sample_key": sample_key,
					"utterances": utterances,
					"emotions": emotions,
					"target_seconds": target_seconds,
					"dialogue_id": dialogue_id,
					"utterance_ids": utterance_ids,
				}
			)
	return rows


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="MELD 批量语音合成（支持 sent_emo / dialog 两种数据源集合）")
	parser.add_argument("--dataset-root", type=str, default="/home/ruixin/dataset/MELD")
	parser.add_argument("--dataset-option", type=str, choices=["sent_emo", "dialog"], default="sent_emo")
	parser.add_argument("--model-dir", type=str, default="/data2/ruixin/index-tts2/checkpoints")
	parser.add_argument("--index-tts-root", type=str, default="/home/ruixin/Dubbing/dubbing/indextts")
	parser.add_argument("--spk-audio-prompt", type=str, required=True)
	parser.add_argument("--output-dir", type=str, required=True)
	parser.add_argument("--sample-size", type=int, default=None, help="指定则固定随机种子42抽样；不指定则全量合成")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--repeat", type=int, default=2)
	parser.add_argument("--batch-size", type=int, default=4,
	                    help="DataLoader 每批的样本数；同 speaker 的样本合并为单次 GPT generate 调用")
	parser.add_argument("--num-beams", type=int, default=2)
	parser.add_argument("--max-text-tokens-per-sentence", type=int, default=200)
	parser.add_argument("--max-mel-tokens", type=int, default=2000)
	parser.add_argument("--is-fp16", action="store_true", default=False)
	parser.add_argument("--dry-run", action="store_true", default=False)
	return parser.parse_args()


def load_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
	dataset_root = Path(args.dataset_root)
	if args.dataset_option == "sent_emo":
		plan = [
			("dev", dataset_root / "dev_sent_emo.csv"),
			("test", dataset_root / "test_sent_emo.csv"),
			("train", dataset_root / "train_sent_emo.csv"),
		]
		loader = load_sent_emo_rows
	else:
		plan = [
			("dev", dataset_root / "dev.csv"),
			("test", dataset_root / "test.csv"),
			("train", dataset_root / "train.csv"),
		]
		loader = load_dialog_rows

	all_rows: list[dict[str, Any]] = []
	for split, path in plan:
		if not path.exists():
			raise FileNotFoundError(f"CSV 不存在: {path}")
		loaded = loader(path, split, start_idx=len(all_rows) + 1)
		print(f"[Load] {path.name}: valid={len(loaded)}")
		all_rows.extend(loaded)

	min_total_words = 5
	before_filter_count = len(all_rows)
	all_rows = [
		row
		for row in all_rows
		if count_total_words([str(x) for x in row.get("utterances", [])]) >= min_total_words
	]
	filtered_out_count = before_filter_count - len(all_rows)
	if filtered_out_count > 0:
		print(f"[Filter] total_words < {min_total_words}: removed={filtered_out_count}, kept={len(all_rows)}")

	if not all_rows:
		raise RuntimeError("没有可用样本，请检查输入 CSV")
	return all_rows


def build_text_and_emotion(sample: dict[str, Any]) -> tuple[str, str, list[list[float]]]:
	text = "|".join(sample["utterances"])
	emo_text = "|".join(sample["emotions"])
	emo_vectors = [get_emotion_vector(e) for e in sample["emotions"]]
	return text, emo_text, emo_vectors


def count_total_words(utterances: list[str]) -> int:
	return sum(len(text.strip().split()) for text in utterances if text and text.strip())


def resolve_spk_prompt_for_sample(
	spk_audio_prompt: Path,
	dataset_option: str,
	sample: dict[str, Any],
) -> Path:
	if dataset_option == "dialog":
		if spk_audio_prompt.is_file():
			return spk_audio_prompt

		if not spk_audio_prompt.is_dir():
			raise FileNotFoundError(f"dialog 模式下 spk_audio_prompt 需要是目录或音频文件: {spk_audio_prompt}")

		split = str(sample.get("split", "")).strip()
		dialogue_id = str(sample.get("dialogue_id", "")).strip()
		utterance_ids = sample.get("utterance_ids", []) or []
		first_utt_id = str(utterance_ids[0]).strip() if len(utterance_ids) > 0 else ""

		candidate_stems: list[str] = []
		if split and dialogue_id and first_utt_id:
			candidate_stems.append(f"{split}_dia{dialogue_id}_utt{first_utt_id}")
		if split and dialogue_id:
			candidate_stems.append(f"{split}_dia{dialogue_id}_utt0")
		candidate_stems.append(str(sample["sample_key"]))

		for stem in candidate_stems:
			for ext in (".wav", ".flac", ".mp3", ".m4a", ".ogg"):
				candidate = spk_audio_prompt / f"{stem}{ext}"
				if candidate.exists():
					return candidate

		raise FileNotFoundError(
			f"dialog 模式下未找到第一句参考音频，split={split}, dialogue_id={dialogue_id}, "
			f"first_utt_id={first_utt_id}, dir={spk_audio_prompt}"
		)

	if dataset_option != "sent_emo":
		if not spk_audio_prompt.is_file():
			raise FileNotFoundError(f"spk_audio_prompt 需要是音频文件: {spk_audio_prompt}")
		return spk_audio_prompt

	if spk_audio_prompt.is_file():
		return spk_audio_prompt

	if not spk_audio_prompt.is_dir():
		raise FileNotFoundError(f"sent_emo 模式下 spk_audio_prompt 需要是目录或文件: {spk_audio_prompt}")

	sample_key = str(sample["sample_key"])
	for ext in (".wav", ".flac", ".mp3", ".m4a", ".ogg"):
		candidate = spk_audio_prompt / f"{sample_key}{ext}"
		if candidate.exists():
			return candidate

	raise FileNotFoundError(f"未找到 sent_emo 参考音频: {spk_audio_prompt / (sample_key + '.wav')}")


def main() -> None:
	args = parse_args()
	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	emb_dir = output_dir.parent / "code_embeddings"
	emb_dir.mkdir(parents=True, exist_ok=True)

	spk_audio_prompt = Path(args.spk_audio_prompt)
	if not spk_audio_prompt.exists():
		raise FileNotFoundError(f"spk_audio_prompt 不存在: {spk_audio_prompt}")

	candidates = load_candidates(args)

	if args.sample_size is not None:
		if args.sample_size <= 0:
			raise ValueError("sample-size 必须 > 0")
		if args.sample_size > len(candidates):
			raise RuntimeError(f"可用样本不足: {len(candidates)} < {args.sample_size}")
		rng = random.Random(42)
		selected = rng.sample(candidates, args.sample_size)
		print(f"[Select] fixed seed=42, sample_size={args.sample_size}")
	else:
		selected = candidates
		print(f"[Select] use all samples: {len(selected)}")

	if args.dry_run:
		print("[DRY RUN] skip model inference")
		return

	sys.path.insert(0, str(Path(args.index_tts_root)))
	infer_batch_mod = importlib.import_module("indextts.infer_batch")
	IndexTTS2Batch = getattr(infer_batch_mod, "IndexTTS2Batch")
	DubbingDataset = getattr(infer_batch_mod, "DubbingDataset")
	collate_items  = getattr(infer_batch_mod, "collate_items")

	from torch.utils.data import DataLoader

	tts = IndexTTS2Batch(
		model_dir=args.model_dir,
		cfg_path=str(Path(args.model_dir) / "config.yaml"),
		is_fp16=args.is_fp16,
		use_cuda_kernel=False,
		s2mel_max_batch_size=args.batch_size,
	)

	# ------------------------------------------------------------------
	# 构建全量工作项列表（每条 = 一个 (sample, rep) 组合）
	# ------------------------------------------------------------------
	fieldnames = [
		"sample_id",
		"split",
		"source_file",
		"dataset_option",
		"sample_key",
		"dialogue_id",
		"utterance_ids",
		"repeat_idx",
		"output_wav",
		"output_txt",
		"source_text",
		"source_emotions",
		"text",
		"emo_text",
		"target_seconds",
		"target_duration_tokens",
		"source_prompt_wav",
		"num_beams",
		"seg_lens",
		"wav_secs",
		"status",
		"error",
	]

	work_items: list[dict[str, Any]] = []
	for sample in selected:
		text, emo_text, emo_vectors = build_text_and_emotion(sample)
		sample_spk_audio_prompt: Path | None = None
		sample_prompt_error = ""
		try:
			sample_spk_audio_prompt = resolve_spk_prompt_for_sample(
				spk_audio_prompt=spk_audio_prompt,
				dataset_option=args.dataset_option,
				sample=sample,
			)
		except Exception as exc:
			sample_prompt_error = str(exc)

		target_duration_tokens = (
			seconds_to_mel_tokens(sample["target_seconds"])
			if sample.get("target_seconds") is not None
			else None
		)

		for rep in range(1, args.repeat + 1):
			out_wav = output_dir / f"{sample['sample_key']}_r{rep}.wav"
			out_txt = out_wav.with_suffix(".txt")

			# 解析已跳过 / 错误条件，供 DataLoader 过滤
			if out_wav.exists():
				pre_status = "skipped_existing"
			elif sample_spk_audio_prompt is None:
				pre_status = "failed_no_prompt"
			else:
				pre_status = "pending"

			work_items.append({
				# DataLoader 所需的核心推理字段
				"spk_audio_prompt": str(sample_spk_audio_prompt) if sample_spk_audio_prompt else "",
				"text":             text,
				"output_path":      str(out_wav),
				"emo_vectors":      emo_vectors,
				"target_duration_tokens": target_duration_tokens,
				# 元数据字段（写 CSV 用）
				"sample_id":           sample["global_idx"],
				"split":               sample["split"],
				"source_file":         sample["source_file"],
				"dataset_option":      args.dataset_option,
				"sample_key":          sample["sample_key"],
				"dialogue_id":         sample["dialogue_id"],
				"utterance_ids":       "|".join(sample["utterance_ids"]),
				"repeat_idx":          rep,
				"output_wav":          str(out_wav),
				"output_txt":          str(out_txt),
				"source_text":         "|".join(sample["utterances"]),
				"source_emotions":     "|".join(sample["emotions"]),
				"emo_text":            emo_text,
				"target_seconds":      str(sample.get("target_seconds")),
				"target_duration_tokens_str": str(target_duration_tokens),
				"source_prompt_wav":   str(sample_spk_audio_prompt) if sample_spk_audio_prompt else "",
				"pre_status":          pre_status,
				"pre_error":           sample_prompt_error,
				"transcript":          " ".join(sample["utterances"]).strip(),
			})

	total = len(work_items)
	print(f"[Plan] total work items: {total}  (samples={len(selected)} × repeat={args.repeat})")

	# ------------------------------------------------------------------
	# DataLoader：按 batch_size 分批，collate_items 保持 dict 列表原样
	# ------------------------------------------------------------------
	dataset = DubbingDataset(work_items)
	loader  = DataLoader(
		dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=0,          # 推理在主进程；数据本身是纯 Python dict，无需多进程
		collate_fn=collate_items,
		drop_last=False,
	)

	metadata_path = output_dir / "generation_metadata.csv"
	current = 0

	with metadata_path.open("w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()

		for batch in loader:
			# ---- 过滤无需推理的条目 ----
			pending   = [item for item in batch if item["pre_status"] == "pending"]
			non_infer = [item for item in batch if item["pre_status"] != "pending"]

			# 对 non_infer 直接写入 CSV
			for item in non_infer:
				status = "skipped_existing" if item["pre_status"] == "skipped_existing" else "failed"
				error  = item["pre_error"] if status == "failed" else ""
				if status == "skipped_existing":
					transcript = item["transcript"]
					if transcript:
						Path(item["output_txt"]).write_text(transcript + "\n", encoding="utf-8")
				writer.writerow({
					"sample_id":            item["sample_id"],
					"split":                item["split"],
					"source_file":          item["source_file"],
					"dataset_option":       item["dataset_option"],
					"sample_key":           item["sample_key"],
					"dialogue_id":          item["dialogue_id"],
					"utterance_ids":        item["utterance_ids"],
					"repeat_idx":           item["repeat_idx"],
					"output_wav":           item["output_wav"],
					"output_txt":           item["output_txt"],
					"source_text":          item["source_text"],
					"source_emotions":      item["source_emotions"],
					"text":                 item["text"],
					"emo_text":             item["emo_text"],
					"target_seconds":       item["target_seconds"],
					"target_duration_tokens": item["target_duration_tokens_str"],
					"source_prompt_wav":    item["source_prompt_wav"],
					"num_beams":            args.num_beams,
					"seg_lens":             "",
					"wav_secs":             "",
					"status":               status,
					"error":                error,
				})
				current += 1
				print(f"[{current}/{total}] {item['sample_key']} rep={item['repeat_idx']} -> {status}")

			if not pending:
				f.flush()
				continue

			# ---- 调用批量推理 ----
			try:
				batch_results = tts.infer_batch(
					spk_audio_prompts=[item["spk_audio_prompt"] for item in pending],
					texts=[item["text"] for item in pending],
					output_paths=[item["output_path"] for item in pending],
					emo_vectors_list=[item["emo_vectors"] for item in pending],
					target_duration_tokens_list=[item["target_duration_tokens"] for item in pending],
					verbose=True,
					num_beams=args.num_beams,
					max_text_tokens_per_sentence=args.max_text_tokens_per_sentence,
					max_mel_tokens=args.max_mel_tokens,
					return_stats=True,
				)
			except Exception as exc:
				traceback.print_exc()
				# 整批失败：逐条写失败记录
				for item in pending:
					writer.writerow({
						"sample_id":            item["sample_id"],
						"split":                item["split"],
						"source_file":          item["source_file"],
						"dataset_option":       item["dataset_option"],
						"sample_key":           item["sample_key"],
						"dialogue_id":          item["dialogue_id"],
						"utterance_ids":        item["utterance_ids"],
						"repeat_idx":           item["repeat_idx"],
						"output_wav":           item["output_wav"],
						"output_txt":           item["output_txt"],
						"source_text":          item["source_text"],
						"source_emotions":      item["source_emotions"],
						"text":                 item["text"],
						"emo_text":             item["emo_text"],
						"target_seconds":       item["target_seconds"],
						"target_duration_tokens": item["target_duration_tokens_str"],
						"source_prompt_wav":    item["source_prompt_wav"],
						"num_beams":            args.num_beams,
						"seg_lens":             "",
						"wav_secs":             "",
						"status":               "failed",
						"error":                str(exc),
					})
					current += 1
					print(f"[{current}/{total}] {item['sample_key']} -> failed (batch error)")
				f.flush()
				continue

			# ---- 处理推理结果 ----
			for item, result in zip(pending, batch_results):
				seg_lens_str = ""
				wav_secs_str = ""
				status       = "ok"
				error_msg    = ""

				if result is None:
					status    = "failed"
					error_msg = "infer_batch returned None"
				else:
					try:
						out_path, seg_ls, wav_secs_raw, inference_stats = result
						seg_lens_str = str(seg_ls)
						wav_secs_str = str(wav_secs_raw)

						# 保存 code embedding（若有）
						s_infer_tensor = inference_stats.get("S_infer") if isinstance(inference_stats, dict) else None
						if s_infer_tensor is not None:
							emb_path = emb_dir / Path(item["output_wav"]).with_suffix(".pt").name
							torch.save(s_infer_tensor, emb_path)
					except Exception as exc2:
						status    = "failed"
						error_msg = str(exc2)

				if status in {"ok", "skipped_existing"}:
					transcript = item["transcript"]
					if transcript:
						Path(item["output_txt"]).write_text(transcript + "\n", encoding="utf-8")

				writer.writerow({
					"sample_id":            item["sample_id"],
					"split":                item["split"],
					"source_file":          item["source_file"],
					"dataset_option":       item["dataset_option"],
					"sample_key":           item["sample_key"],
					"dialogue_id":          item["dialogue_id"],
					"utterance_ids":        item["utterance_ids"],
					"repeat_idx":           item["repeat_idx"],
					"output_wav":           item["output_wav"],
					"output_txt":           item["output_txt"],
					"source_text":          item["source_text"],
					"source_emotions":      item["source_emotions"],
					"text":                 item["text"],
					"emo_text":             item["emo_text"],
					"target_seconds":       item["target_seconds"],
					"target_duration_tokens": item["target_duration_tokens_str"],
					"source_prompt_wav":    item["source_prompt_wav"],
					"num_beams":            args.num_beams,
					"seg_lens":             seg_lens_str,
					"wav_secs":             wav_secs_str,
					"status":               status,
					"error":                error_msg,
				})
				current += 1
				print(f"[{current}/{total}] {item['sample_key']} rep={item['repeat_idx']} -> {status}")

			f.flush()

	print(f"Done. metadata csv saved to: {metadata_path}")


if __name__ == "__main__":
	main()
