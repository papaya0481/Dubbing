"""MFA alignment helper for metadata-driven dataset structure.

Directory layout expected under ``src``:
  src/
    metadata.csv
    audios/
      ost/*.wav
      vocals/*_vocals.wav (optional)

Workflow:
1. Read ``metadata.csv`` and fetch transcript from ``Utterance`` column.
2. For each sample, prefer ``audios/vocals/{name}_vocals.wav``;
   fallback to ``audios/ost/{name}.wav``.
3. Write transcript txt beside selected wav for MFA compatibility.
4. Run MFA alignment and export ``TextGrid`` into ``audios/aligned/{name}.TextGrid``.
5. Copy all MFA CSV files into ``audios/aligned`` before temp cleanup.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SUPPORTED_AUDIO_KEYS = (
    "Audio_Filename",
    "audio_filename",
    "audio",
    "filename",
    "wav",
)

SUPPORTED_TEXT_KEYS = (
    "Utterance",
    "utterance",
    "Transcript",
    "transcript",
    "Text",
    "text",
)


@dataclass
class SampleItem:
    sample_name: str
    utterance: str
    selected_audio_path: Path


@dataclass
class TranscriptBackup:
    txt_path: Path
    existed_before: bool
    original_content: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MFA alignment from metadata.csv with vocals-first fallback."
    )
    parser.add_argument("src", type=Path, help="Dataset root containing metadata.csv")
    parser.add_argument(
        "--dictionary",
        default="english_us_arpa",
        help="MFA pronunciation dictionary name/path (default: english_us_arpa)",
    )
    parser.add_argument(
        "--acoustic-model",
        default="english_us_arpa",
        help="MFA acoustic model name/path (default: english_us_arpa)",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=8,
        help="Number of MFA jobs (default: 8)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Pass --clean to MFA to remove previous temp state",
    )
    parser.add_argument(
        "--single-speaker",
        action="store_true",
        help="Pass --single_speaker to MFA",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary corpus/output folders for debugging",
    )
    parser.add_argument(
        "--keep-audio-txt",
        action="store_true",
        help="Do not remove generated transcript txt next to source audio",
    )
    parser.add_argument(
        "--add-words",
        action="store_true",
        default=False,
        help="Generate pronunciations for OOV words and add them to the dictionary",
    )
    return parser.parse_args()


def first_non_empty(row: dict[str, str], keys: Iterable[str]) -> str:
    for key in keys:
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def normalize_sample_name(audio_filename_value: str) -> str:
    stem = Path(audio_filename_value).stem
    if stem.endswith("_vocals"):
        return stem[: -len("_vocals")]
    return stem


def resolve_audio_path(src_root: Path, sample_name: str) -> Path | None:
    vocals_wav = src_root / "audios" / "vocals" / f"{sample_name}_vocals.wav"
    ost_wav = src_root / "audios" / "ost" / f"{sample_name}.wav"

    if vocals_wav.exists():
        return vocals_wav
    if ost_wav.exists():
        return ost_wav
    return None


def load_samples(src_root: Path) -> list[SampleItem]:
    metadata_path = src_root / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")

    samples: list[SampleItem] = []
    skipped_missing_audio_key: list[str] = []
    skipped_missing_text: list[str] = []
    missing_audio: list[str] = []

    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("metadata.csv has no header")

        for row_idx, row in enumerate(reader, start=2):
            audio_value = first_non_empty(row, SUPPORTED_AUDIO_KEYS)
            utterance = first_non_empty(row, SUPPORTED_TEXT_KEYS)

            if not audio_value:
                skipped_missing_audio_key.append(
                    f"line={row_idx}, missing audio filename, supported keys: {SUPPORTED_AUDIO_KEYS}"
                )
                continue
            if not utterance:
                skipped_missing_text.append(
                    f"line={row_idx}, sample={audio_value}, missing transcript, supported keys: {SUPPORTED_TEXT_KEYS}"
                )
                continue

            sample_name = normalize_sample_name(audio_value)
            selected_audio = resolve_audio_path(src_root, sample_name)
            if selected_audio is None:
                missing_audio.append(
                    f"line={row_idx}, sample={sample_name}, tried: "
                    f"audios/vocals/{sample_name}_vocals.wav, audios/ost/{sample_name}.wav"
                )
                continue

            samples.append(
                SampleItem(
                    sample_name=sample_name,
                    utterance=utterance,
                    selected_audio_path=selected_audio,
                )
            )

    if skipped_missing_audio_key:
        preview = skipped_missing_audio_key[:20]
        suffix = " ..." if len(skipped_missing_audio_key) > 20 else ""
        print(
            "[WARN] Skipped rows with missing audio filename "
            f"({len(skipped_missing_audio_key)}): {preview}{suffix}"
        )

    if skipped_missing_text:
        preview = skipped_missing_text[:20]
        suffix = " ..." if len(skipped_missing_text) > 20 else ""
        print(
            "[WARN] Skipped rows with missing transcript "
            f"({len(skipped_missing_text)}): {preview}{suffix}"
        )

    if missing_audio:
        preview = missing_audio[:20]
        suffix = " ..." if len(missing_audio) > 20 else ""
        print(
            "[WARN] Skipped rows with missing audio files after vocals->ost fallback "
            f"({len(missing_audio)}): {preview}{suffix}"
        )

    if not samples:
        raise ValueError("No valid rows found in metadata.csv")

    return samples


def write_transcript_next_to_audio(sample: SampleItem) -> TranscriptBackup:
    txt_path = sample.selected_audio_path.with_suffix(".txt")
    existed_before = txt_path.exists()
    original_content = txt_path.read_text(encoding="utf-8") if existed_before else None
    txt_path.write_text(sample.utterance.strip() + "\n", encoding="utf-8")
    return TranscriptBackup(
        txt_path=txt_path,
        existed_before=existed_before,
        original_content=original_content,
    )


def prepare_temp_corpus(
    src_root: Path, samples: list[SampleItem]
) -> tuple[Path, dict[str, str], list[TranscriptBackup]]:
    tmp_corpus = src_root / "audios" / ".mfa_tmp_corpus"
    if tmp_corpus.exists():
        shutil.rmtree(tmp_corpus)
    tmp_corpus.mkdir(parents=True, exist_ok=True)

    # map aligned base name in temp corpus -> canonical sample name for final output naming.
    name_mapping: dict[str, str] = {}
    transcript_backups: list[TranscriptBackup] = []

    for sample in samples:
        backup = write_transcript_next_to_audio(sample)
        transcript_backups.append(backup)

        temp_base = sample.selected_audio_path.stem
        if temp_base in name_mapping:
            raise ValueError(
                f"Name collision in temporary corpus for '{temp_base}'. "
                "Please ensure unique audio stems across selected samples."
            )

        wav_target = tmp_corpus / f"{temp_base}.wav"
        txt_target = tmp_corpus / f"{temp_base}.txt"
        shutil.copy2(sample.selected_audio_path, wav_target)
        shutil.copy2(sample.selected_audio_path.with_suffix(".txt"), txt_target)
        name_mapping[temp_base] = sample.sample_name

    return tmp_corpus, name_mapping, transcript_backups


def cleanup_generated_audio_txt(transcript_backups: list[TranscriptBackup]) -> None:
    for backup in transcript_backups:
        if backup.existed_before:
            if backup.original_content is not None:
                backup.txt_path.write_text(backup.original_content, encoding="utf-8")
            continue

        if backup.txt_path.exists():
            backup.txt_path.unlink()


def run_g2p_and_add_words(
    corpus_dir: Path,
    dictionary: str,
    src_root: Path,
    clean: bool,
) -> None:
    g2p_output = src_root / "audios" / "g2pped_oovs.txt"

    cmd_g2p = [
        "mfa",
        "g2p",
        str(corpus_dir),
        dictionary,
        str(g2p_output),
        "--dictionary_path",
        dictionary,
    ]
    if clean:
        cmd_g2p.append("--clean")

    print("[INFO] Running G2P command:", " ".join(cmd_g2p))
    subprocess.run(cmd_g2p, check=True)

    if g2p_output.exists():
        cmd_add = ["mfa", "model", "add_words", dictionary, str(g2p_output)]
        print("[INFO] Adding words to dictionary:", " ".join(cmd_add))
        subprocess.run(cmd_add, check=True)
        print(f"[INFO] OOV pronunciations saved to: {g2p_output}")


def run_mfa(
    corpus_dir: Path,
    output_dir: Path,
    dictionary: str,
    acoustic_model: str,
    num_jobs: int,
    clean: bool,
    single_speaker: bool,
) -> None:
    cmd = [
        "mfa",
        "align",
        str(corpus_dir),
        dictionary,
        acoustic_model,
        str(output_dir),
        "--num_jobs",
        str(num_jobs),
        "--clean",
        "--single_speaker",
    ]

    if clean:
        cmd.append("--clean")
    if single_speaker:
        cmd.append("--single_speaker")

    print("[INFO] Running MFA command:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def collect_textgrids(
    mfa_output_dir: Path,
    aligned_dir: Path,
    name_mapping: dict[str, str],
) -> tuple[int, int, list[str]]:
    aligned_dir.mkdir(parents=True, exist_ok=True)

    found = 0
    for textgrid_file in mfa_output_dir.rglob("*.TextGrid"):
        temp_base = textgrid_file.stem
        if temp_base not in name_mapping:
            continue
        canonical_name = name_mapping[temp_base]
        target = aligned_dir / f"{canonical_name}.TextGrid"
        shutil.copy2(textgrid_file, target)
        found += 1

    missing = sorted(set(name_mapping.values()) - {p.stem for p in aligned_dir.glob("*.TextGrid")})
    return len(name_mapping), found, missing


def collect_csvs(mfa_output_dir: Path, aligned_dir: Path) -> list[Path]:
    aligned_dir.mkdir(parents=True, exist_ok=True)

    copied_targets: list[Path] = []
    name_counts: dict[str, int] = {}
    csv_candidates = sorted(mfa_output_dir.rglob("*.csv"))

    for src_csv in csv_candidates:
        base_name = src_csv.name
        count = name_counts.get(base_name, 0)
        target_name = base_name if count == 0 else f"{src_csv.stem}_{count}{src_csv.suffix}"

        target_path = aligned_dir / target_name
        shutil.copy2(src_csv, target_path)
        copied_targets.append(target_path)
        name_counts[base_name] = count + 1

    return copied_targets


def main() -> int:
    args = parse_args()
    src_root = args.src.resolve()

    if not src_root.exists():
        print(f"[ERROR] src path not found: {src_root}", file=sys.stderr)
        return 1

    transcript_backups: list[TranscriptBackup] = []
    tmp_corpus: Path | None = None
    tmp_output: Path | None = None

    try:
        samples = load_samples(src_root)
        print(f"[INFO] Loaded {len(samples)} samples from metadata.csv")

        tmp_corpus, name_mapping, transcript_backups = prepare_temp_corpus(src_root, samples)
        tmp_output = src_root / "audios" / ".mfa_tmp_output"
        if tmp_output.exists():
            shutil.rmtree(tmp_output)
        tmp_output.mkdir(parents=True, exist_ok=True)

        if args.add_words:
            run_g2p_and_add_words(
                corpus_dir=tmp_corpus,
                dictionary=args.dictionary,
                src_root=src_root,
                clean=args.clean,
            )

        run_mfa(
            corpus_dir=tmp_corpus,
            output_dir=tmp_output,
            dictionary=args.dictionary,
            acoustic_model=args.acoustic_model,
            num_jobs=args.num_jobs,
            clean=args.clean,
            single_speaker=args.single_speaker,
        )

        aligned_dir = src_root / "audios" / "aligned"
        expected, found, missing = collect_textgrids(tmp_output, aligned_dir, name_mapping)
        print(f"[INFO] Saved TextGrid files to: {aligned_dir}")
        if found != expected:
            preview = missing[:20]
            suffix = " ..." if len(missing) > 20 else ""
            print(
                "[WARN] TextGrid export mismatch "
                f"(expected={expected}, got={found}). "
                f"Missing({len(missing)}): {preview}{suffix}"
            )

        copied_csvs = collect_csvs(tmp_output, aligned_dir)
        if copied_csvs:
            print(
                "[INFO] Saved MFA CSV files:",
                ", ".join(str(p.name) for p in copied_csvs),
            )
        else:
            print("[WARN] No CSV found in MFA output")

        if args.keep_temp:
            print(f"[INFO] Temporary corpus kept: {tmp_corpus}")
            print(f"[INFO] Temporary output kept: {tmp_output}")

    except Exception as exc:  # pragma: no cover - CLI-level protection
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    finally:
        if not args.keep_audio_txt and transcript_backups:
            cleanup_generated_audio_txt(transcript_backups)
            print("[INFO] Generated audio-side transcript txt files cleaned")

        if not args.keep_temp:
            if tmp_corpus is not None:
                shutil.rmtree(tmp_corpus, ignore_errors=True)
            if tmp_output is not None:
                shutil.rmtree(tmp_output, ignore_errors=True)
            print("[INFO] Temporary MFA folders cleaned")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())