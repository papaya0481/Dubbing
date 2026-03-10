"""
过滤 metadata.csv：
  - Utterance 词数 < 5 且 WER > 0.8 → Keep = False
  - 其余 → Keep = True
输出覆盖原文件（或指定 --output）
"""

import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str,
                        default="/data2/ruixin/datasets/MELD_raw/metadata.csv")
    parser.add_argument("--output", type=str, default=None,
                        help="输出路径，默认覆盖原文件")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    out_path = Path(args.output) if args.output else csv_path

    rows: list[dict] = []
    fieldnames: list[str] = []
    filtered = 0

    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if "Keep" not in fieldnames:
            fieldnames.append("Keep")
        for row in reader:
            utterance = str(row.get("Utterance", "")).strip()
            try:
                wer = float(row.get("WER", 0))
            except ValueError:
                wer = 0.0
            word_count = len(utterance.split())
            if word_count < 5 or wer > 0.8:
                row["Keep"] = "False"
                filtered += 1
            else:
                row["Keep"] = "True"
            rows.append(row)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    print(f"总样本数   : {total}")
    print(f"Keep=False : {filtered}  ({filtered/total*100:.1f}%)")
    print(f"Keep=True  : {total - filtered}  ({(total-filtered)/total*100:.1f}%)")
    print(f"已保存至   : {out_path}")


if __name__ == "__main__":
    main()
