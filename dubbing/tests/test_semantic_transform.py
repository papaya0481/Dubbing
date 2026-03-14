"""
测试 SemanticTransformer 与 IndexTTS2Semantic
================================================

包含两层测试：

1. 单元测试（不需要真实模型）：
   - SemanticTransformer.warp() 的向量化正确性
   - SemanticTransformer.transform() 的形状与时长推导
   - 锚点构建与 PCHIP warping path 计算

2. 集成测试（需要真实模型 + MFA + GPU）：
   - IndexTTS2Semantic.infer_with_semantic_warp() 端到端流程

参考：dubbing/tests/test_mel_transfrom.py
"""

import sys
from pathlib import Path

# 确保 dubbing 根目录在 sys.path 中
project_dubbing_root = Path(__file__).resolve().parents[1]
if str(project_dubbing_root) not in sys.path:
    sys.path.insert(0, str(project_dubbing_root))

import os

import torch
import tgt
import numpy as np
import pytest


# ============================================================
# 工具：构造合成 TextGrid
# ============================================================

def make_synthetic_textgrid(
    word_intervals: list[tuple[float, float, str]],
    phone_intervals: list[tuple[float, float, str]],
    end_time: float,
) -> tgt.TextGrid:
    """根据给定的词/音素区间构造 tgt.TextGrid 对象。

    显式用 end_time 作为 tier 的尾部时间，确保 tier.end_time 正确。
    """
    tg = tgt.TextGrid()

    # 指定 end_time 确保 tier.end_time 返回正确值
    words = tgt.IntervalTier(start_time=0.0, end_time=end_time, name="words")
    for start, end, text in word_intervals:
        words.add_interval(tgt.Interval(start, end, text))
    tg.add_tier(words)

    phones = tgt.IntervalTier(start_time=0.0, end_time=end_time, name="phones")
    for start, end, text in phone_intervals:
        phones.add_interval(tgt.Interval(start, end, text))
    tg.add_tier(phones)

    return tg


# ============================================================
# 1. 单元测试：SemanticTransformer（无真实模型）
# ============================================================

def test_warp_shape():
    """验证 warp() 输出 shape 正确。"""
    from dubbing.modules.semantic_stretch.semantic_transform import SemanticTransformer

    transformer = SemanticTransformer(device="cpu", verbose=True)

    B, T_src, D = 1, 80, 512
    s_input = torch.randn(B, T_src, D)
    # 线性映射：目标帧 i → 源帧 i * 0.8（压缩）
    T_tgt = 100
    warping_path = torch.linspace(0, T_src - 1, T_tgt)
    out = transformer.warp(s_input, warping_path)

    assert out.shape == (B, T_tgt, D), f"期望 {(B, T_tgt, D)}, 实际 {out.shape}"
    print(f"[PASS] warp shape: {out.shape}")


def test_warp_identity():
    """恒等映射：warp(s, identity_path) ≈ s（双线性插值误差）。"""
    from dubbing.modules.semantic_stretch.semantic_transform import SemanticTransformer

    transformer = SemanticTransformer(device="cpu")

    B, T, D = 1, 60, 128
    s = torch.randn(B, T, D)
    identity_path = torch.linspace(0, T - 1, T)
    out = transformer.warp(s, identity_path)

    assert out.shape == s.shape
    max_err = (out - s).abs().max().item()
    assert max_err < 1e-4, f"恒等映射误差过大：{max_err}"
    print(f"[PASS] warp identity, max_err={max_err:.2e}")


def test_silence_mask():
    """静音 mask：对应帧应被置零。"""
    from dubbing.modules.semantic_stretch.semantic_transform import SemanticTransformer

    transformer = SemanticTransformer(device="cpu")

    B, T_src, D = 1, 50, 64
    s = torch.ones(B, T_src, D)
    T_tgt = 60
    warping_path = torch.linspace(0, T_src - 1, T_tgt)
    # 标记前 10 帧为静音
    silence_mask = torch.zeros(T_tgt, dtype=torch.bool)
    silence_mask[:10] = True

    out = transformer.warp(s, warping_path, silence_mask=silence_mask)
    assert out[:, :10, :].abs().max().item() == 0.0, "静音帧未被置零"
    assert out[:, 10:, :].abs().max().item() > 0.0, "非静音帧不应为零"
    print("[PASS] silence_mask 置零正确")


def test_transform_shape_and_duration():
    """transform() 输出形状与 tgt_duration 应与目标 TextGrid 一致。"""
    from dubbing.modules.semantic_stretch.semantic_transform import SemanticTransformer

    transformer = SemanticTransformer(device="cpu", verbose=True)

    # 构造 source TextGrid（模拟 TTS 输出对齐，时长 2.0s）
    src_tg = make_synthetic_textgrid(
        word_intervals=[(0.1, 0.5, "HELLO"), (0.6, 1.5, "WORLD")],
        phone_intervals=[
            (0.1, 0.3, "HH"), (0.3, 0.5, "AH"),
            (0.6, 0.9, "W"), (0.9, 1.2, "ER"), (1.2, 1.5, "L"),
        ],
        end_time=2.0,
    )

    # 构造 target TextGrid（用户提供，时长 2.5s，略有不同）
    tgt_tg = make_synthetic_textgrid(
        word_intervals=[(0.1, 0.6, "HELLO"), (0.7, 2.0, "WORLD")],
        phone_intervals=[
            (0.1, 0.35, "HH"), (0.35, 0.6, "AH"),
            (0.7, 1.0, "W"), (1.0, 1.5, "ER"), (1.5, 2.0, "L"),
        ],
        end_time=2.5,
    )

    # S_infer 的 T_src = src_duration * 50 = 2.0 * 50 = 100
    B, T_src, D = 1, 100, 512
    s_infer = torch.randn(B, T_src, D)

    warped, tgt_duration = transformer.transform(
        s_infer=s_infer,
        source_textgrid=src_tg,
        target_textgrid=tgt_tg,
        tier_name="phones",
    )

    expected_T_tgt = int(round(2.5 * 50))  # 125
    assert warped.shape == (B, expected_T_tgt, D), (
        f"期望 shape {(B, expected_T_tgt, D)}, 实际 {warped.shape}"
    )
    assert abs(tgt_duration - 2.5) < 1e-6, f"tgt_duration 错误: {tgt_duration}"
    print(f"[PASS] transform shape={warped.shape}, tgt_duration={tgt_duration:.3f}s")


def test_transform_word_tier():
    """使用 words tier 时，transform() 应正常工作。"""
    from dubbing.modules.semantic_stretch.semantic_transform import SemanticTransformer

    transformer = SemanticTransformer(device="cpu")

    src_tg = make_synthetic_textgrid(
        word_intervals=[(0.1, 0.8, "HELLO"), (0.9, 2.0, "WORLD")],
        phone_intervals=[],
        end_time=2.0,
    )
    tgt_tg = make_synthetic_textgrid(
        word_intervals=[(0.2, 1.0, "HELLO"), (1.1, 2.4, "WORLD")],
        phone_intervals=[],
        end_time=2.4,
    )

    B, T_src, D = 1, 100, 256
    s_infer = torch.randn(B, T_src, D)

    warped, tgt_duration = transformer.transform(
        s_infer=s_infer,
        source_textgrid=src_tg,
        target_textgrid=tgt_tg,
        tier_name="words",
    )
    assert warped.ndim == 3 and warped.shape[0] == 1
    print(f"[PASS] words tier transform shape={warped.shape}")


def test_vectorized_no_for_loop():
    """简单验证：大 batch 下 warp() 返回正确 batch size。"""
    from dubbing.modules.semantic_stretch.semantic_transform import SemanticTransformer

    transformer = SemanticTransformer(device="cpu")

    B, T_src, D = 4, 60, 128
    T_tgt = 80
    s = torch.randn(B, T_src, D)
    wp = torch.linspace(0, T_src - 1, T_tgt)
    out = transformer.warp(s, wp)
    assert out.shape == (B, T_tgt, D)
    print(f"[PASS] batch warp shape={out.shape}")


# ============================================================
# 2. 集成测试：IndexTTS2Semantic（需要真实模型）
# ============================================================

def test_integration_infer_with_semantic_warp():
    """
        端到端批量测试（前 20 条）：
            1. 从 /data2/ruixin/datasets/MELD_raw/audios/ost 读取前 20 个样本
            2. 按 metadata.csv 优先、txt 回退方式获取文本/情感
            3. 调用 infer_with_semantic_warp 做推理 + warp
            4. 每条样本保存为 {name}_mid.wav 与 {name}_final.wav

    需要真实模型文件与 MFA 环境，仅在对应机器上运行。
    """
    # ---- 路径配置（按实际环境修改）----
    CHECKPOINT_DIR = "/data2/ruixin/index-tts2/checkpoints"
    CFG_PATH       = f"{CHECKPOINT_DIR}/config.yaml"
    target_root = Path("/data2/ruixin/datasets/MELD_raw/audios")
    ost_dir = target_root / "ost"
    aligned_dir = target_root / "aligned"
    output_dir = Path("/data2/ruixin/ours/test_outputs/semantic_warp_integration")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip when required resources are absent
    _required = [
        Path(CHECKPOINT_DIR),
        Path(CFG_PATH),
        ost_dir,
        aligned_dir,
    ]
    if not all(p.exists() for p in _required):
        pytest.skip(
            "Skipping integration test: one or more required paths not found: "
            + str([str(p) for p in _required if not p.exists()])
        )

    import csv
    import re

    metadata_map = {}
    metadata_csv = aligned_dir / "generation_metadata.csv"
    if metadata_csv.exists():
        with metadata_csv.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                key = (str(row.get("sample_key", "")).strip(), str(row.get("repeat_idx", "")).strip())
                metadata_map[key] = row

    EMOTION_DIMS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]
    ALIASES = {
        "joy": "happy", "anger": "angry", "sadness": "sad", "fear": "afraid",
        "fearful": "afraid", "disgust": "disgusted", "surprise": "surprised", "neutral": "calm",
    }

    def _emo_vec(emo: str) -> list[float]:
        emo = ALIASES.get(emo.lower().strip(), emo.lower().strip())
        emo = emo if emo in EMOTION_DIMS else "calm"
        vec = [0.0] * len(EMOTION_DIMS)
        vec[EMOTION_DIMS.index(emo)] = 1.0
        return vec

    def _load_text_and_emo(name: str) -> tuple[list[str], list[list[float]]]:
        m = re.match(r"^(.+)_r(\d+)$", name)
        sample_key = m.group(1) if m else name
        repeat_idx = m.group(2) if m else "1"

        row = metadata_map.get((sample_key, repeat_idx))
        if row is not None:
            text_raw = str(row.get("text", "")).strip()
            if text_raw:
                text_parts = [x.strip() for x in text_raw.split("|") if x.strip()]
                if not text_parts:
                    text_parts = [text_raw]
                emo_labels = [x.strip() for x in str(row.get("emo_text", "")).split("|") if x.strip()]
                if not emo_labels:
                    emo_labels = ["calm"] * len(text_parts)
                if len(emo_labels) < len(text_parts):
                    emo_labels.extend([emo_labels[-1]] * (len(text_parts) - len(emo_labels)))
                emo = [_emo_vec(e) for e in emo_labels[:len(text_parts)]]
                return text_parts, emo

        txt_path = ost_dir / f"{name}.txt"
        if txt_path.exists():
            raw_text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        else:
            raw_text = ""

        if not raw_text:
            text_parts = [name]
        else:
            words = raw_text.split()
            if len(words) <= 1:
                text_parts = [raw_text]
            else:
                split_idx = max(1, len(words) // 2)
                text_parts = [" ".join(words[:split_idx]), " ".join(words[split_idx:])]

        emo = [_emo_vec("happy" if i % 2 == 0 else "angry") for i in range(len(text_parts))]
        return text_parts, emo
    

    # ---- 初始化 IndexTTS2Semantic ----
    from indextts.infer_semantic import IndexTTS2Semantic
    model = IndexTTS2Semantic(
        cfg_path=CFG_PATH,
        model_dir=CHECKPOINT_DIR,
        is_fp16=False,
        verbose_transform=True,
    )
    
    # ---- 初始化 MFAAligner ----
    from modules.mfa_alinger import MFAAligner
    aligner = MFAAligner(
        acoustic_model="english_us_arpa",
        dictionary_model="english_us_arpa",
    )
    
    model.mfa_aligner = aligner

    wav_paths = sorted(ost_dir.glob("*.wav"))[:20]
    if not wav_paths:
        pytest.skip(f"No wav files found in ost dir: {ost_dir}")

    processed = 0
    for wav_path in wav_paths:
        name = wav_path.stem
        tg_path = aligned_dir / f"{name}.TextGrid"
        if not tg_path.exists():
            tg_path = aligned_dir / f"{name}.textgrid"
        if not tg_path.exists():
            print(f"[SKIP] {name}: target textgrid not found")
            continue

        text, emo_vector = _load_text_and_emo(name)

        final_out_path = output_dir / f"{name}_final.wav"
        result = model.infer_with_semantic_warp(
            spk_audio_prompt=str(wav_path),
            text=text,
            output_path=str(final_out_path),
            target_textgrid=str(tg_path),
            tier_name="phones",
            emo_audio_prompt=str(wav_path),
            emo_vector=emo_vector,
            verbose=True,
            save_mid=True,
        )

        # infer_semantic.py 内部 save_mid 使用 <output_stem>_mid.wav，
        # 这里统一重命名为 {name}_mid.wav。
        raw_mid = final_out_path.with_stem(final_out_path.stem + "_mid")
        renamed_mid = output_dir / f"{name}_mid.wav"
        if raw_mid.exists():
            if renamed_mid.exists():
                renamed_mid.unlink()
            raw_mid.replace(renamed_mid)

        assert final_out_path.exists(), f"输出文件不存在: {final_out_path}"
        assert renamed_mid.exists(), f"中间态文件不存在: {renamed_mid}"

        tg_tgt = tgt.io.read_textgrid(str(tg_path))
        tgt_duration = tg_tgt.get_tier_by_name("phones").end_time
        assert abs(result[2] - tgt_duration) < 0.5, (
            f"[{name}] 输出时长 {result[2]:.2f}s 偏离目标 {tgt_duration:.2f}s 过多"
        )

        print(
            f"[PASS] {name}: mid={renamed_mid.name}, final={final_out_path.name}, "
            f"wav={result[2]:.2f}s, tgt={tgt_duration:.2f}s"
        )
        processed += 1

    assert processed > 0, "没有成功处理任何样本，请检查输入数据与对齐文件"
    print(f"[DONE] 共处理 {processed} 条样本，输出目录: {output_dir}")


def test_integration_infer_with_cond_warp():
    """
    端到端批量测试（前 20 条，cond-first 路径）：
      1. 从 /data2/ruixin/datasets/MELD_raw/audios/ost 读取前 20 个样本
      2. 按 metadata.csv 优先、txt 回退方式获取文本/情感
      3. 调用 infer_with_cond_warp（先 LR 得 cond，再对 cond 做 warp）
      4. 每条样本保存为 {name}_mid.wav 与 {name}_final.wav

    需要真实模型文件与 MFA 环境，仅在对应机器上运行。
    """
    CHECKPOINT_DIR = "/data2/ruixin/index-tts2/checkpoints"
    CFG_PATH = f"{CHECKPOINT_DIR}/config.yaml"
    target_root = Path("/data2/ruixin/datasets/MELD_raw/audios")
    ost_dir = target_root / "ost"
    aligned_dir = target_root / "aligned"
    output_dir = Path("/data2/ruixin/ours/test_outputs/semantic_warp_integration_cond_first")
    output_dir.mkdir(parents=True, exist_ok=True)

    _required = [
        Path(CHECKPOINT_DIR),
        Path(CFG_PATH),
        ost_dir,
        aligned_dir,
    ]
    if not all(p.exists() for p in _required):
        pytest.skip(
            "Skipping integration test: one or more required paths not found: "
            + str([str(p) for p in _required if not p.exists()])
        )

    import csv
    import re

    metadata_map = {}
    metadata_csv = aligned_dir / "generation_metadata.csv"
    if metadata_csv.exists():
        with metadata_csv.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                key = (str(row.get("sample_key", "")).strip(), str(row.get("repeat_idx", "")).strip())
                metadata_map[key] = row

    EMOTION_DIMS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]
    ALIASES = {
        "joy": "happy", "anger": "angry", "sadness": "sad", "fear": "afraid",
        "fearful": "afraid", "disgust": "disgusted", "surprise": "surprised", "neutral": "calm",
    }

    def _emo_vec(emo: str) -> list[float]:
        emo = ALIASES.get(emo.lower().strip(), emo.lower().strip())
        emo = emo if emo in EMOTION_DIMS else "calm"
        vec = [0.0] * len(EMOTION_DIMS)
        vec[EMOTION_DIMS.index(emo)] = 1.0
        return vec

    def _load_text_and_emo(name: str) -> tuple[list[str], list[list[float]]]:
        m = re.match(r"^(.+)_r(\d+)$", name)
        sample_key = m.group(1) if m else name
        repeat_idx = m.group(2) if m else "1"

        row = metadata_map.get((sample_key, repeat_idx))
        if row is not None:
            text_raw = str(row.get("text", "")).strip()
            if text_raw:
                text_parts = [x.strip() for x in text_raw.split("|") if x.strip()]
                if not text_parts:
                    text_parts = [text_raw]
                emo_labels = [x.strip() for x in str(row.get("emo_text", "")).split("|") if x.strip()]
                if not emo_labels:
                    emo_labels = ["calm"] * len(text_parts)
                if len(emo_labels) < len(text_parts):
                    emo_labels.extend([emo_labels[-1]] * (len(text_parts) - len(emo_labels)))
                emo = [_emo_vec(e) for e in emo_labels[:len(text_parts)]]
                return text_parts, emo

        txt_path = ost_dir / f"{name}.txt"
        if txt_path.exists():
            raw_text = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
        else:
            raw_text = ""

        if not raw_text:
            text_parts = [name]
        else:
            words = raw_text.split()
            if len(words) <= 1:
                text_parts = [raw_text]
            else:
                split_idx = max(1, len(words) // 2)
                text_parts = [" ".join(words[:split_idx]), " ".join(words[split_idx:])]

        emo = [_emo_vec("happy" if i % 2 == 0 else "angry") for i in range(len(text_parts))]
        return text_parts, emo

    from indextts.infer_semantic import IndexTTS2Semantic
    model = IndexTTS2Semantic(
        cfg_path=CFG_PATH,
        model_dir=CHECKPOINT_DIR,
        is_fp16=False,
        verbose_transform=True,
    )

    from modules.mfa_alinger import MFAAligner
    aligner = MFAAligner(
        acoustic_model="english_us_arpa",
        dictionary_model="english_us_arpa",
    )
    model.mfa_aligner = aligner

    wav_paths = sorted(ost_dir.glob("*.wav"))[:20]
    if not wav_paths:
        pytest.skip(f"No wav files found in ost dir: {ost_dir}")

    processed = 0
    for wav_path in wav_paths:
        name = wav_path.stem
        tg_path = aligned_dir / f"{name}.TextGrid"
        if not tg_path.exists():
            tg_path = aligned_dir / f"{name}.textgrid"
        if not tg_path.exists():
            print(f"[SKIP] {name}: target textgrid not found")
            continue

        text, emo_vector = _load_text_and_emo(name)

        final_out_path = output_dir / f"{name}_final.wav"
        result = model.infer_with_cond_warp(
            spk_audio_prompt=str(wav_path),
            text=text,
            output_path=str(final_out_path),
            target_textgrid=str(tg_path),
            tier_name="phones",
            emo_audio_prompt=str(wav_path),
            emo_vector=emo_vector,
            verbose=True,
            save_mid=True,
        )

        raw_mid = final_out_path.with_stem(final_out_path.stem + "_mid")
        renamed_mid = output_dir / f"{name}_mid.wav"
        if raw_mid.exists():
            if renamed_mid.exists():
                renamed_mid.unlink()
            raw_mid.replace(renamed_mid)

        assert final_out_path.exists(), f"输出文件不存在: {final_out_path}"
        assert renamed_mid.exists(), f"中间态文件不存在: {renamed_mid}"

        tg_tgt = tgt.io.read_textgrid(str(tg_path))
        tgt_duration = tg_tgt.get_tier_by_name("phones").end_time
        assert abs(result[2] - tgt_duration) < 0.5, (
            f"[{name}] 输出时长 {result[2]:.2f}s 偏离目标 {tgt_duration:.2f}s 过多"
        )

        print(
            f"[PASS-cond] {name}: mid={renamed_mid.name}, final={final_out_path.name}, "
            f"wav={result[2]:.2f}s, tgt={tgt_duration:.2f}s"
        )
        processed += 1

    assert processed > 0, "没有成功处理任何样本，请检查输入数据与对齐文件"
    print(f"[DONE-cond] 共处理 {processed} 条样本，输出目录: {output_dir}")


# ============================================================
# 可视化：比较 S_infer 拉伸前后（参考 test_mel_transfrom.py）
# ============================================================

def visualize_warp_comparison(
    s_infer: torch.Tensor,      # (1, T_src, D)
    s_warped: torch.Tensor,     # (1, T_tgt, D)
    save_path: str = "test_semantic_warp_comparison.png",
):
    """将 S_infer 与 S_warped 的 L2 norm 曲线对比可视化。"""
    import matplotlib.pyplot as plt

    def _norm_curve(s: torch.Tensor) -> np.ndarray:
        # (1, T, D) → (T,)
        return s.squeeze(0).detach().cpu().norm(dim=-1).numpy()

    norm_src = _norm_curve(s_infer)
    norm_tgt = _norm_curve(s_warped)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=False)
    axes[0].plot(norm_src, color="steelblue", linewidth=0.8)
    axes[0].set_title("S_infer (before warp) — L2 norm per frame")
    axes[0].set_xlabel("Code Frame Index")
    axes[0].set_ylabel("L2 norm")

    axes[1].plot(norm_tgt, color="tomato", linewidth=0.8)
    axes[1].set_title("S_warped (after warp) — L2 norm per frame")
    axes[1].set_xlabel("Code Frame Index")
    axes[1].set_ylabel("L2 norm")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[可视化] 保存至 {save_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("单元测试（无需真实模型）")
    print("=" * 60)
    test_warp_shape()
    test_warp_identity()
    test_silence_mask()
    test_transform_shape_and_duration()
    test_transform_word_tier()
    test_vectorized_no_for_loop()
    print("\n所有单元测试通过！\n")

    # ---- 可视化演示（用合成数据）----
    print("=" * 60)
    print("可视化演示（合成数据）")
    print("=" * 60)
    from dubbing.modules.semantic_stretch.semantic_transform import SemanticTransformer

    transformer = SemanticTransformer(device="cpu", verbose=True)

    src_tg = make_synthetic_textgrid(
        word_intervals=[(0.1, 0.5, "HELLO"), (0.6, 1.5, "WORLD")],
        phone_intervals=[
            (0.1, 0.3, "HH"), (0.3, 0.5, "AH"),
            (0.6, 0.9, "W"), (0.9, 1.2, "ER"), (1.2, 1.5, "L"),
        ],
        end_time=2.0,
    )
    tgt_tg = make_synthetic_textgrid(
        word_intervals=[(0.1, 0.6, "HELLO"), (0.7, 2.0, "WORLD")],
        phone_intervals=[
            (0.1, 0.35, "HH"), (0.35, 0.6, "AH"),
            (0.7, 1.0, "W"), (1.0, 1.5, "ER"), (1.5, 2.0, "L"),
        ],
        end_time=2.5,
    )
    s_demo = torch.randn(1, 100, 512)
    s_w, _ = transformer.transform(s_demo, src_tg, tgt_tg, tier_name="phones")
    visualize_warp_comparison(s_demo, s_w, save_path="test_semantic_warp_comparison.png")

    # ---- 集成测试（可选，注释掉以跳过）----
    print("=" * 60)
    print("集成测试（需要真实模型）")
    print("=" * 60)

    # test_integration_infer_with_semantic_warp()
    test_integration_infer_with_cond_warp()
