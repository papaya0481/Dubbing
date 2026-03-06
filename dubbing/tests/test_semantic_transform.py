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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import tgt
import numpy as np


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
    from indextts.semantic_transform import SemanticTransformer

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
    from indextts.semantic_transform import SemanticTransformer

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
    from indextts.semantic_transform import SemanticTransformer

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
    from indextts.semantic_transform import SemanticTransformer

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
    from indextts.semantic_transform import SemanticTransformer

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
    from indextts.semantic_transform import SemanticTransformer

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
    端到端测试：
      1. 用 IndexTTS2Semantic 做一次正常推理（第一次 pass）
      2. MFA 对齐输出 wav
      3. SemanticTransformer 扭曲 S_infer
      4. 重新生成最终 wav

    需要真实模型文件与 MFA 环境，仅在对应机器上运行。
    """
    # ---- 路径配置（按实际环境修改）----
    CHECKPOINT_DIR = "/data2/ruixin/index-tts2/checkpoints"
    CFG_PATH       = f"{CHECKPOINT_DIR}/config.yaml"
    SPK_PROMPT     = "/data2/ruixin/ted-tts/AllInferenceResults/ESD/0001/Angry/0001_000351.wav"

    # target_textgrid：使用 test_output/ 里已有的 TextGrid
    target_root = "/data2/ruixin/datasets/MELD_gen_pairs/"
    target_base = "dialog"
    target_name = "dev_dia4_row2_r2"
    TARGET_TG_PATH = (
        Path(target_root) / target_base / "aligned" / f"{target_name}.TextGrid"
    )
    OUTPUT_PATH = str(project_dubbing_root.parent / "test_output" / "test_semantic_warp.wav")

    # read text from ost path
    ost_txt_path = f"/data2/ruixin/datasets/MELD_gen_pairs/{target_base}/ost/{target_name.replace('_r2', '_r1')}.txt"
    with open(ost_txt_path, "r") as f:
        raw_text = f.read().strip()

    # 将文本随机分成两段（在单词边界处切割）
    import random
    words = raw_text.split()
    if len(words) <= 1:
        TEXT = [raw_text, raw_text]
    else:
        split_idx = random.randint(1, len(words) - 1)
        TEXT = [" ".join(words[:split_idx]), " ".join(words[split_idx:])]
    print(f"[文本分割] 第1段: '{TEXT[0]}' | 第2段: '{TEXT[1]}'")
    

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

    # ---- 执行语义扭曲推理 ----
    emo_vector = [[0.0]*8, [0.0]*8]  # 示例情感向量，实际使用时应根据模型需求构造
    emo_vector[0][0] = 1.0  # 假设第一个维度代表某种情感（如愤怒），这里设置为 1.0 以激活该情感
    emo_vector[1][1] = 1.0  # 假设第二个维度代表另一种情感（如快乐），这里设置为 1.0 以激活该情感
    result = model.infer_with_semantic_warp(
        spk_audio_prompt=SPK_PROMPT,
        text=TEXT,
        output_path=OUTPUT_PATH,
        target_textgrid=str(TARGET_TG_PATH),
        tier_name="phones",
        emo_audio_prompt=SPK_PROMPT,
        emo_vector=emo_vector,
        verbose=True,
    )

    print(f"\n[集成测试结果] output_path={result[0]}, wav_length={result[2]:.2f}s")

    # 验证输出文件存在
    assert os.path.isfile(result[0]), f"输出文件不存在: {result[0]}"

    # 验证输出 wav 时长接近目标 TextGrid 时长
    tg_tgt = tgt.io.read_textgrid(str(TARGET_TG_PATH))
    tgt_duration = tg_tgt.get_tier_by_name("phones").end_time
    assert abs(result[2] - tgt_duration) < 0.5, (
        f"输出时长 {result[2]:.2f}s 偏离目标 {tgt_duration:.2f}s 过多"
    )
    print(f"[PASS] 集成测试通过，输出时长={result[2]:.2f}s，目标={tgt_duration:.2f}s")


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
    from indextts.semantic_transform import SemanticTransformer

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
    
    test_integration_infer_with_semantic_warp()
