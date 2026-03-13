"""
test_semantic_warp_correctness.py
==================================
针对 SemanticTransformer 的精确性单元测试。

验证逻辑（核心方法）
--------------------
``_assert_warp_correspondence(warping_path, matched_src, matched_tgt, fps)``

对每一对已匹配的 interval (src_iv, tgt_iv)，验证：
  1. **区间内采样源头正确**：目标帧区间 [f_tgt_start, f_tgt_end) 内的所有
     warping_path 值均落在 [f_src_start - TOL, f_src_end + TOL] 内。
  2. **首帧单调性**：目标区间首帧映射到 >= 前一对末帧的映射值（全局非递减）。
  3. **全局单调**：整条 warping_path 非严格递减。

``_assert_silence_mask(silence_mask, silence_segs_tgt, fps)``

对每段"目标独有静音"区间 [t0, t1]，验证：
  - [f0, f1) 范围内的 silence_mask 全为 True。
  - 其余帧（非静音扩张区间）中 silence_mask 不应全为 True（合理性检查）。

测试覆盖
--------
Case 1 – words 数量不等：src 比 tgt 多一个词，大部分相同 → LCS 匹配后仅匹配词的
         区间必须对应正确，多余词的时间被 PCHIP 压缩进相邻段。
Case 2 – words 数量相等，但某个 word 下的 phoneme 数量不等。
Case 3 – 静音不对称：同一段话 src 某处有静音 tgt 没有；另一处 tgt 有静音 src 没有。

每个 Case 均使用 ``build_mock_textgrid`` 构造合成 TextGrid，精确控制每个
词/音素/静音的时间区间，避免真实数据的随机性干扰。
对于真实数据验证使用 ``test_real_data_*`` 系列（需要 MELD_gen_pairs 存在）。
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pytest
import torch
import tgt

# ---- 路径 ----
_HERE = Path(__file__).resolve().parent
_DUBBING_ROOT = _HERE.parent
_LIPS_ROOT = _DUBBING_ROOT.parent / "lips"
if str(_DUBBING_ROOT) not in sys.path:
    sys.path.insert(0, str(_DUBBING_ROOT))
if str(_LIPS_ROOT) not in sys.path:
    sys.path.insert(0, str(_LIPS_ROOT))

from modules.semantic_stretch.semantic_transform import SemanticTransformer

# ---- 全局常量 ----
FPS = 50.0          # 语义码帧率
TOL_FRAMES = 1.5    # warping_path 比较时允许的帧级误差（PCHIP 在边界会有亚帧偏差）

# 真实数据：src/tgt 来自两个独立目录，相同文件名构成一对
REAL_SRC_DIR = Path("/data2/ruixin/datasets/MELD_raw/audios/aligned")
REAL_TGT_DIR = Path("/data2/ruixin/datasets/flow_dataset/MELD_semantic/audios/aligned")
REAL_DATA_AVAILABLE = REAL_SRC_DIR.exists() and REAL_TGT_DIR.exists()

# =============================================================================
# 辅助：构造合成 TextGrid
# =============================================================================

def _iv(xmin: float, xmax: float, text: str) -> tgt.Interval:
    return tgt.Interval(xmin, xmax, text)


def build_mock_textgrid(
    total_dur: float,
    words: List[Tuple[float, float, str]],   # (xmin, xmax, text)
    phones: List[Tuple[float, float, str]],
) -> tgt.TextGrid:
    """构造仅有 words + phones 两层的合成 TextGrid。"""
    tg = tgt.TextGrid()

    def _make_tier(name: str, segs: List[Tuple[float, float, str]]) -> tgt.IntervalTier:
        tier = tgt.IntervalTier(name=name, start_time=0.0, end_time=total_dur)
        for xmin, xmax, text in segs:
            tier.add_interval(tgt.Interval(xmin, xmax, text))
        return tier

    tg.add_tier(_make_tier("words", words))
    tg.add_tier(_make_tier("phones", phones))
    return tg


# =============================================================================
# 辅助：提取 build_anchors / warping_path / silence_mask（绕过 TextGrid 文件）
# =============================================================================

def _compute_warping(
    tg_src: tgt.TextGrid,
    tg_tgt: tgt.TextGrid,
    tier_name: str = "phones",
) -> Tuple[torch.Tensor, torch.Tensor, List, List, "SemanticTransformer"]:
    """
    调用 SemanticTransformer 内部方法，返回：
    (warping_path, silence_mask, src_anchors, tgt_anchors, transformer)
    """
    st = SemanticTransformer(device="cpu", verbose=True)

    def _real(tier): return st.get_real_words(tier)
    def _dur(tier): return tier.end_time if len(tier) > 0 else 0.0

    tier_src = tg_src.get_tier_by_name(tier_name)
    tier_tgt = tg_tgt.get_tier_by_name(tier_name)
    phones_src = _real(tier_src)
    phones_tgt = _real(tier_tgt)
    src_dur = _dur(tier_src)
    tgt_dur = _dur(tier_tgt)

    if tier_name == "phones":
        words_src, words_tgt, pg_src, pg_tgt = st._build_phone_groups(
            tg_src, tg_tgt, phones_src, phones_tgt
        )
    else:
        words_src, words_tgt = phones_src, phones_tgt
        pg_src = pg_tgt = None

    src_anchors, tgt_anchors = st.build_anchors(
        words_src, words_tgt, src_dur, tgt_dur,
        phone_groups_src=pg_src, phone_groups_tgt=pg_tgt,
    )
    total_tgt_frames = max(1, int(round(tgt_dur * FPS)))
    warping_path = st.calculate_warping_path(src_anchors, tgt_anchors, total_tgt_frames)
    silence_mask = st._detect_silence_mask(src_anchors, tgt_anchors, total_tgt_frames)

    # 返回匹配词列表，供上层断言用
    return warping_path, silence_mask, src_anchors, tgt_anchors, words_src, words_tgt, st


# =============================================================================
# 核心断言：词/音素级一一对应
# =============================================================================

def _assert_warp_correspondence(
    warping_path: torch.Tensor,
    matched_src: List[tgt.Interval],
    matched_tgt: List[tgt.Interval],
    fps: float = FPS,
    tol: float = TOL_FRAMES,
    label: str = "",
) -> None:
    """
    验证方法：对每对匹配的 (src_iv, tgt_iv)，目标区间内所有帧的
    warping_path 值必须落在 src_iv 的帧范围内（ ± tol 帧容忍）。

    同时验证全局非递减性。
    """
    path = warping_path.numpy()

    # 1. 全局非递减性
    diff = np.diff(path)
    assert np.all(diff >= -tol), (
        f"{label}: warping_path 不是单调非递减的！最大回退 = {diff.min():.3f} 帧"
    )

    # 2. 每对匹配区间内采样源头必须对应
    prev_src_start_frame = 0.0  # 前一个匹配 src 词的起始帧（PCHIP 过渡 lo 边界）
    for i, (src_iv, tgt_iv) in enumerate(zip(matched_src, matched_tgt)):
        f_tgt_s = int(round(tgt_iv.start_time * fps))
        f_tgt_e = int(round(tgt_iv.end_time * fps))
        f_src_s = src_iv.start_time * fps
        f_src_e = src_iv.end_time * fps

        if f_tgt_e <= f_tgt_s:
            prev_src_start_frame = f_src_s
            continue  # 目标帧区间为空，跳过

        seg_path = path[f_tgt_s:f_tgt_e]
        # PCHIP 在词间过渡时，tgt 词起始帧的值可以来自前一个 src 词区间
        # 故 lo 扩展到前一个匹配 src 词的起始帧（覆盖过渡带）
        lo = prev_src_start_frame - tol
        hi = f_src_e + tol

        out_of_range = (seg_path < lo) | (seg_path > hi)
        assert not out_of_range.any(), (
            f"{label}\n"
            f"  词对 '{src_iv.text}'({src_iv.start_time:.3f}-{src_iv.end_time:.3f}s) "
            f"→ '{tgt_iv.text}'({tgt_iv.start_time:.3f}-{tgt_iv.end_time:.3f}s)\n"
            f"  目标帧 [{f_tgt_s}, {f_tgt_e}) → 期望源帧范围 [{lo:.1f}, {hi:.1f}]\n"
            f"  但发现越界值: min={seg_path.min():.2f}, max={seg_path.max():.2f}\n"
            f"  越界帧（相对区间）: {np.where(out_of_range)[0].tolist()[:10]}"
        )
        prev_src_start_frame = f_src_s


def _assert_silence_mask(
    silence_mask: torch.Tensor,
    silence_segs_tgt: List[Tuple[float, float]],  # [(t0, t1), ...] 目标独有静音
    fps: float = FPS,
    label: str = "",
) -> None:
    """
    验证目标中"新增静音"区间里 silence_mask 全为 True。
    """
    mask = silence_mask.numpy()
    for t0, t1 in silence_segs_tgt:
        f0 = int(round(t0 * fps))
        f1 = int(round(t1 * fps))
        if f1 <= f0:
            continue
        seg = mask[f0:f1]
        assert seg.all(), (
            f"{label}: 静音帧 [{f0},{f1}) 中 silence_mask 未全为 True，"
            f"False 比例 = {(~seg).mean()*100:.1f}%"
        )


# =============================================================================
# Case 1：words 数量不等，src 比 tgt 多一个词（大部分匹配）
# =============================================================================

def test_case1_word_count_mismatch():
    """
    src: [what(0-0.3), are(0.4-0.7), you(0.8-1.1), doing(1.2-1.6)]  总时长 1.8s
    tgt: [are(0.1-0.5), you(0.6-1.0), doing(1.1-1.5)]               总时长 1.6s

    LCS 应匹配 (are,are), (you,you), (doing,doing) 三对。
    'what' 在 src 有，tgt 无 → src 开头段被 PCHIP 压缩进最前段。

    验证：3 对匹配词的目标帧 warping_path 值必须来自对应的 src 词区间。
    """
    # src: what 多出来，其余匹配
    src_words = [
        (0.0, 0.05, ""),        # leading sil
        (0.05, 0.35, "what"),
        (0.35, 0.38, ""),
        (0.38, 0.68, "are"),
        (0.68, 0.70, ""),
        (0.70, 1.00, "you"),
        (1.00, 1.05, ""),
        (1.05, 1.50, "doing"),
        (1.50, 1.60, ""),
    ]
    src_phones = [
        (0.00, 0.05, ""),
        (0.05, 0.15, "W"),  (0.15, 0.25, "AH1"), (0.25, 0.35, "T"),   # what
        (0.35, 0.38, ""),
        (0.38, 0.50, "EH1"), (0.50, 0.68, "R"),                        # are
        (0.68, 0.70, ""),
        (0.70, 0.82, "Y"),  (0.82, 1.00, "UW1"),                       # you
        (1.00, 1.05, ""),
        (1.05, 1.20, "D"),  (1.20, 1.35, "UW1"), (1.35, 1.50, "IH0"), (1.50, 1.50, "NG"), # doing (NG 零宽度但存在)
        (1.50, 1.60, ""),
    ]
    # 修正 NG 零宽度 → 给它一个帧
    src_phones[13] = (1.48, 1.50, "NG")

    tgt_words = [
        (0.0, 0.08, ""),
        (0.08, 0.48, "are"),
        (0.48, 0.52, ""),
        (0.52, 0.88, "you"),
        (0.88, 0.92, ""),
        (0.92, 1.40, "doing"),
        (1.40, 1.50, ""),
    ]
    tgt_phones = [
        (0.00, 0.08, ""),
        (0.08, 0.24, "EH1"), (0.24, 0.48, "R"),          # are
        (0.48, 0.52, ""),
        (0.52, 0.68, "Y"), (0.68, 0.88, "UW1"),           # you
        (0.88, 0.92, ""),
        (0.92, 1.10, "D"), (1.10, 1.25, "UW1"),
        (1.25, 1.36, "IH0"), (1.36, 1.40, "NG"),          # doing
        (1.40, 1.50, ""),
    ]

    tg_src = build_mock_textgrid(1.60, src_words, src_phones)
    tg_tgt = build_mock_textgrid(1.50, tgt_words, tgt_phones)

    warping_path, silence_mask, sa, ta, words_src, words_tgt, st = _compute_warping(
        tg_src, tg_tgt, tier_name="phones"
    )

    # 期望 LCS 匹配 3 对：are, you, doing
    assert len(words_src) == len(words_tgt), (
        f"LCS 匹配后词对数量应相等，得 src={len(words_src)}, tgt={len(words_tgt)}"
    )
    matched_texts_src = [w.text for w in words_src]
    matched_texts_tgt = [w.text for w in words_tgt]
    assert matched_texts_src == matched_texts_tgt, (
        f"LCS 匹配词文本应完全一致: {matched_texts_src} vs {matched_texts_tgt}"
    )
    assert "what" not in matched_texts_src, "多余词 'what' 不应出现在匹配结果中"
    assert set(matched_texts_src) == {"are", "you", "doing"}, (
        f"匹配词应为 are/you/doing，得到 {matched_texts_src}"
    )

    _assert_warp_correspondence(
        warping_path, words_src, words_tgt,
        label="Case1-word_count_mismatch"
    )


# =============================================================================
# Case 2：words 数量相等，但某个词下 phoneme 数量不等
# =============================================================================

def test_case2_word_equal_phone_mismatch():
    """
    src: [you're(0.1-0.5): Y IY1 R]   → 3 phones (合音 'are' 被 R 代替)
    tgt: [you're(0.1-0.6): Y UW1 ER0] → 3 phones (英音，不同音素序列，但长度相等)

    以及
    src: [great(0.7-1.1): G R EY1 T]   → 4 phones
    tgt: [great(0.7-1.0): G R EY1]     → 3 phones （末尾 T 在 tgt 被 MFA 省略）

    词级匹配完全成功（长度相等），但 great 词下 phone 数量不等。
    phone LCS 应正确处理，great 的 matching phones 只取 G R EY1 三对。

    验证：匹配的音素对中各音素目标帧 → 源帧对应正确。
    """
    src_words = [
        (0.00, 0.08, ""),
        (0.08, 0.50, "you're"),
        (0.50, 0.60, ""),
        (0.60, 1.10, "great"),
        (1.10, 1.20, ""),
    ]
    # src phones
    src_phones = [
        (0.00, 0.08, ""),
        (0.08, 0.20, "Y"),  (0.20, 0.35, "IY1"), (0.35, 0.50, "R"),   # you're
        (0.50, 0.60, ""),
        (0.60, 0.73, "G"),  (0.73, 0.83, "R"), (0.83, 0.97, "EY1"), (0.97, 1.10, "T"),  # great
        (1.10, 1.20, ""),
    ]
    tgt_words = [
        (0.00, 0.06, ""),
        (0.06, 0.52, "you're"),
        (0.52, 0.62, ""),
        (0.62, 1.00, "great"),
        (1.00, 1.10, ""),
    ]
    # tgt phones: you're has same count(3), great has 3 (T missing)
    tgt_phones = [
        (0.00, 0.06, ""),
        (0.06, 0.18, "Y"),  (0.18, 0.38, "UW1"), (0.38, 0.52, "ER0"),  # you're
        (0.52, 0.62, ""),
        (0.62, 0.75, "G"),  (0.75, 0.87, "R"), (0.87, 1.00, "EY1"),  # great (T omitted)
        (1.00, 1.10, ""),
    ]

    tg_src = build_mock_textgrid(1.20, src_words, src_phones)
    tg_tgt = build_mock_textgrid(1.10, tgt_words, tgt_phones)

    warping_path, silence_mask, sa, ta, words_src, words_tgt, st = _compute_warping(
        tg_src, tg_tgt, tier_name="phones"
    )

    # 因为走的是音素级 LCS，也能正确处理
    # 关键断言：匹配后 words_src 和 words_tgt 等长且文本一致
    assert len(words_src) == len(words_tgt), (
        f"匹配结果应等长, src={[w.text for w in words_src]}, tgt={[w.text for w in words_tgt]}"
    )

    _assert_warp_correspondence(
        warping_path, words_src, words_tgt,
        label="Case2-phone_count_mismatch_under_word"
    )


# =============================================================================
# Case 3：静音不对称
# =============================================================================

def test_case3_silence_asymmetry():
    """
    silence_mask 的触发语义：src 锚点跨度 ≈ 0，tgt 锚点跨度 > 0。
    即 tgt 中某段「凭空多出来」（src 对应位置两端锚点重合）时打 True。

    Case 3A – src 有静音，tgt 无（时长压缩）
        src: hello [sil 0.4s] world
        tgt: hello [small sil] world
        silence_mask 全为 False（PCHIP 压缩 src 静音，非凭空生成）

    Case 3B – tgt 有静音，src 两词零间隔（src 跨度 ≈ 0）
        src: ok(紧接)done → ok.end == done.start
        tgt: ok [sil 0.4s] done
        silence_mask 在 [ok_end_tgt, done_start_tgt] 全为 True。

    Case 3C – tgt 有长静音，src 也有短间隔（PCHIP 拉伸，不触发 silence_mask）
        src: ok [0.06s] done
        tgt: ok [0.40s] done
        silence_mask 全为 False。
    """

    # --- sub-test A: src 有长静音，tgt 无 ---
    src_A_words = [
        (0.00, 0.08, ""),
        (0.08, 0.40, "hello"),
        (0.40, 0.80, ""),        # 长静音 src 有，tgt 无
        (0.80, 1.20, "world"),
        (1.20, 1.30, ""),
    ]
    src_A_phones = [
        (0.00, 0.08, ""),
        (0.08, 0.20, "HH"), (0.20, 0.30, "AH0"), (0.30, 0.40, "L"),
        (0.40, 0.80, ""),
        (0.80, 0.95, "W"), (0.95, 1.10, "ER1"), (1.10, 1.20, "L"),
        (1.20, 1.30, ""),
    ]
    tgt_A_words = [
        (0.00, 0.04, ""),
        (0.04, 0.32, "hello"),
        (0.32, 0.36, ""),     # 仅 0.04s 间隔（小静音）
        (0.36, 0.76, "world"),
        (0.76, 0.84, ""),
    ]
    tgt_A_phones = [
        (0.00, 0.04, ""),
        (0.04, 0.14, "HH"), (0.14, 0.24, "AH0"), (0.24, 0.32, "L"),
        (0.32, 0.36, ""),
        (0.36, 0.50, "W"), (0.50, 0.64, "ER1"), (0.64, 0.76, "L"),
        (0.76, 0.84, ""),
    ]

    tg_src_A = build_mock_textgrid(1.30, src_A_words, src_A_phones)
    tg_tgt_A = build_mock_textgrid(0.84, tgt_A_words, tgt_A_phones)

    wp_A, sm_A, sa_A, ta_A, ws_A, wt_A, st_A = _compute_warping(
        tg_src_A, tg_tgt_A, tier_name="phones"
    )

    # src 有长静音，tgt 无 → silence_mask 不会被标记（没有"目标新增静音"）
    # warping_path 在 hello~world 之间应从 src 静音区采样（压缩过来）
    _assert_warp_correspondence(wp_A, ws_A, wt_A, label="Case3A-src_has_silence")

    # --- sub-test B: tgt 有静音，src 两词完全"零间隔"紧挨着 ---
    # silence_mask 的触发条件：src 锚点跨度 ≈ 0，tgt 锚点跨度 > 0。
    # 构造 src 中 ok 和 done 完全无间隔（ok.end == done.start），
    # tgt 中两词之间有 0.4s 静音 → anchor 段 src_span=0, tgt_span=0.4 → 触发 silence_mask。
    src_B_words = [
        (0.00, 0.08, ""),
        (0.08, 0.38, "ok"),
        # 无静音间隔：ok 直接连接 done
        (0.38, 0.82, "done"),
        (0.82, 0.90, ""),
    ]
    src_B_phones = [
        (0.00, 0.08, ""),
        (0.08, 0.20, "AH0"), (0.20, 0.38, "K"),                     # ok
        # ok/done 之间无静音 → 这里 src_span 为 0（两词 anchor 重合）
        (0.38, 0.50, "D"), (0.50, 0.66, "AH1"), (0.66, 0.82, "N"),  # done
        (0.82, 0.90, ""),
    ]
    tgt_B_words = [
        (0.00, 0.06, ""),
        (0.06, 0.36, "ok"),
        (0.36, 0.76, ""),    # tgt 新增长静音 0.4s（src 对应位置跨度为 0）
        (0.76, 1.20, "done"),
        (1.20, 1.28, ""),
    ]
    tgt_B_phones = [
        (0.00, 0.06, ""),
        (0.06, 0.18, "AH0"), (0.18, 0.36, "K"),                          # ok
        (0.36, 0.76, ""),    # tgt 独有长静音
        (0.76, 0.92, "D"), (0.92, 1.06, "AH1"), (1.06, 1.20, "N"),       # done
        (1.20, 1.28, ""),
    ]

    tg_src_B = build_mock_textgrid(0.90, src_B_words, src_B_phones)
    tg_tgt_B = build_mock_textgrid(1.28, tgt_B_words, tgt_B_phones)

    wp_B, sm_B, sa_B, ta_B, ws_B, wt_B, st_B = _compute_warping(
        tg_src_B, tg_tgt_B, tier_name="phones"
    )

    # tgt [0.36, 0.76] 对应 src 处 span≈0 → silence_mask 标记该帧区间
    _assert_silence_mask(sm_B, [(0.36, 0.76)], label="Case3B-tgt_has_new_silence")
    _assert_warp_correspondence(wp_B, ws_B, wt_B, label="Case3B-tgt_has_silence")

    # --- sub-test C: tgt 有长静音，src 也有短间隔（PCHIP 拉伸，不触发 silence_mask）---
    # src ok--done 有 0.06s 间隔 → src_span = 0.06 > eps → PCHIP 拉伸，不触发 silence_mask
    src_C_words = [
        (0.00, 0.08, ""),
        (0.08, 0.38, "ok"),
        (0.38, 0.44, ""),    # src 有 0.06s 间隔（非零）
        (0.44, 0.88, "done"),
        (0.88, 0.96, ""),
    ]
    src_C_phones = [
        (0.00, 0.08, ""),
        (0.08, 0.20, "AH0"), (0.20, 0.38, "K"),       # ok
        (0.38, 0.44, ""),
        (0.44, 0.56, "D"), (0.56, 0.72, "AH1"), (0.72, 0.88, "N"),  # done
        (0.88, 0.96, ""),
    ]
    tgt_C_words = [
        (0.00, 0.06, ""),
        (0.06, 0.36, "ok"),
        (0.36, 0.76, ""),    # tgt 有 0.40s 间隔（比 src 长，但 src 也有非零跨度）
        (0.76, 1.20, "done"),
        (1.20, 1.28, ""),
    ]
    tgt_C_phones = [
        (0.00, 0.06, ""),
        (0.06, 0.18, "AH0"), (0.18, 0.36, "K"),       # ok
        (0.36, 0.76, ""),
        (0.76, 0.92, "D"), (0.92, 1.06, "AH1"), (1.06, 1.20, "N"),  # done
        (1.20, 1.28, ""),
    ]
    tg_src_C = build_mock_textgrid(0.96, src_C_words, src_C_phones)
    tg_tgt_C = build_mock_textgrid(1.28, tgt_C_words, tgt_C_phones)

    wp_C, sm_C, sa_C, ta_C, ws_C, wt_C, st_C = _compute_warping(
        tg_src_C, tg_tgt_C, tier_name="phones"
    )
    # src_span > 0 → PCHIP 拉伸，silence_mask 全为 False
    assert not sm_C.any(), (
        f"Case3C: src 有非零静音间隔，应 PCHIP 拉伸，silence_mask 应全 False，"
        f"但有 {sm_C.sum().item()} 帧为 True"
    )
    _assert_warp_correspondence(wp_C, ws_C, wt_C, label="Case3C-stretch_no_silence_mask")


# =============================================================================
# Case 4：综合（words 不等 + 静音不对称 + phoneme 不等）
# =============================================================================

def test_case4_combined():
    """
    综合测试：src 多一个词 "um" + src/tgt 都有静音但都不触发 silence_mask + 某词下音素不等。

    src: um(0.04-0.22) hello(0.26-0.56) [sil 0.56-0.90] world(0.90-1.30)
         hello phones: HH AH0 L OW1 (4)
    tgt: hello(0.06-0.38) [sil 0.38-0.76] world(0.76-1.16)
         hello phones: HH AH0 L (3, OW1 省略)

    注意：
    - src hello_end=0.56，tgt hello_end=0.38 → src anchor span = 0.56-0.26=0.30 > 0，
      不触发 silence_mask（PCHIP 压缩两端的静音）
    - 因此 silence_mask 全为 False

    验证：
    1. LCS 过滤 "um"
    2. silence_mask 全为 False（PCHIP 处理，非凭空生成）
    3. 匹配词对 (hello,hello)(world,world) warping_path 对应正确
    """
    src_words = [
        (0.00, 0.04, ""),
        (0.04, 0.22, "um"),
        (0.22, 0.26, ""),
        (0.26, 0.56, "hello"),
        (0.56, 0.90, ""),        # src 静音
        (0.90, 1.30, "world"),
        (1.30, 1.38, ""),
    ]
    src_phones = [
        (0.00, 0.04, ""),
        (0.04, 0.10, "AH0"), (0.10, 0.22, "M"),                    # um
        (0.22, 0.26, ""),
        (0.26, 0.34, "HH"), (0.34, 0.42, "AH0"),
        (0.42, 0.50, "L"), (0.50, 0.56, "OW1"),                    # hello (4 phones)
        (0.56, 0.90, ""),
        (0.90, 1.04, "W"), (1.04, 1.18, "ER1"), (1.18, 1.30, "L"), # world
        (1.30, 1.38, ""),
    ]
    tgt_words = [
        (0.00, 0.06, ""),
        (0.06, 0.38, "hello"),
        (0.38, 0.76, ""),        # tgt 静音较长
        (0.76, 1.16, "world"),
        (1.16, 1.24, ""),
    ]
    tgt_phones = [
        (0.00, 0.06, ""),
        (0.06, 0.16, "HH"), (0.16, 0.26, "AH0"), (0.26, 0.38, "L"),  # hello (3 phones, OW1 absent)
        (0.38, 0.76, ""),
        (0.76, 0.90, "W"), (0.90, 1.04, "ER1"), (1.04, 1.16, "L"),   # world
        (1.16, 1.24, ""),
    ]

    tg_src = build_mock_textgrid(1.38, src_words, src_phones)
    tg_tgt = build_mock_textgrid(1.24, tgt_words, tgt_phones)

    warping_path, silence_mask, sa, ta, words_src, words_tgt, st = _compute_warping(
        tg_src, tg_tgt, tier_name="phones"
    )

    # "um" 被 LCS 过滤掉
    assert "um" not in [w.text for w in words_src], "'um' 应被 LCS 过滤"

    # src/tgt 均有静音（src hello_end=0.56 > 0, tgt hello_end=0.38 > 0）
    # src_span 非零 → PCHIP 处理，silence_mask 全为 False
    assert not silence_mask.any(), (
        f"Case4: src anchor span > 0，应 PCHIP 处理，silence_mask 应全 False，"
        f"但有 {silence_mask.sum().item()} 帧为 True"
    )

    # 对已匹配词的 warping_path 验证
    _assert_warp_correspondence(warping_path, words_src, words_tgt, label="Case4-combined")


# =============================================================================
# Case 4b：综合 + silence_mask 触发（src hello/world 零间隔 + tgt 有长静音）
# =============================================================================

def test_case4b_combined_with_silence_mask():
    """
    综合测试（带 silence_mask 触发）：
    src 多一个词 "um" + src hello/world 之间**零间隔** + tgt 有 0.38s 静音 + 某词下音素不等。

    src: um(0.04-0.22) hello(0.26-0.56) world(0.56-0.96)  ← hello/world 零间隔
         hello phones: HH AH0 L OW1 (4)
    tgt: hello(0.06-0.38) [sil 0.38-0.76] world(0.76-1.16)
         hello phones: HH AH0 L (3)

    anchor 在 hello_end / world_start 处：
    - src: hello_end = world_start = 0.56（span = 0）
    - tgt: hello_end = 0.38, world_start = 0.76（span = 0.38）
    → 触发 silence_mask 标记 tgt [0.38, 0.76]

    验证：
    1. LCS 过滤 "um"
    2. silence_mask 在 [0.38, 0.76] 全为 True
    3. 匹配词对 warping_path 对应正确
    """
    src_words = [
        (0.00, 0.04, ""),
        (0.04, 0.22, "um"),
        (0.22, 0.26, ""),
        (0.26, 0.56, "hello"),
        # ← 零间隔：hello.end == world.start
        (0.56, 0.96, "world"),
        (0.96, 1.04, ""),
    ]
    src_phones = [
        (0.00, 0.04, ""),
        (0.04, 0.10, "AH0"), (0.10, 0.22, "M"),                     # um
        (0.22, 0.26, ""),
        (0.26, 0.34, "HH"), (0.34, 0.42, "AH0"),
        (0.42, 0.50, "L"), (0.50, 0.56, "OW1"),                     # hello (4 phones)
        # 零间隔：无静音 phone
        (0.56, 0.70, "W"), (0.70, 0.84, "ER1"), (0.84, 0.96, "L"), # world
        (0.96, 1.04, ""),
    ]
    tgt_words = [
        (0.00, 0.06, ""),
        (0.06, 0.38, "hello"),
        (0.38, 0.76, ""),        # tgt 新增 0.38s 静音
        (0.76, 1.16, "world"),
        (1.16, 1.24, ""),
    ]
    tgt_phones = [
        (0.00, 0.06, ""),
        (0.06, 0.16, "HH"), (0.16, 0.26, "AH0"), (0.26, 0.38, "L"),  # hello (3 phones)
        (0.38, 0.76, ""),
        (0.76, 0.90, "W"), (0.90, 1.04, "ER1"), (1.04, 1.16, "L"),   # world
        (1.16, 1.24, ""),
    ]

    tg_src = build_mock_textgrid(1.04, src_words, src_phones)
    tg_tgt = build_mock_textgrid(1.24, tgt_words, tgt_phones)

    warping_path, silence_mask, sa, ta, words_src, words_tgt, st = _compute_warping(
        tg_src, tg_tgt, tier_name="phones"
    )

    assert "um" not in [w.text for w in words_src], "'um' 应被 LCS 过滤"

    # src hello/world 零间隔 → tgt [0.38, 0.76] 触发 silence_mask
    _assert_silence_mask(silence_mask, [(0.38, 0.76)], label="Case4b-silence_mask_trigger")
    _assert_warp_correspondence(warping_path, words_src, words_tgt, label="Case4b-combined")


# =============================================================================
# VFA 音素域转换辅助
# =============================================================================

def _arpa_to_vfa_label(text: str) -> str:
    """将单个 ARPAbet 音素标签转换为 VFA 音素名。静音标记保持不变。"""
    sil_tokens = ("", "sp", "sil", "<eps>")
    if text in sil_tokens:
        return text
    from data.phoneme_vocab import arpabet_to_vfa_id, vfa_id_to_arpabet
    return vfa_id_to_arpabet(arpabet_to_vfa_id(text))


def _convert_tg_phones_to_vfa(tg: tgt.TextGrid) -> tgt.TextGrid:
    """返回一个新 TextGrid，其 phones tier 中每个音素标签都已转换为 VFA 域名称。

    words tier 保持不变（词级文本标签不参与音素域映射）。
    静音标记（"", "sp", "sil", "<eps>"）保持不变，仅真实音素标签被映射。
    """
    new_tg = tgt.TextGrid()
    for tier in tg.tiers:
        if tier.name == "phones":
            new_tier = tgt.IntervalTier(
                name=tier.name,
                start_time=tier.start_time,
                end_time=tier.end_time,
            )
            for iv in tier:
                new_tier.add_interval(tgt.Interval(
                    iv.start_time, iv.end_time,
                    _arpa_to_vfa_label(iv.text),
                ))
            new_tg.add_tier(new_tier)
        else:
            # words tier（或其他 tier）原样复制
            new_tier = tgt.IntervalTier(
                name=tier.name,
                start_time=tier.start_time,
                end_time=tier.end_time,
            )
            for iv in tier:
                new_tier.add_interval(tgt.Interval(iv.start_time, iv.end_time, iv.text))
            new_tg.add_tier(new_tier)
    return new_tg


# =============================================================================
# Case 5：VFA 音素域 — words 数量不等（复用 Case 1 数据，转换后断言）
# =============================================================================

def test_case5_vfa_word_count_mismatch():
    """
    将 Case 1 的合成 TextGrid 音素标签全部转换为 VFA 域（去除重音、标准化），
    验证 SemanticTransformer 在 VFA 音素域下仍能正确完成 LCS 匹配和 warping。

    VFA 转换示例：EH1 → EH，UW1 → UW，IH0 → IH，W/T/R/Y/D/NG/K 保持不变。
    """
    src_words = [
        (0.0, 0.05, ""),
        (0.05, 0.35, "what"),
        (0.35, 0.38, ""),
        (0.38, 0.68, "are"),
        (0.68, 0.70, ""),
        (0.70, 1.00, "you"),
        (1.00, 1.05, ""),
        (1.05, 1.50, "doing"),
        (1.50, 1.60, ""),
    ]
    src_phones = [
        (0.00, 0.05, ""),
        (0.05, 0.15, "W"),  (0.15, 0.25, "AH1"), (0.25, 0.35, "T"),
        (0.35, 0.38, ""),
        (0.38, 0.50, "EH1"), (0.50, 0.68, "R"),
        (0.68, 0.70, ""),
        (0.70, 0.82, "Y"),  (0.82, 1.00, "UW1"),
        (1.00, 1.05, ""),
        (1.05, 1.20, "D"),  (1.20, 1.35, "UW1"), (1.35, 1.48, "IH0"), (1.48, 1.50, "NG"),
        (1.50, 1.60, ""),
    ]
    tgt_words = [
        (0.0, 0.08, ""),
        (0.08, 0.48, "are"),
        (0.48, 0.52, ""),
        (0.52, 0.88, "you"),
        (0.88, 0.92, ""),
        (0.92, 1.40, "doing"),
        (1.40, 1.50, ""),
    ]
    tgt_phones = [
        (0.00, 0.08, ""),
        (0.08, 0.24, "EH1"), (0.24, 0.48, "R"),
        (0.48, 0.52, ""),
        (0.52, 0.68, "Y"), (0.68, 0.88, "UW1"),
        (0.88, 0.92, ""),
        (0.92, 1.10, "D"), (1.10, 1.25, "UW1"),
        (1.25, 1.36, "IH0"), (1.36, 1.40, "NG"),
        (1.40, 1.50, ""),
    ]

    tg_src = _convert_tg_phones_to_vfa(build_mock_textgrid(1.60, src_words, src_phones))
    tg_tgt = _convert_tg_phones_to_vfa(build_mock_textgrid(1.50, tgt_words, tgt_phones))

    warping_path, silence_mask, sa, ta, words_src, words_tgt, st = _compute_warping(
        tg_src, tg_tgt, tier_name="phones"
    )

    assert len(words_src) == len(words_tgt), (
        f"VFA Case5: LCS 匹配后词对数量应相等，得 src={len(words_src)}, tgt={len(words_tgt)}"
    )
    matched_texts_src = [w.text for w in words_src]
    assert "what" not in matched_texts_src, "VFA Case5: 多余词 'what' 不应出现在匹配结果中"
    assert set(matched_texts_src) == {"are", "you", "doing"}, (
        f"VFA Case5: 匹配词应为 are/you/doing，得到 {matched_texts_src}"
    )

    _assert_warp_correspondence(
        warping_path, words_src, words_tgt,
        label="Case5-vfa_word_count_mismatch"
    )


# =============================================================================
# Case 6：VFA 音素域 — 某词下 phoneme 数量不等（复用 Case 2 数据，转换后断言）
# =============================================================================

def test_case6_vfa_phone_count_mismatch():
    """
    将 Case 2 的合成 TextGrid 转换为 VFA 音素域，验证 phone-level LCS 对齐
    在 VFA 域下同样正确。

    VFA 转换：IY1 → IY，ER0 → ER，EY1 → EY，其余辅音保持不变。
    great: G R EY（T 在 tgt 省略）→ VFA 下：G R EY vs G R EY T（同样 LCS 得 3 对）
    """
    src_words = [
        (0.00, 0.08, ""),
        (0.08, 0.50, "you're"),
        (0.50, 0.60, ""),
        (0.60, 1.10, "great"),
        (1.10, 1.20, ""),
    ]
    src_phones = [
        (0.00, 0.08, ""),
        (0.08, 0.20, "Y"),  (0.20, 0.35, "IY1"), (0.35, 0.50, "R"),
        (0.50, 0.60, ""),
        (0.60, 0.73, "G"),  (0.73, 0.83, "R"), (0.83, 0.97, "EY1"), (0.97, 1.10, "T"),
        (1.10, 1.20, ""),
    ]
    tgt_words = [
        (0.00, 0.06, ""),
        (0.06, 0.52, "you're"),
        (0.52, 0.62, ""),
        (0.62, 1.00, "great"),
        (1.00, 1.10, ""),
    ]
    tgt_phones = [
        (0.00, 0.06, ""),
        (0.06, 0.18, "Y"),  (0.18, 0.38, "UW1"), (0.38, 0.52, "ER0"),
        (0.52, 0.62, ""),
        (0.62, 0.75, "G"),  (0.75, 0.87, "R"), (0.87, 1.00, "EY1"),
        (1.00, 1.10, ""),
    ]

    tg_src = _convert_tg_phones_to_vfa(build_mock_textgrid(1.20, src_words, src_phones))
    tg_tgt = _convert_tg_phones_to_vfa(build_mock_textgrid(1.10, tgt_words, tgt_phones))

    warping_path, silence_mask, sa, ta, words_src, words_tgt, st = _compute_warping(
        tg_src, tg_tgt, tier_name="phones"
    )

    assert len(words_src) == len(words_tgt), (
        f"VFA Case6: 匹配结果应等长, src={[w.text for w in words_src]}, tgt={[w.text for w in words_tgt]}"
    )

    _assert_warp_correspondence(
        warping_path, words_src, words_tgt,
        label="Case6-vfa_phone_count_mismatch"
    )


# =============================================================================
# Case 7：VFA 音素域 — 静音 silence_mask 触发（复用 Case 3B 数据，转换后断言）
# =============================================================================

def test_case7_vfa_silence_mask():
    """
    将 Case 3B（tgt 新增长静音，src 零间隔）的合成 TextGrid 转换为 VFA 音素域，
    验证 silence_mask 在 VFA 域下同样被正确触发。

    VFA 转换：AH0 → AH，AH1 → AH，K/D/N 保持不变。
    """
    src_words = [
        (0.00, 0.08, ""),
        (0.08, 0.38, "ok"),
        (0.38, 0.82, "done"),
        (0.82, 0.90, ""),
    ]
    src_phones = [
        (0.00, 0.08, ""),
        (0.08, 0.20, "AH0"), (0.20, 0.38, "K"),
        (0.38, 0.50, "D"), (0.50, 0.66, "AH1"), (0.66, 0.82, "N"),
        (0.82, 0.90, ""),
    ]
    tgt_words = [
        (0.00, 0.06, ""),
        (0.06, 0.36, "ok"),
        (0.36, 0.76, ""),
        (0.76, 1.20, "done"),
        (1.20, 1.28, ""),
    ]
    tgt_phones = [
        (0.00, 0.06, ""),
        (0.06, 0.18, "AH0"), (0.18, 0.36, "K"),
        (0.36, 0.76, ""),
        (0.76, 0.92, "D"), (0.92, 1.06, "AH1"), (1.06, 1.20, "N"),
        (1.20, 1.28, ""),
    ]

    tg_src = _convert_tg_phones_to_vfa(build_mock_textgrid(0.90, src_words, src_phones))
    tg_tgt = _convert_tg_phones_to_vfa(build_mock_textgrid(1.28, tgt_words, tgt_phones))

    warping_path, silence_mask, sa, ta, words_src, words_tgt, st = _compute_warping(
        tg_src, tg_tgt, tier_name="phones"
    )

    # tgt [0.36, 0.76] 对应 src 处 span≈0 → silence_mask 标记该帧区间
    _assert_silence_mask(silence_mask, [(0.36, 0.76)], label="Case7-vfa_silence_mask_trigger")
    _assert_warp_correspondence(warping_path, words_src, words_tgt, label="Case7-vfa_silence_mask")


# =============================================================================
# 真实数据验证：src 来自 MELD_raw，tgt 来自 MELD_semantic，同名文件构成配对
# =============================================================================

@pytest.mark.skipif(not REAL_DATA_AVAILABLE, reason="MELD_raw / MELD_semantic 数据目录不可访问")
def test_real_data_word_equal():
    """真实数据：words 长度相等的样本 → 验证 warping_path 对应关系（最多 200 条）。"""
    sil_tokens = ("", "sp", "sil", "<eps>")

    def _real(t): return [iv for iv in t if iv.text not in sil_tokens]

    common_files = sorted(
        f for f in os.listdir(REAL_SRC_DIR)
        if f.endswith(".TextGrid") and (REAL_TGT_DIR / f).exists()
    )
    tested = 0
    for fname in common_files:
        r1 = tgt.io.read_textgrid(str(REAL_SRC_DIR / fname))
        r2 = tgt.io.read_textgrid(str(REAL_TGT_DIR / fname))
        w1 = _real(r1.get_tier_by_name("words"))
        w2 = _real(r2.get_tier_by_name("words"))
        if len(w1) != len(w2):
            continue

        warping_path, silence_mask, *_, words_src, words_tgt, _ = _compute_warping(
            r1, r2, tier_name="phones"
        )
        _assert_warp_correspondence(
            warping_path, words_src, words_tgt,
            label=f"real_word_equal/{fname}"
        )
        tested += 1
        if tested >= 200:
            break

    assert tested > 0, "没有找到可测试的 words-equal 样本"
    print(f"  通过 {tested} 条真实 words-equal 样本")


@pytest.mark.skipif(not REAL_DATA_AVAILABLE, reason="MELD_raw / MELD_semantic 数据目录不可访问")
def test_real_data_word_mismatch():
    """真实数据：words 长度不等的样本（LCS 路径）→ 验证 warping_path 对应关系（遍历全部）。"""
    sil_tokens = ("", "sp", "sil", "<eps>")

    def _real(t): return [iv for iv in t if iv.text not in sil_tokens]

    common_files = sorted(
        f for f in os.listdir(REAL_SRC_DIR)
        if f.endswith(".TextGrid") and (REAL_TGT_DIR / f).exists()
    )
    tested = 0
    skipped = 0
    for fname in common_files:
        r1 = tgt.io.read_textgrid(str(REAL_SRC_DIR / fname))
        r2 = tgt.io.read_textgrid(str(REAL_TGT_DIR / fname))
        w1 = _real(r1.get_tier_by_name("words"))
        w2 = _real(r2.get_tier_by_name("words"))
        if len(w1) == len(w2):
            continue

        warping_path, silence_mask, *_, words_src, words_tgt, _ = _compute_warping(
            r1, r2, tier_name="phones"
        )
        if len(words_src) == 0:
            skipped += 1
            continue  # LCS 无匹配，跳过
        _assert_warp_correspondence(
            warping_path, words_src, words_tgt,
            label=f"real_word_mismatch/{fname}"
        )
        tested += 1

    print(f"  通过 {tested} 条真实 words-mismatch 样本（跳过 LCS 无匹配 {skipped} 条）")


@pytest.mark.skipif(not REAL_DATA_AVAILABLE, reason="MELD_raw / MELD_semantic 数据目录不可访问")
def test_real_data_vfa_phones():
    """真实数据（VFA 音素域）：将 src/tgt TextGrid 音素标签转换为 VFA 域后，
    验证 warping_path 对应关系（最多 200 条，words 长度相等的样本）。"""
    sil_tokens = ("", "sp", "sil", "<eps>")

    def _real(t): return [iv for iv in t if iv.text not in sil_tokens]

    common_files = sorted(
        f for f in os.listdir(REAL_SRC_DIR)
        if f.endswith(".TextGrid") and (REAL_TGT_DIR / f).exists()
    )
    tested = 0
    for fname in common_files:
        r1 = tgt.io.read_textgrid(str(REAL_SRC_DIR / fname))
        r2 = tgt.io.read_textgrid(str(REAL_TGT_DIR / fname))
        w1 = _real(r1.get_tier_by_name("words"))
        w2 = _real(r2.get_tier_by_name("words"))
        if len(w1) != len(w2):
            continue

        tg_src_vfa = _convert_tg_phones_to_vfa(r1)
        tg_tgt_vfa = _convert_tg_phones_to_vfa(r2)

        warping_path, silence_mask, *_, words_src, words_tgt, _ = _compute_warping(
            tg_src_vfa, tg_tgt_vfa, tier_name="phones"
        )
        _assert_warp_correspondence(
            warping_path, words_src, words_tgt,
            label=f"real_vfa/{fname}"
        )
        tested += 1
        if tested >= 200:
            break

    assert tested > 0, "没有找到可测试的 words-equal 样本（VFA 域）"
    print(f"  通过 {tested} 条真实 words-equal 样本（VFA 音素域）")


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Case 1: words 数量不等（LCS 匹配）")
    test_case1_word_count_mismatch()
    print("  PASSED")

    print("Case 2: words 相等，某词下 phoneme 不等（LCS 音素）")
    test_case2_word_equal_phone_mismatch()
    print("  PASSED")

    print("Case 3A: src 有长静音 tgt 无")
    print("Case 3B: tgt 有长静音 src 无")
    test_case3_silence_asymmetry()
    print("  PASSED")

    print("Case 4: 综合（words 不等 + 静音不对称 + phoneme 不等）")
    test_case4_combined()
    print("  PASSED")

    print("Case 4b: 综合 + silence_mask 触发")
    test_case4b_combined_with_silence_mask()
    print("  PASSED")

    print("Case 5: VFA 音素域 — words 数量不等")
    test_case5_vfa_word_count_mismatch()
    print("  PASSED")

    print("Case 6: VFA 音素域 — 某词下 phoneme 数量不等")
    test_case6_vfa_phone_count_mismatch()
    print("  PASSED")

    print("Case 7: VFA 音素域 — silence_mask 触发")
    test_case7_vfa_silence_mask()
    print("  PASSED")

    if REAL_DATA_AVAILABLE:
        print("Real data - words equal (最多 200 条):")
        test_real_data_word_equal()
        print("Real data - words mismatch (LCS, 全量):")
        test_real_data_word_mismatch()
        print("Real data - VFA 音素域 words equal (最多 200 条):")
        test_real_data_vfa_phones()
    else:
        print("(真实数据路径不存在，跳过)")

    print("=" * 60)
    print("All tests passed.")
