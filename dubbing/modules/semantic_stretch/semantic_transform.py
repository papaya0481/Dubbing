"""
SemanticTransformer
===================
将 S_infer（语义码 latent，shape: B x T x D，50 fps）沿时序维度进行扭曲，
使其韵律/时长从 source TextGrid（TTS 输出 wav 的 MFA 对齐结果）映射到
target TextGrid（用户提供的目标时长信息）。

设计原则：
- 对应 mel_transform.py 中的 GlobalWarpTransformer，但作用于离散语义码 latent。
- 使用 F.grid_sample 完全向量化时序插值，无 Python for 循环。
- 锚点构建逻辑与 GlobalWarpTransformer 完全一致（PCHIP + 单调保证）。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import tgt
from scipy.interpolate import PchipInterpolator


class SemanticTransformer:
    """基于 TextGrid 对齐的 latent 时序扭曲器。

    输入 x 的 shape 为 ``(B, T_src, D)``，支持语义码（50 fps）或 cond（mel fps）空间。
    输出 warped x 的 shape 为 ``(B, T_tgt, D)``，时序对齐到目标 TextGrid。

    典型用法::

        transformer = SemanticTransformer(device="cuda", input_type="semantic")
        x_warped, tgt_duration = transformer.transform(
            x=x,                          # (B, T_src, D)
            source_textgrid=src_tg,       # MFA 对齐的源 TextGrid
            target_textgrid=tgt_tg,       # 用户提供的目标 TextGrid
            tier_name="phones",
        )
    """

    #: 语义码的帧率（semantic codec 运行在 50 Hz）
    CODES_PER_SECOND: float = 50.0
    MEL_PER_SECOND: float = 22050.0 / 256.0  # mel 帧率（hop=256, sr=22050）

    def __init__(
        self,
        device: str = "cuda",
        verbose: bool = False,
        input_type: str = "cond",  # "semantic" (50 fps) or "cond" (mel fps)
        grid_sample_mode: str = "bilinear",
        use_phone_lcs_within_word: bool = False,  # 单词内音素不一致时是否使用 LCS
    ) -> None:
        assert input_type in ("semantic", "cond"), f"Unknown input_type: {input_type!r}"
        self.device = device
        self.verbose = verbose
        self.input_type = input_type
        self.grid_sample_mode = grid_sample_mode
        self.use_phone_lcs_within_word = use_phone_lcs_within_word

    # ------------------------------------------------------------------
    # 工具：提取有效 interval（排除静音标记）
    # ------------------------------------------------------------------

    def get_real_words(self, tier: tgt.IntervalTier) -> List[tgt.Interval]:
        """过滤掉静音/空白 interval。"""
        return [iv for iv in tier if iv.text not in ("", "sp", "sil", "<eps>")]

    # ------------------------------------------------------------------
    # 秒 → 帧 转换
    # ------------------------------------------------------------------

    def _sec_to_frame(self, sec: float) -> float:
        """将时间（秒）换算为（小数）码帧索引。"""
        if self.input_type == "semantic":
            return sec * self.CODES_PER_SECOND
        else:
            return sec * self.MEL_PER_SECOND

    # ------------------------------------------------------------------
    # 锚点构建（与 GlobalWarpTransformer 逻辑完全对应）
    # ------------------------------------------------------------------

    def _append_monotonic_anchor(
        self,
        src_anchors: List[float],
        tgt_anchors: List[float],
        src_time: float,
        tgt_time: float,
        eps: float = 1e-4,
        dup_tol: float = 1e-6,
    ) -> None:
        """追加锚点，保证 tgt_anchors 严格单调递增。"""
        if tgt_time <= tgt_anchors[-1]:
            is_same = (
                abs(tgt_time - tgt_anchors[-1]) < dup_tol
                and abs(src_time - src_anchors[-1]) < dup_tol
            )
            if is_same:
                return
            tgt_time = tgt_anchors[-1] + eps
        tgt_anchors.append(tgt_time)
        src_anchors.append(src_time)

    def _group_phones_by_words(
        self,
        phones: List[tgt.Interval],
        words: List[tgt.Interval],
    ) -> List[List[tgt.Interval]]:
        """将音素列表按单词边界分组。"""
        groups: List[List[tgt.Interval]] = [[] for _ in words]
        for phone in phones:
            p_mid = (phone.start_time + phone.end_time) / 2.0
            for i, word in enumerate(words):
                if word.start_time - 1e-6 <= p_mid <= word.end_time + 1e-6:
                    groups[i].append(phone)
                    break
        return groups

    def _align_intervals_lcs(
        self,
        seq_src: List[tgt.Interval],
        seq_tgt: List[tgt.Interval],
    ) -> Tuple[List[tgt.Interval], List[tgt.Interval]]:
        """LCS 对齐两个 Interval 序列（按 .text 匹配），返回平行的最长公共匹配子序列。

        时间复杂度 O(n*m)，对于典型句子长度（< 100 词 / < 300 音素）完全可接受。

        Returns:
            ``(matched_src, matched_tgt)``：两个等长的平行列表，仅包含双侧均有对应的 interval。
        """
        n, m = len(seq_src), len(seq_tgt)

        # 构建 DP 表（滚动数组可省内存，但此处保留完整以便回溯）
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(1, n + 1):
            txt_s = seq_src[i - 1].text.strip().lower()
            for j in range(1, m + 1):
                txt_t = seq_tgt[j - 1].text.strip().lower()
                if txt_s == txt_t:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        # 回溯，逆序收集匹配对
        matched_s: List[tgt.Interval] = []
        matched_t: List[tgt.Interval] = []
        i, j = n, m
        while i > 0 and j > 0:
            txt_s = seq_src[i - 1].text.strip().lower()
            txt_t = seq_tgt[j - 1].text.strip().lower()
            if txt_s == txt_t:
                matched_s.append(seq_src[i - 1])
                matched_t.append(seq_tgt[j - 1])
                i -= 1
                j -= 1
            elif dp[i - 1][j] >= dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
        matched_s.reverse()
        matched_t.reverse()
        return matched_s, matched_t

    def _build_phone_groups(
        self,
        tg_src: tgt.TextGrid,
        tg_tgt: tgt.TextGrid,
        phones_src: List[tgt.Interval],
        phones_tgt: List[tgt.Interval],
    ) -> Tuple[
        List[tgt.Interval],
        List[tgt.Interval],
        Optional[List[List[tgt.Interval]]],
        Optional[List[List[tgt.Interval]]],
    ]:
        """按 words tier 构建 word-guided phone groups，自动处理长度不等的情况。

        三级降级策略：

        1. **词长相等**：直接一一配对（原行为）。
        2. **词长不等**：用 LCS 找最长公共匹配子序列，仅保留双侧均有对应的词对作为
           anchor；未匹配词的时间区间由后续 PCHIP 平滑内插，不会引入错位。
        3. **无词层 / 异常**：降级到音素级对齐；若音素列表长度也不一致，同样用 LCS。
        """
        try:
            words_tier_src = tg_src.get_tier_by_name("words")
            words_tier_tgt = tg_tgt.get_tier_by_name("words")
            real_words_src = self.get_real_words(words_tier_src)
            real_words_tgt = self.get_real_words(words_tier_tgt)

            if len(real_words_src) > 0 and len(real_words_tgt) > 0:
                if len(real_words_src) == len(real_words_tgt):
                    # 长度相等：直接配对，保持原行为
                    pg_src = self._group_phones_by_words(phones_src, real_words_src)
                    pg_tgt = self._group_phones_by_words(phones_tgt, real_words_tgt)
                    return real_words_src, real_words_tgt, pg_src, pg_tgt
                else:
                    # 长度不等：LCS 词级对齐，只用匹配词对构 anchor
                    matched_src, matched_tgt = self._align_intervals_lcs(
                        real_words_src, real_words_tgt
                    )
                    if self.verbose:
                        print(
                            f"[SemanticTransformer] word LCS: "
                            f"src={len(real_words_src)} tgt={len(real_words_tgt)} "
                            f"matched={len(matched_src)}"
                        )
                    if len(matched_src) > 0:
                        pg_src = self._group_phones_by_words(phones_src, matched_src)
                        pg_tgt = self._group_phones_by_words(phones_tgt, matched_tgt)
                        return matched_src, matched_tgt, pg_src, pg_tgt
        except Exception:
            pass

        # 降级：音素级对齐；长度不等时同样用 LCS
        if len(phones_src) != len(phones_tgt):
            matched_p_src, matched_p_tgt = self._align_intervals_lcs(phones_src, phones_tgt)
            if self.verbose:
                print(
                    f"[SemanticTransformer] phone LCS fallback: "
                    f"src={len(phones_src)} tgt={len(phones_tgt)} "
                    f"matched={len(matched_p_src)}"
                )
            if len(matched_p_src) > 0:
                return matched_p_src, matched_p_tgt, None, None
        return phones_src, phones_tgt, None, None

    def build_anchors(
        self,
        words_src: List[tgt.Interval],
        words_tgt: List[tgt.Interval],
        src_duration: float,
        tgt_duration: float,
        eps: float = 1e-4,
        phone_groups_src: Optional[List[List[tgt.Interval]]] = None,
        phone_groups_tgt: Optional[List[List[tgt.Interval]]] = None,
    ) -> Tuple[List[float], List[float]]:
        """构建源/目标单调时间锚点列表（秒）。"""
        src_anchors = [0.0]
        tgt_anchors = [0.0]
        use_phones = phone_groups_src is not None and phone_groups_tgt is not None

        for idx, (w_src, w_tgt) in enumerate(zip(words_src, words_tgt)):
            phones_aligned = False
            if use_phones:
                pg_src = phone_groups_src[idx]
                pg_tgt = phone_groups_tgt[idx]
                if len(pg_src) > 0 and len(pg_tgt) > 0:
                    if len(pg_src) == len(pg_tgt):
                        # 音素数量一致，直接对齐
                        for p_src, p_tgt in zip(pg_src, pg_tgt):
                            self._append_monotonic_anchor(
                                src_anchors, tgt_anchors,
                                p_src.start_time, p_tgt.start_time, eps=eps,
                            )
                            self._append_monotonic_anchor(
                                src_anchors, tgt_anchors,
                                p_src.end_time, p_tgt.end_time, eps=eps,
                            )
                        phones_aligned = True
                    elif self.use_phone_lcs_within_word:
                        # 音素数量不一致，且开关开启：使用 LCS 对齐（原逻辑）
                        matched_p_src, matched_p_tgt = self._align_intervals_lcs(pg_src, pg_tgt)
                        if len(matched_p_src) > 0:
                            if self.verbose:
                                print(
                                    f"[SemanticTransformer] word {idx} phone LCS: "
                                    f"src={len(pg_src)} tgt={len(pg_tgt)} "
                                    f"matched={len(matched_p_src)}"
                                )
                            for p_src, p_tgt in zip(matched_p_src, matched_p_tgt):
                                self._append_monotonic_anchor(
                                    src_anchors, tgt_anchors,
                                    p_src.start_time, p_tgt.start_time, eps=eps,
                                )
                                self._append_monotonic_anchor(
                                    src_anchors, tgt_anchors,
                                    p_src.end_time, p_tgt.end_time, eps=eps,
                                )
                            phones_aligned = True
                    # else: 音素数量不一致，且开关关闭：回退到单词级别（phones_aligned 保持 False）

            if not phones_aligned:
                # 回退到单词级别对齐
                self._append_monotonic_anchor(
                    src_anchors, tgt_anchors,
                    w_src.start_time, w_tgt.start_time, eps=eps,
                )
                self._append_monotonic_anchor(
                    src_anchors, tgt_anchors,
                    w_src.end_time, w_tgt.end_time, eps=eps,
                )

        if tgt_anchors[-1] < tgt_duration - eps:
            tgt_anchors.append(tgt_duration)
            src_anchors.append(src_duration)

        return src_anchors, tgt_anchors

    # ------------------------------------------------------------------
    # 核心：时序扭曲路径计算（PCHIP，全向量化）
    # ------------------------------------------------------------------

    def calculate_warping_path(
        self,
        src_anchors: List[float],
        tgt_anchors: List[float],
        total_tgt_frames: int,
    ) -> torch.Tensor:
        """计算目标每帧对应的源帧索引（shape: ``(total_tgt_frames,)``）。

        实现与 GlobalWarpTransformer 完全一致，单位换算为码帧（50 fps）。
        """
        src_frames_raw = np.array([self._sec_to_frame(t) for t in src_anchors])
        tgt_frames_raw = np.array([self._sec_to_frame(t) for t in tgt_anchors])

        # 去除 src 零跨度段的内部重复点，保留段首段尾（形成平台）
        keep = np.ones(len(src_frames_raw), dtype=bool)
        i = 0
        while i < len(src_frames_raw) - 1:
            if abs(src_frames_raw[i + 1] - src_frames_raw[i]) < 0.5:
                j = i + 1
                while (
                    j < len(src_frames_raw) - 1
                    and abs(src_frames_raw[j + 1] - src_frames_raw[j]) < 0.5
                ):
                    keep[j] = False
                    j += 1
                i = j + 1
            else:
                i += 1
        src_frames = src_frames_raw[keep]
        tgt_frames = tgt_frames_raw[keep]

        # 保证 tgt 维度严格单调（PCHIP 要求）
        _, unique_idx = np.unique(tgt_frames, return_index=True)
        src_frames = src_frames[unique_idx]
        tgt_frames = tgt_frames[unique_idx]

        interpolator = PchipInterpolator(tgt_frames, src_frames)
        grid_src = interpolator(np.arange(total_tgt_frames))
        # 防止末端轻微外推导致越界（尤其在 cond 帧率下总帧数更大时）。
        # 这里裁剪到源锚点范围，保证后续采样稳定且不出现负值。
        grid_src = np.clip(grid_src, src_frames.min(), src_frames.max())

        return torch.from_numpy(grid_src.astype(np.float32)).to(self.device)

    def _detect_silence_mask(
        self,
        src_anchors: List[float],
        tgt_anchors: List[float],
        total_tgt_frames: int,
        src_eps: float = 1e-4,
    ) -> torch.Tensor:
        """构建布尔 mask：目标中对应插入静音的帧为 True。"""
        mask = torch.zeros(total_tgt_frames, dtype=torch.bool)
        for i in range(len(src_anchors) - 1):
            src_span = abs(src_anchors[i + 1] - src_anchors[i])
            tgt_span = tgt_anchors[i + 1] - tgt_anchors[i]
            if src_span < src_eps and tgt_span > src_eps:
                f_start = max(0, int(round(self._sec_to_frame(tgt_anchors[i]))))
                f_end = min(
                    total_tgt_frames,
                    int(round(self._sec_to_frame(tgt_anchors[i + 1]))),
                )
                if f_end > f_start:
                    mask[f_start:f_end] = True
        return mask

    # ------------------------------------------------------------------
    # 核心：向量化 warp（F.grid_sample，无 for 循环）
    # ------------------------------------------------------------------

    def warp(
        self,
        x: torch.Tensor,              # (B, T_src, D)
        warping_path: torch.Tensor,   # (T_tgt,) or (B, T_tgt) — 目标帧 → 源帧索引（float）
        silence_mask: Optional[torch.Tensor] = None,  # (T_tgt,) or (B, T_tgt) bool
    ) -> torch.Tensor:                # (B, T_tgt, D)
        """用 F.grid_sample 沿时序轴向量化扭曲输入。

        将 x 从 ``(B, T_src, D)`` 插值到 ``(B, T_tgt, D)``。
        全程无 Python-level for 循环，由 CUDA kernel 完成采样。

        支持批量推理：warping_path 和 silence_mask 可以是 (B, T_tgt) 形状，
        为每个样本使用不同的 warping path。
        """
        B, T_src, D = x.shape

        # 处理 warping_path 的形状
        if warping_path.dim() == 1:
            # 单个 path，广播到所有 batch
            T_tgt = warping_path.shape[0]
            warping_path = warping_path.unsqueeze(0).expand(B, T_tgt)  # (B, T_tgt)
        else:
            # 批量 paths
            T_tgt = warping_path.shape[1]

        # 变形为 (B, D, 1, T_src)：把 D 当"通道"，T 当"宽度"，高度=1
        s_4d = x.permute(0, 2, 1).unsqueeze(2)  # (B, D, 1, T_src)

        # 归一化坐标到 [-1, 1]，支持批量
        norm_x = 2.0 * warping_path.clamp(0, T_src - 1) / max(T_src - 1, 1) - 1.0  # (B, T_tgt)

        # 构建采样 grid: (B, 1, T_tgt, 2)  [x, y] — y 固定为 0
        grid_x = norm_x.unsqueeze(1).unsqueeze(-1)  # (B, 1, T_tgt, 1)
        grid_y = torch.zeros_like(grid_x)
        grid = torch.cat([grid_x, grid_y], dim=-1)  # (B, 1, T_tgt, 2)

        # 双线性插值（等价于 1-D 线性插值）；padding_mode='border' 防越界
        warped_4d = F.grid_sample(
            s_4d, grid,
            mode=self.grid_sample_mode,
            padding_mode="border",
            align_corners=True,
        )  # (B, D, 1, T_tgt)

        warped = warped_4d.squeeze(2).permute(0, 2, 1)  # (B, T_tgt, D)

        # 插入静音段：将无源内容的帧置零（完全向量化）
        if silence_mask is not None and silence_mask.any():
            silence_mask = silence_mask.to(warped.device)
            if silence_mask.dim() == 1:
                # 单个 mask，广播到所有 batch
                warped[:, silence_mask, :] = 0.0
            else:
                # 批量 masks，向量化应用：(B, T_tgt, 1) * (B, T_tgt, D)
                mask_expanded = silence_mask.unsqueeze(-1)  # (B, T_tgt, 1)
                warped = warped.masked_fill(mask_expanded, 0.0)

        return warped


    # ------------------------------------------------------------------
    # 高层接口
    # ------------------------------------------------------------------

    def transform(
        self,
        x: torch.Tensor,                                                    # (B, T_src, D)
        source_textgrid: Union[str, Path, tgt.TextGrid, List[Union[str, Path, tgt.TextGrid]]],
        target_textgrid: Union[str, Path, tgt.TextGrid, List[Union[str, Path, tgt.TextGrid]]],
        tier_name: str = "phones",
    ) -> Tuple[torch.Tensor, Union[float, List[float]]]:
        """根据 TextGrid 对齐将 S_infer 从源时序扭曲到目标时序。

        支持批量推理：source_textgrid 和 target_textgrid 可以是列表，
        长度必须等于 batch size B。

        Args:
            x: 形状 ``(B, T_src, D)`` 的输入 latent（语义码或 cond）。
            source_textgrid: 源 TextGrid（TTS 输出 wav 的 MFA 对齐结果）。
                可以是单个 TextGrid（兼容旧版）或 TextGrid 列表（批量推理）。
            target_textgrid: 目标 TextGrid（用户提供）。
                可以是单个 TextGrid（兼容旧版）或 TextGrid 列表（批量推理）。
            tier_name: 对齐使用的 tier（``"phones"`` 或 ``"words"``）。

        Returns:
            ``(warped, tgt_duration)``

            - ``warped``: shape ``(B, T_tgt, D)``，时序对齐到目标时长。
            - ``tgt_duration``: 目标时长（秒）。单个值（兼容旧版）或列表（批量推理）。
        """
        if self.input_type == "cond":
            assert x.size(-1) == 512, f"Expected cond dimension 512, got {x.size(-1)}"

        B = x.shape[0]

        # 检测是否为批量推理模式
        is_batch_mode = isinstance(source_textgrid, list) or isinstance(target_textgrid, list)

        if is_batch_mode:
            # 批量推理模式
            if not isinstance(source_textgrid, list):
                source_textgrid = [source_textgrid] * B
            if not isinstance(target_textgrid, list):
                target_textgrid = [target_textgrid] * B

            assert len(source_textgrid) == B, f"source_textgrid list length {len(source_textgrid)} != batch size {B}"
            assert len(target_textgrid) == B, f"target_textgrid list length {len(target_textgrid)} != batch size {B}"

            return self._transform_batch(x, source_textgrid, target_textgrid, tier_name)
        else:
            # 兼容旧版：单个 TextGrid
            return self._transform_single(x, source_textgrid, target_textgrid, tier_name)

    def _process_single_textgrid_pair(
        self,
        source_tg: Union[str, Path, tgt.TextGrid],
        target_tg: Union[str, Path, tgt.TextGrid],
        tier_name: str,
    ) -> Tuple[List[float], List[float], int, float]:
        """处理单个 TextGrid 对，返回 anchors 和目标帧数。"""
        def _as_tg(x: Union[str, Path, tgt.TextGrid]) -> tgt.TextGrid:
            return x if isinstance(x, tgt.TextGrid) else tgt.io.read_textgrid(str(x))

        def _duration(tier: tgt.IntervalTier) -> float:
            return tier.end_time if len(tier) > 0 else 0.0

        tg_src = _as_tg(source_tg)
        tg_tgt = _as_tg(target_tg)

        tier_src = tg_src.get_tier_by_name(tier_name)
        tier_tgt = tg_tgt.get_tier_by_name(tier_name)

        phones_src = self.get_real_words(tier_src)
        phones_tgt = self.get_real_words(tier_tgt)
        src_duration = _duration(tier_src)
        tgt_duration = _duration(tier_tgt)

        if tier_name == "phones":
            words_src, words_tgt, pg_src, pg_tgt = self._build_phone_groups(
                tg_src, tg_tgt, phones_src, phones_tgt
            )
        else:
            words_src, words_tgt = phones_src, phones_tgt
            pg_src = pg_tgt = None

        src_anchors, tgt_anchors = self.build_anchors(
            words_src, words_tgt,
            src_duration, tgt_duration,
            phone_groups_src=pg_src,
            phone_groups_tgt=pg_tgt,
        )

        total_tgt_frames = max(1, int(round(self._sec_to_frame(tgt_duration))))

        return src_anchors, tgt_anchors, total_tgt_frames, tgt_duration

    def _transform_single(
        self,
        x: torch.Tensor,
        source_textgrid: Union[str, Path, tgt.TextGrid],
        target_textgrid: Union[str, Path, tgt.TextGrid],
        tier_name: str,
    ) -> Tuple[torch.Tensor, float]:
        """单样本转换（兼容旧版 API）。"""
        src_anchors, tgt_anchors, total_tgt_frames, tgt_duration = self._process_single_textgrid_pair(
            source_textgrid, target_textgrid, tier_name
        )

        if self.verbose:
            print(
                f"[SemanticTransformer] src_dur → tgt_dur={tgt_duration:.3f}s, "
                f"src_frames={x.shape[1]}, tgt_frames={total_tgt_frames}"
            )

        warping_path = self.calculate_warping_path(src_anchors, tgt_anchors, total_tgt_frames)
        silence_mask = self._detect_silence_mask(src_anchors, tgt_anchors, total_tgt_frames)
        warped = self.warp(x.to(self.device), warping_path, silence_mask)

        return warped, tgt_duration

    def _transform_batch(
        self,
        x: torch.Tensor,
        source_textgrids: List[Union[str, Path, tgt.TextGrid]],
        target_textgrids: List[Union[str, Path, tgt.TextGrid]],
        tier_name: str,
    ) -> Tuple[torch.Tensor, List[float]]:
        """批量样本转换（新 API），无 Python for 循环处理 tensor。"""
        B = x.shape[0]

        # 并行处理所有 TextGrid 对（这部分无法避免，因为涉及文件 I/O 和复杂逻辑）
        batch_data = [
            self._process_single_textgrid_pair(source_textgrids[b], target_textgrids[b], tier_name)
            for b in range(B)
        ]

        # 解包结果
        src_anchors_list = [d[0] for d in batch_data]
        tgt_anchors_list = [d[1] for d in batch_data]
        tgt_frames_list = [d[2] for d in batch_data]
        tgt_durations = [d[3] for d in batch_data]

        max_tgt_frames = max(tgt_frames_list)

        # 批量计算 warping paths（向量化）
        warping_paths = [
            self.calculate_warping_path(src_anchors_list[b], tgt_anchors_list[b], tgt_frames_list[b])
            for b in range(B)
        ]

        # 批量计算 silence masks（向量化）
        silence_masks = [
            self._detect_silence_mask(src_anchors_list[b], tgt_anchors_list[b], tgt_frames_list[b])
            for b in range(B)
        ]

        # Pad 到相同长度并 stack（向量化操作）
        warping_paths_padded = [
            F.pad(wp, (0, max_tgt_frames - wp.shape[0]), mode='constant', value=wp[-1].item())
            if wp.shape[0] < max_tgt_frames else wp
            for wp in warping_paths
        ]

        silence_masks_padded = [
            F.pad(sm.float(), (0, max_tgt_frames - sm.shape[0]), mode='constant', value=1.0).bool()
            if sm.shape[0] < max_tgt_frames else sm
            for sm in silence_masks
        ]

        # Stack into batch tensors（向量化）
        warping_path_batch = torch.stack(warping_paths_padded, dim=0)  # (B, max_tgt_frames)
        silence_mask_batch = torch.stack(silence_masks_padded, dim=0)  # (B, max_tgt_frames)

        if self.verbose:
            for b in range(B):
                print(
                    f"[SemanticTransformer] batch {b}: tgt_dur={tgt_durations[b]:.3f}s, "
                    f"src_frames={x.shape[1]}, tgt_frames={tgt_frames_list[b]}"
                )

        # 批量 warp（完全向量化，无 for 循环）
        warped = self.warp(x.to(self.device), warping_path_batch, silence_mask_batch)

        return warped, tgt_durations

