"""
IndexTTS2Semantic
=================
继承 IndexTTS2，在正常 TTS 推理后额外执行：

  1. 用 MFAAligner 对输出 wav 做强制对齐 → source_textgrid
  2. 用 SemanticTransformer 将 S_infer 按 target_textgrid 进行时长扭曲 → S_warped
  3. 将 S_warped 送入 length_regulator → CFM → BigVGAN，生成最终 wav

环境隔离：本文件重载 IndexTTS2，所有语义扭曲逻辑封装于此。
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional, Union

import torch
import torchaudio
import tgt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "modules"))

from infer_v2 import IndexTTS2
from semantic_transform import SemanticTransformer
from modules.mfa_alinger import MFAAligner


class IndexTTS2Semantic(IndexTTS2):
    """在 IndexTTS2 基础上增加语义 latent 时序扭曲能力。

    新增推理入口：:meth:`infer_with_semantic_warp`。

    Args:
        mfa_aligner: ``MFAAligner`` 实例，用于对 TTS 输出 wav 做强制对齐。
            可于构造时传入，也可以在 ``infer_with_semantic_warp()`` 调用时传入。
        semantic_transformer_device: SemanticTransformer 使用的 device；
            默认与主模型保持一致。
        verbose_transform: 是否打印 SemanticTransformer 的调试信息。
        *args / **kwargs: 其余参数与 IndexTTS2 完全一致。
    """

    def __init__(
        self,
        *args,
        mfa_beam: int = 20,
        mfa_retry_beam: int = 200,
        semantic_transformer_device: Optional[str] = None,
        verbose_transform: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mfa_aligner = MFAAligner(beam=mfa_beam, retry_beam=mfa_retry_beam)
        _dev = semantic_transformer_device or self.device
        self.semantic_transformer = SemanticTransformer(device=_dev, verbose=verbose_transform)

    # ------------------------------------------------------------------
    # 内部：用扭曲后的 S_warped 重跑 length_regulator → CFM → BigVGAN
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _decode_s_warped_to_wav(
        self,
        s_warped: torch.Tensor,       # (B, T_tgt_codes, D)
        tgt_duration: float,           # 目标时长（秒），用于计算目标 mel 帧数
        diffusion_steps: int = 25,
        inference_cfg_rate: float = 0.7,
    ) -> torch.Tensor:
        """用扭曲后的 S_infer 重跑 length_regulator → CFM → BigVGAN，返回 wav。

        依赖 ``self.cache_*`` 缓存（由最后一次 ``infer()`` 填充）。

        Returns:
            wav: shape ``(1, N_samples)``，float32，未 clamp。
        """
        prompt_condition = self.cache_s2mel_prompt   # (1, T_prompt, D_cond)
        ref_mel = self.cache_mel                      # (1, n_mels, T_ref)
        style = self.cache_s2mel_style                # (1, D_style)

        if prompt_condition is None or ref_mel is None or style is None:
            raise RuntimeError(
                "缓存为空，请先调用一次 infer() 以填充 prompt_condition / ref_mel / style。"
            )

        # 目标 mel 帧数（与 infer_v2.py 保持一致：22050 Hz / 256 hop）
        _MEL_SR, _MEL_HOP = 22050, 256
        target_mel_len = max(1, int(round(tgt_duration * _MEL_SR / _MEL_HOP)))
        target_lengths = torch.LongTensor([target_mel_len]).to(self.device)

        # length_regulator: (B, T_tgt_codes, D) → (B, T_tgt_mel, D_cond)
        cond = self.s2mel.models["length_regulator"](
            s_warped, ylens=target_lengths, n_quantizers=3, f0=None
        )[0]

        # 拼接 prompt condition 并送入 CFM
        cat_condition = torch.cat([prompt_condition, cond], dim=1)
        vc_target = self.s2mel.models["cfm"].inference(
            cat_condition,
            torch.LongTensor([cat_condition.size(1)]).to(cat_condition.device),
            ref_mel,
            style,
            None,
            diffusion_steps,
            inference_cfg_rate=inference_cfg_rate,
        )
        vc_target = vc_target[:, :, ref_mel.size(-1):]   # 去掉 prompt 帧

        wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0)   # (1, N)
        return wav

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def infer_with_semantic_warp(
        self,
        spk_audio_prompt: Union[str, list],
        text: str,
        output_path: Optional[str],
        target_textgrid: Union[str, Path, tgt.TextGrid],
        mfa_aligner=None,
        tier_name: str = "phones",
        diffusion_steps: int = 25,
        inference_cfg_rate: float = 0.7,
        sampling_rate: int = 22050,
        emo_vector: Optional[torch.Tensor] = None,
        **infer_kwargs,
    ):
        """先做一次正常 TTS 推理，再对输出 wav 做 MFA 对齐，最后用 SemanticTransformer
        将 S_infer 扭曲到目标时长后重新生成音频。

        Args:
            spk_audio_prompt: 参考音频路径（或路径列表）。
            text: 输入文本（支持 ``|`` 多段）。
            output_path: 最终 wav 保存路径；``None`` 时以 ``(sr, ndarray)`` 返回。
            target_textgrid: 目标时长对应的 TextGrid（用户提供），可为路径或对象。
            mfa_aligner: ``MFAAligner`` 实例；``None`` 时使用 ``self.mfa_aligner``。
            tier_name: TextGrid tier 名，``"phones"``（默认）或 ``"words"``。
            first_pass_output_path: 第一次推理结果保存路径；``None`` 时自动使用临时文件。
            diffusion_steps: CFM 扩散步数（默认 25）。
            inference_cfg_rate: CFM classifier-free guidance 比例（默认 0.7）。
            sampling_rate: 输出采样率，固定 22050 Hz。
            **infer_kwargs: 其余关键字参数透传给底层 ``infer()``。

        Returns:
            - 若 ``output_path`` 非 None：``(output_path, seg_lens, wav_length_final)``
            - 若 ``output_path`` 为 None：``(sampling_rate, wav_ndarray)``
        """
        aligner = mfa_aligner if mfa_aligner is not None else self.mfa_aligner


        # ----------------------------------------------------------
        # 1. 第一次推理：缓存 speaker/style/prompt，并获取 S_infer
        # ----------------------------------------------------------
        first_result = self.infer(
            spk_audio_prompt=spk_audio_prompt,
            text=text,
            emo_vector=emo_vector,
            return_stats=True,
            method = "hmm",    # 强制使用 hmm 以获得更稳定的时长预测（更适合后续扭曲）
            **infer_kwargs,
        )
        seg_lens, sampling_rate, wav_out, inference_stats = first_result
        S_infer: torch.Tensor = inference_stats["S_infer"].to(self.device)

        # ----------------------------------------------------------
        # 2. 加载第一次推理的 wav，准备 MFA 对齐
        # ----------------------------------------------------------
        wav_out_mono = wav_out                     # (N,)
        if wav_out_mono.dtype == torch.int16:
            wav_out_mono = wav_out_mono.float() / 32767.0

        # ----------------------------------------------------------
        # 3. MFA 强制对齐：得到 source_textgrid
        #    去除 "|" 分隔符，MFA 只处理纯文本
        # ----------------------------------------------------------
        clean_text = " ".join(text).strip()
        import time
        start_time = time.time()
        mfa_result = aligner.align_one_wav(
            wavs=wav_out_mono,
            sampling_rate=sampling_rate,
            text=clean_text,
            return_textgrid=True,
        )
        end_time = time.time()
        print(f"MFA 对齐耗时：{end_time - start_time:.2f} 秒")
        # ctm_to_textgrid_fast 返回 (tgt.TextGrid, phone_groups)
        source_tg: tgt.TextGrid = mfa_result[0]

        # ----------------------------------------------------------
        # 4. SemanticTransformer：将 S_infer 扭曲到目标时长
        # ----------------------------------------------------------
        S_warped, tgt_duration = self.semantic_transformer.transform(
            s_infer=S_infer,
            source_textgrid=source_tg,
            target_textgrid=target_textgrid,
            tier_name=tier_name,
        )

        # ----------------------------------------------------------
        # 5. 用 S_warped 重新生成 wav
        # ----------------------------------------------------------
        wav_final = self._decode_s_warped_to_wav(
            s_warped=S_warped,
            tgt_duration=tgt_duration,
            diffusion_steps=diffusion_steps,
            inference_cfg_rate=inference_cfg_rate,
        )

        wav_final = torch.clamp(32767 * wav_final, -32767.0, 32767.0).cpu()
        wav_length_final = wav_final.shape[-1] / sampling_rate

        # ----------------------------------------------------------
        # 6. 保存或返回
        # ----------------------------------------------------------
        if output_path is not None:
            if os.path.isfile(output_path):
                os.remove(output_path)
            _out_dir = os.path.dirname(os.path.abspath(output_path))
            if _out_dir:
                os.makedirs(_out_dir, exist_ok=True)
            torchaudio.save(output_path, wav_final.type(torch.int16), sampling_rate)
            return output_path, seg_lens, wav_length_final
        else:
            wav_data = wav_final.type(torch.int16).numpy().T
            return seg_lens, sampling_rate, wav_data
