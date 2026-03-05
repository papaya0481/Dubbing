"""
IndexTTS2 批量推理封装
======================
核心优化：
1. 相同说话人/情绪的音频条件只提取一次（字典缓存）
2. 同一 speaker-group 的所有样本合并为一次 inference_speech 调用
   → GPT 自回归生成在底层仅 forward 一次（最大瓶颈优化）
3. gpt.forward + s2mel + BigVGAN 逐样本处理（留有进一步批量空间）
4. 提供 DubbingDataset / collate 供 gen.py 的 DataLoader 使用

NOTE：
- gpt.inference_speech 的内部实现把所有 segments 拼成一条长序列、一次 generate，
  因此对于同 speaker 的 N 个样本，把它们的 text_inputs_list / emo_vecs 合并后传入
  即可实现跨样本的单次自回归生成（真正的 batch inference）。
- 若需要跨不同 speaker 的真正 GPU batch（batch_size > 1 on generate），需要改造
  gpt/model_v2.py::prepare_gpt_inputs 以支持填充 + mask，目前不在此处做。
- s2mel CFM 的 setup_caches(max_batch_size=1) 被本类覆盖为指定值，
  以便在未来对 CFM 跨段批量推理时使用。
"""

from __future__ import annotations

import os
import time
import random
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from indextts.infer_v2 import IndexTTS2


# ---------------------------------------------------------------------------
# Dataset / Collate
# ---------------------------------------------------------------------------

class DubbingDataset(Dataset):
    """把 (spk_prompt, text, output_path, …) dict 列表包装成 Dataset。

    每个 item 是一个 dict，包含以下必需字段：
        spk_audio_prompt (str)
        text             (str)  支持 "|" 分隔多句
        output_path      (str)
    可选字段：
        emo_audio_prompt (str | None)
        emo_vectors      (list | None)   每句一个 8 维情绪向量
        target_duration_tokens (list | None)
    """

    def __init__(self, items: List[Dict[str, Any]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


def collate_items(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """恒等 collation：数据本身不是张量，直接返回 list。"""
    return batch


# ---------------------------------------------------------------------------
# IndexTTS2Batch
# ---------------------------------------------------------------------------

class IndexTTS2Batch(IndexTTS2):
    """IndexTTS2 批量推理版本。

    Args:
        s2mel_max_batch_size: 预留给 s2mel CFM estimator 的最大 batch_size，
            默认与 DataLoader batch_size 相同或更大。
        其余参数与 IndexTTS2 相同。
    """

    def __init__(self, *args, s2mel_max_batch_size: int = 8, **kwargs):
        super().__init__(*args, **kwargs)
        # 重新设置 s2mel cfm 的 cache 上限，为跨段批量预留空间
        self.s2mel.models['cfm'].estimator.setup_caches(
            max_batch_size=s2mel_max_batch_size,
            max_seq_length=8192,
        )
        self.s2mel_max_batch_size = s2mel_max_batch_size

        # 批量专用字典缓存（key = prompt 路径）
        self._spk_cache: Dict[str, tuple] = {}   # -> (spk_cond_emb, style, prompt_cond, ref_mel)
        self._emo_cache: Dict[str, Any] = {}      # -> emo_cond_emb

    # ------------------------------------------------------------------
    # 私有辅助：提取并缓存单个说话人/情绪条件
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _prepare_spk(self, spk_audio_prompt: str, verbose: bool = False) -> tuple:
        """提取说话人条件并放入字典缓存；若已存在直接返回。"""
        if spk_audio_prompt in self._spk_cache:
            return self._spk_cache[spk_audio_prompt]

        audio, sr = self._load_and_cut_audio(spk_audio_prompt, 15, verbose)
        audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
        audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

        inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
        spk_cond_emb = self.get_emb(
            inputs["input_features"].to(self.device),
            inputs["attention_mask"].to(self.device),
        )

        _, S_ref = self.semantic_codec.quantize(spk_cond_emb)
        ref_mel = self.mel_fn(audio_22k.to(spk_cond_emb.device).float())
        ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)

        feat = torchaudio.compliance.kaldi.fbank(
            audio_16k.to(ref_mel.device),
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000,
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        style = self.campplus_model(feat.unsqueeze(0))

        prompt_condition = self.s2mel.models['length_regulator'](
            S_ref, ylens=ref_target_lengths, n_quantizers=3, f0=None
        )[0]

        result = (spk_cond_emb, style, prompt_condition, ref_mel)
        self._spk_cache[spk_audio_prompt] = result
        return result

    @torch.no_grad()
    def _prepare_emo(self, emo_audio_prompt: str, verbose: bool = False) -> Any:
        """提取情绪条件并放入字典缓存；若已存在直接返回。"""
        if emo_audio_prompt in self._emo_cache:
            return self._emo_cache[emo_audio_prompt]

        emo_audio, _ = self._load_and_cut_audio(emo_audio_prompt, 15, verbose, sr=16000)
        emo_inputs = self.extract_features(emo_audio, sampling_rate=16000, return_tensors="pt")
        emo_cond_emb = self.get_emb(
            emo_inputs["input_features"].to(self.device),
            emo_inputs["attention_mask"].to(self.device),
        )
        self._emo_cache[emo_audio_prompt] = emo_cond_emb
        return emo_cond_emb

    # ------------------------------------------------------------------
    # 私有辅助：构建单个样本的 emovecs（情绪向量列表，每句一个）
    # ------------------------------------------------------------------

    def _build_emovecs(
        self,
        spk_cond_emb: torch.Tensor,
        emo_cond_emb: torch.Tensor,
        emo_vectors: Optional[List],
        emo_alpha: float,
        use_random: bool,
        style: torch.Tensor,
    ) -> List[torch.Tensor]:
        """复现 infer_v2.py 中的 emovecs 构建逻辑，返回每个 segment 的 emovec。"""
        spk_cond_lengths = torch.tensor(
            [spk_cond_emb.shape[-1]], device=self.device
        )
        emo_cond_lengths = torch.tensor(
            [emo_cond_emb.shape[-1]], device=self.device
        )
        base_emovec = self.gpt.merge_emovec(
            spk_cond_emb, emo_cond_emb,
            spk_cond_lengths, emo_cond_lengths,
            alpha=emo_alpha,
        )

        if emo_vectors is None or len(emo_vectors) == 0:
            return [base_emovec]

        emo_vectors_tensor = torch.as_tensor(
            emo_vectors, device=self.device, dtype=style.dtype
        )
        emo_weight_sums = emo_vectors_tensor.sum(dim=1, keepdim=True)  # [n_segs, 1]

        if use_random:
            emovec_mats = []
            for evo in emo_vectors_tensor:
                ri = [random.randint(0, x - 1) for x in self.emo_num]
                emo_matrix = torch.cat(
                    [m[idx].unsqueeze(0) for idx, m in zip(ri, self.emo_matrix)], dim=0
                )
                emovec_mats.append(torch.sum(evo.unsqueeze(1) * emo_matrix, 0, keepdim=True))
        else:
            ri = [self.find_most_similar_cosine(style, m) for m in self.spk_matrix]
            emo_matrix = torch.cat(
                [m[idx].unsqueeze(0) for idx, m in zip(ri, self.emo_matrix)], dim=0
            )
            emovec_mats_tensor = emo_vectors_tensor @ emo_matrix  # [n_segs, C]
            emovec_mats = list(emovec_mats_tensor.split(1, dim=0))

        emovec_mats_cat = torch.cat(emovec_mats, dim=0)  # [n_segs, C]
        mixed = emovec_mats_cat + (
            1 - emo_weight_sums.to(base_emovec.dtype)
        ) * base_emovec.expand(emovec_mats_cat.size(0), -1)
        return list(mixed.split(1, dim=0))

    # ------------------------------------------------------------------
    # 主批量推理函数
    # ------------------------------------------------------------------

    def infer_batch(
        self,
        spk_audio_prompts: List[str],
        texts: List[str],
        output_paths: List[str],
        emo_audio_prompts: Optional[List[str]] = None,
        emo_vectors_list: Optional[List[Any]] = None,
        emo_alpha: float = 1.0,
        use_random: bool = False,
        verbose: bool = False,
        interval_silence: int = 200,
        max_text_tokens_per_sentence: int = 120,
        target_duration_tokens_list: Optional[List[Any]] = None,
        return_stats: bool = False,
        **generation_kwargs,
    ) -> List:
        """批量推理 N 个样本。

        关键流程：
        1. 提取所有唯一说话人/情绪条件（字典缓存，同 prompt 只提取一次）
        2. 按说话人 prompt 分组
        3. 每组内所有样本的 text_inputs_list + emovecs 合并后调用
           一次 gpt.inference_speech → 一次 GPT 自回归生成（真正的批量）
        4. 按 seg_lens 将 codes 分片映射回各样本
        5. 逐样本 gpt.forward + s2mel + BigVGAN + 保存音频

        Args:
            spk_audio_prompts: 长度 N 的说话人音频路径列表
            texts: 长度 N 的文本列表（支持 "|" 多句，每段对应一个 emo_vector）
            output_paths: 长度 N 的输出 wav 路径列表
            emo_audio_prompts: None 则与 spk_audio_prompts 相同
            emo_vectors_list: 每个样本的情绪向量列表；None 则不使用情绪矩阵
            target_duration_tokens_list: 每个样本的时长 token 约束；None 则不限制
            return_stats: 是否在返回结果中包含统计信息

        Returns:
            results: 每个样本的 infer() 返回值（output_path 或 (sr, wav_data)）
        """
        N = len(texts)
        assert len(spk_audio_prompts) == N == len(output_paths), (
            "spk_audio_prompts / texts / output_paths 长度必须一致"
        )

        if emo_audio_prompts is None:
            emo_audio_prompts = list(spk_audio_prompts)
        if emo_vectors_list is None:
            emo_vectors_list = [None] * N
        if target_duration_tokens_list is None:
            target_duration_tokens_list = [None] * N

        # 从 generation_kwargs 提取推理超参
        do_sample         = generation_kwargs.pop("do_sample", True)
        top_p             = generation_kwargs.pop("top_p", 0.8)
        top_k             = generation_kwargs.pop("top_k", 30)
        temperature       = generation_kwargs.pop("temperature", 0.8)
        length_penalty    = generation_kwargs.pop("length_penalty", 0.0)
        num_beams         = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens    = generation_kwargs.pop("max_mel_tokens", 1500)
        method            = generation_kwargs.pop("method", "hmm")
        save_attn_maps    = generation_kwargs.pop("save_attention_maps", False)
        sampling_rate     = 22050

        total_t0 = time.perf_counter()

        # ---------------------------------------------------------------
        # Phase 1：预提取所有唯一说话人 / 情绪条件
        # ---------------------------------------------------------------
        for prompt in dict.fromkeys(spk_audio_prompts):
            self._prepare_spk(prompt, verbose=verbose)
        for prompt in dict.fromkeys(emo_audio_prompts):
            self._prepare_emo(prompt, verbose=verbose)

        # ---------------------------------------------------------------
        # Phase 2：按说话人分组，每组做一次 inference_speech（批量 GPT）
        #   group_map: spk_prompt -> list of item indices
        # ---------------------------------------------------------------
        group_map: Dict[str, List[int]] = defaultdict(list)
        for i, prompt in enumerate(spk_audio_prompts):
            group_map[prompt].append(i)

        # results[i] 保存第 i 个样本的输出
        results: List[Any] = [None] * N

        for spk_prompt, item_indices in group_map.items():
            spk_cond_emb, style, prompt_condition, ref_mel = self._spk_cache[spk_prompt]

            # ---- 2a. 为本组所有样本构建 text_inputs_list 和 emovecs ----
            # 同时记录每个样本贡献了多少个 token 序列（句子数）
            group_text_list: List[torch.Tensor] = []   # 全组展平后的 text tensors
            group_emovecs:   List[torch.Tensor] = []   # 对应 emovecs
            item_seg_counts: List[int]           = []  # 每个样本贡献的句子数

            target_dur_tokens_group: Optional[List[int]] = []
            has_target_dur = any(
                target_duration_tokens_list[i] is not None for i in item_indices
            )

            for i in item_indices:
                emo_cond_emb = self._emo_cache[emo_audio_prompts[i]]
                emo_vectors  = emo_vectors_list[i]
                text_i       = texts[i]

                # 分句 + tokenize
                text_parts = text_i.split("|")
                text_tokens_i = []
                for part in text_parts:
                    toks = self.tokenizer.tokenize(part)
                    ids  = self.tokenizer.convert_tokens_to_ids(toks)
                    text_tokens_i.append(
                        torch.tensor(ids, dtype=torch.int32, device=self.device).unsqueeze(0)
                    )

                # 构建 emovecs（每句一个）
                emovecs_i = self._build_emovecs(
                    spk_cond_emb, emo_cond_emb,
                    emo_vectors, emo_alpha, use_random, style,
                )
                # 如果样本只有 1 句但 emo_vectors 有多条，取第一条匹配
                if len(emovecs_i) < len(text_tokens_i):
                    emovecs_i = emovecs_i + [emovecs_i[-1]] * (
                        len(text_tokens_i) - len(emovecs_i)
                    )
                elif len(emovecs_i) > len(text_tokens_i):
                    emovecs_i = emovecs_i[:len(text_tokens_i)]

                group_text_list.extend(text_tokens_i)
                group_emovecs.extend(emovecs_i)
                item_seg_counts.append(len(text_tokens_i))

                # 时长约束
                if has_target_dur:
                    tdt = target_duration_tokens_list[i]
                    if tdt is None:
                        target_dur_tokens_group.extend([None] * len(text_tokens_i))
                    else:
                        target_dur_tokens_group.extend(
                            tdt if isinstance(tdt, list) else [tdt]
                        )

            if not has_target_dur:
                target_dur_tokens_group = None

            # ---- 2b. 一次 GPT 自回归生成整组所有 segment ----
            spk_cond_lengths = torch.tensor(
                [spk_cond_emb.shape[-1]], device=self.device
            )
            emo_cond_emb_group = self._emo_cache[emo_audio_prompts[item_indices[0]]]
            emo_cond_lengths   = torch.tensor(
                [emo_cond_emb_group.shape[-1]], device=self.device
            )

            gpt_t0 = time.perf_counter()
            with torch.no_grad():
                with torch.amp.autocast(
                    self.device.split(":")[0],
                    enabled=self.dtype is not None,
                    dtype=self.dtype,
                ):
                    gpt_output = self.gpt.inference_speech(
                        spk_cond_emb,
                        group_text_list,
                        emo_cond_emb_group,
                        cond_lengths=spk_cond_lengths,
                        emo_cond_lengths=emo_cond_lengths,
                        emo_vecs=group_emovecs,
                        do_sample=do_sample,
                        top_p=top_p,
                        top_k=top_k,
                        temperature=temperature,
                        num_return_sequences=1,
                        length_penalty=length_penalty,
                        num_beams=num_beams,
                        repetition_penalty=repetition_penalty,
                        max_generate_length=max_mel_tokens,
                        target_duration_tokens=target_dur_tokens_group,
                        method=method,
                        save_attention_maps=save_attn_maps,
                        **generation_kwargs,
                    )

            codes_all, speech_cond_latent, attn_mask, seg_lens, token_gen_time, aligned_seqs = gpt_output
            gpt_elapsed = time.perf_counter() - gpt_t0

            n_group = len(item_indices)
            n_segs  = sum(item_seg_counts)
            self.logger.info(
                f"[infer_batch] group spk={os.path.basename(spk_prompt)} | "
                f"items={n_group} segs={n_segs} | "
                f"GPT gen: {gpt_elapsed:.2f}s"
            )

            # ---- 2c. 按 seg_lens 将 codes_all 分片映射回各样本 ----
            # seg_lens: list of int，每个 segment 产生的 code 数量
            # 如果 seg_lens 为空（单段生成）用 codes_all 长度代替
            if not seg_lens:
                seg_lens = [codes_all.shape[-1]]

            # 将 seg_lens 按样本聚合
            seg_cursor = 0
            item_codes: List[torch.Tensor] = []
            item_seg_lens_list: List[List[int]] = []

            for i_local, seg_cnt in enumerate(item_seg_counts):
                item_sl = seg_lens[seg_cursor: seg_cursor + seg_cnt]
                seg_cursor += seg_cnt
                item_seg_lens_list.append(item_sl)

                # 拼出该样本的 codes 段（各 segment 顺序拼接）
                start = sum(seg_lens[:sum(item_seg_counts[:i_local])])
                end   = start + sum(item_sl)
                item_codes.append(codes_all[:, start:end])

            # ---- 2d. 逐样本：gpt.forward + s2mel + BigVGAN ----
            for i_local, global_i in enumerate(item_indices):
                codes_i   = item_codes[i_local]               # [1, T_codes_i]
                seg_ls    = item_seg_lens_list[i_local]       # per-seg lengths
                out_path  = output_paths[global_i]
                text_i    = texts[global_i]

                text_parts    = text_i.split("|")
                text_tokens_i = []
                for part in text_parts:
                    toks = self.tokenizer.tokenize(part)
                    ids  = self.tokenizer.convert_tokens_to_ids(toks)
                    text_tokens_i.append(
                        torch.tensor(ids, dtype=torch.int32, device=self.device).unsqueeze(0)
                    )

                # 当前样本的 emovecs
                emo_idx_start = sum(item_seg_counts[:i_local])
                emovecs_i = group_emovecs[emo_idx_start: emo_idx_start + item_seg_counts[i_local]]

                code_lens_i = torch.tensor(
                    [codes_i.shape[-1]], device=self.device, dtype=torch.long
                )
                use_speed   = torch.zeros(
                    spk_cond_emb.size(0), device=self.device, dtype=torch.long
                )

                spk_cond_lengths_i = torch.tensor(
                    [spk_cond_emb.shape[-1]], device=self.device
                )
                emo_cond_emb_i    = self._emo_cache[emo_audio_prompts[global_i]]
                emo_cond_lengths_i = torch.tensor(
                    [emo_cond_emb_i.shape[-1]], device=self.device
                )

                gpt_fwd_t0 = time.perf_counter()
                with torch.amp.autocast(
                    self.device.split(":")[0],
                    enabled=self.dtype is not None,
                    dtype=self.dtype,
                ):
                    latent_i = self.gpt(
                        speech_cond_latent,
                        torch.cat(text_tokens_i, dim=1),
                        torch.tensor(
                            [text_tokens_i[0].shape[-1]], device=self.device
                        ),
                        codes_i,
                        code_lens_i,
                        emo_cond_emb_i,
                        cond_mel_lengths=spk_cond_lengths_i,
                        emo_cond_mel_lengths=emo_cond_lengths_i,
                        emo_vecs=emovecs_i,
                        use_speed=use_speed,
                        attention_mask=attn_mask,
                    )
                gpt_fwd_elapsed = time.perf_counter() - gpt_fwd_t0

                # s2mel + BigVGAN
                s2mel_t0 = time.perf_counter()
                dtype_ctx = None
                with torch.amp.autocast(
                    self.device.split(":")[0],
                    enabled=dtype_ctx is not None,
                    dtype=dtype_ctx,
                ):
                    latent_i = self.s2mel.models['gpt_layer'](latent_i)
                    S_infer  = self.semantic_codec.quantizer.vq2emb(
                        codes_i.unsqueeze(1)
                    ).transpose(1, 2)
                    S_infer  = S_infer + latent_i

                    tdt_i = target_duration_tokens_list[global_i]
                    mel_ratio  = 1 / 50 * 22050 / 256
                    if tdt_i is not None:
                        tdt_tensor = torch.tensor(
                            tdt_i if isinstance(tdt_i, list) else [tdt_i],
                            device=self.device, dtype=torch.long,
                        )
                        target_lengths = tdt_tensor[:1]  # take total sum if multi-seg
                    else:
                        target_lengths = (code_lens_i * mel_ratio).long()

                    cond_s2mel = self.s2mel.models['length_regulator'](
                        S_infer, ylens=target_lengths, n_quantizers=3, f0=None
                    )[0]
                    cat_cond = torch.cat([prompt_condition, cond_s2mel], dim=1)

                    vc_target = self.s2mel.models['cfm'].inference(
                        cat_cond,
                        torch.LongTensor([cat_cond.size(1)]).to(cat_cond.device),
                        ref_mel,
                        style,
                        None,
                        25,
                        inference_cfg_rate=0.7,
                    )
                    vc_target = vc_target[:, :, ref_mel.size(-1):]
                s2mel_elapsed = time.perf_counter() - s2mel_t0

                bigvgan_t0 = time.perf_counter()
                with torch.no_grad():
                    wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0)
                wav = wav.squeeze(1)
                bigvgan_elapsed = time.perf_counter() - bigvgan_t0

                wav = torch.clamp(32767 * wav, -32767.0, 32767.0).cpu()

                self.logger.info(
                    f"  item[{global_i}] gpt_fwd={gpt_fwd_elapsed:.2f}s "
                    f"s2mel={s2mel_elapsed:.2f}s bigvgan={bigvgan_elapsed:.2f}s"
                )

                # 保存音频
                if out_path:
                    if os.path.isfile(out_path):
                        os.remove(out_path)
                    if os.path.dirname(out_path):
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    torchaudio.save(out_path, wav.type(torch.int16), sampling_rate)
                    results[global_i] = (out_path, seg_ls,
                                        wav.shape[-1] / sampling_rate, {})
                else:
                    wav_np = wav.type(torch.int16).numpy().T
                    results[global_i] = (sampling_rate, wav_np)

        total_elapsed = time.perf_counter() - total_t0
        self.logger.info(
            f"[infer_batch] {N} items done in {total_elapsed:.2f}s "
            f"({total_elapsed / max(N, 1):.2f}s/item)"
        )
        return results
