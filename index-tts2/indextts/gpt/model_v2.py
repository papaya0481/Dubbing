import functools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import LogitsProcessor
from transformers import GPT2Config, LogitsProcessorList
from indextts.gpt.transformers_gpt2 import GPT2PreTrainedModel, GPT2Model

# from transformers import GPT2Config, GPT2PreTrainedModel, LogitsProcessorList
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils.model_parallel_utils import (assert_device_map,
                                                     get_device_map)

from indextts.gpt.conformer_encoder import ConformerEncoder
from indextts.gpt.perceiver import PerceiverResampler
from indextts.utils.arch_util import AttentionBlock
from indextts.utils.typical_sampling import TypicalLogitsWarper

from .attn_map import AttentionMapProcessor
from .hmm import StreamingHMMAligner
from .transformers_generation_utils import construct_attn_mask

def null_position_embeddings(range, dim):
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)

class MinimalDurationLogitsProcessor(LogitsProcessor):
    # 这个是旧版本，不启用
    def __init__(
        self,
        target_length: int,
        stop_token_id: int,
        verbose: bool = True,
        min_ratio: float = 0.5,
        neutral_ratio: tuple = (0.8, 1.2),
        max_ratio: float = 2.0,
        max_negative_bias: float = -5.0,
        max_positive_bias: float = 10.0
    ):
        # Core parameters
        self.target_length = target_length
        self.stop_token_id = stop_token_id
        self.verbose = verbose
        
        # Duration control hyperparameters
        self.min_ratio = min_ratio
        self.neutral_start_ratio = neutral_ratio[0]
        self.neutral_end_ratio = neutral_ratio[1]
        self.max_ratio = max_ratio
        self.max_negative_bias = max_negative_bias
        self.max_positive_bias = max_positive_bias
        
        # Validate parameters
        assert 0.0 < min_ratio < neutral_ratio[0] < neutral_ratio[1] < max_ratio, \
            "Ratios must satisfy: 0 < min_ratio < neutral_start < neutral_end < max_ratio"
        
        # Compute and cache length thresholds
        self._update_thresholds()
        
        # State tracking
        self.current_length = 0
        self.initial_offset = None  # 新增：记录前置token偏移量
        
        if self.verbose:
            print(f"[MinimalDuration] Initialized:")
            print(f"   Target: {self.target_length}")
            print(f"   Region A (suppress): < {self.min_length}")
            print(f"   Region B (ramp down): {self.min_length} → {self.neutral_start_length}")
            print(f"   Region C (neutral): {self.neutral_start_length} → {self.neutral_end_length}")
            print(f"   Region D (ramp up): {self.neutral_end_length} → {self.max_length}")
            print(f"   Region E (encourage): > {self.max_length}")
    
    def _update_thresholds(self):
        """Precompute length thresholds based on target and ratios."""
        self.min_length = int(self.min_ratio * self.target_length)
        self.neutral_start_length = int(self.neutral_start_ratio * self.target_length)
        self.neutral_end_length = int(self.neutral_end_ratio * self.target_length)
        self.max_length = int(self.max_ratio * self.target_length)
    
    def _compute_bias(self, current_len: int) -> float:
        """
        Compute the bias value for the stop token based on current generation length.
        
        Returns a smooth piecewise-linear function of progress toward target_length.
        """
        if current_len < self.min_length:
            # Region A: Strong suppression
            return self.max_negative_bias
        
        elif current_len < self.neutral_start_length:
            # Region B: Linear ramp from max_negative_bias → 0
            progress = (current_len - self.min_length) / (self.neutral_start_length - self.min_length)
            return self.max_negative_bias * (1.0 - progress)
        
        elif current_len < self.neutral_end_length:
            # Region C: Neutral zone (no bias)
            return 0.0
        
        elif current_len < self.max_length:
            # Region D: Linear ramp from 0 → max_positive_bias
            progress = (current_len - self.neutral_end_length) / (self.max_length - self.neutral_end_length)
            return self.max_positive_bias * progress
        
        else:
            # Region E: Strong encouragement
            return self.max_positive_bias
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply bias to stop token logit based on current generation progress.
        
        Args:
            input_ids: Generated token IDs so far [batch_size, seq_len]
            scores: Raw logits for next token [batch_size, vocab_size]
        
        Returns:
            Modified scores with bias applied to stop_token_id
        """
        # 第一次调用时，记录初始长度作为偏移量
        if self.initial_offset is None:
            self.initial_offset = input_ids.shape[1]
            if self.verbose:
                print(f">> [MinimalDuration] Detected offset: {self.initial_offset} tokens (condition + text)")
        
        # 计算实际生成的semantic token数量（减去前置偏移）
        total_length = input_ids.shape[1]
        self.current_length = total_length - self.initial_offset
        
        # Compute bias for current progress
        bias = self._compute_bias(self.current_length)
        
        # Apply bias to stop token logit for all samples in batch
        scores[:, self.stop_token_id] += bias
        
        # Optional: Print progress (every N tokens to reduce spam)
        if self.verbose and self.current_length % 50 == 0:
            ratio = self.current_length / self.target_length
            print(f">> [MinimalDuration] Progress: {self.current_length}/{self.target_length} "
                  f"({ratio:.1%}) | Bias: {bias:+.2f} | Total len: {total_length}")
        
        return scores

class RemainingBudgetEOSProcessor(LogitsProcessor):
    """
    段独立的 EOS 偏置处理器（最后一段控制）
    
    1. 知道当前在哪一段
    2. 每段基于自己的目标计算 progress
    3. 最后段控制：
       - 非最后一段：始终最大抑制 EOS（max_negative_bias）
       - 最后一段：根据 progress 动态调整 bias
    4. 前段误差不会影响后续段的 EOS 控制
    
    """
    
    def __init__(
        self,
        target_tokens_per_segment: list,  # 每段的目标长度 [100, 100, 150, ...]
        stop_token_id: int,
        verbose: bool = True,
        # Duration control hyperparameters
        min_ratio: float = 0.5,
        neutral_ratio: tuple = (0.8, 1.2),
        max_ratio: float = 2.0,
        max_negative_bias: float = -5.0,
        max_positive_bias: float = 10.0,
    ):
        self.target_tokens_per_segment = target_tokens_per_segment
        self.num_segments = len(target_tokens_per_segment)
        self.stop_token_id = stop_token_id
        self.verbose = verbose
        
        self.min_ratio = min_ratio
        self.neutral_start_ratio = neutral_ratio[0]
        self.neutral_end_ratio = neutral_ratio[1]
        self.max_ratio = max_ratio
        self.max_negative_bias = max_negative_bias
        self.max_positive_bias = max_positive_bias
        
        # Validate
        assert 0.0 < min_ratio < neutral_ratio[0] < neutral_ratio[1] < max_ratio
        
        self.initial_offset = None  # 前置 token 数量（condition + text）
        
        self.current_segment_idx = None  # will be [0, 0, ..., 0] for each beam
        
        # actual_generated_per_segment[beam_id][seg_idx] = 实际生成的 token 数
        self.actual_generated_per_segment = None
        
        # 当前段内已生成的 token 数
        self.generated_since_last_segment = None
        
        # 段切换位置记录（从外部传入）
        self.segment_positions = None
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"[RemainingBudgetEOS] Initialized")
            print(f"  Num segments: {self.num_segments}")
            print(f"  Target per segment: {self.target_tokens_per_segment}")
            print(f"  Total target: {sum(self.target_tokens_per_segment)}")
            print(f"  Control ratios: min={min_ratio}, neutral={neutral_ratio}, max={max_ratio}")
            print(f"  Bias range: [{max_negative_bias}, {max_positive_bias}]")
            print(f"{'='*70}\n")
    
    def _initialize_state(self, batch_size: int):
        self.current_segment_idx = [0] * batch_size
        self.generated_since_last_segment = [0] * batch_size
        self.actual_generated_per_segment = [
            [0] * self.num_segments for _ in range(batch_size)
        ]
    
    def update_segment_positions(self, segment_positions: dict):
        """
        外部调用：更新段切换位置
        
        Args:
            segment_positions: {beam_id: [pos1, pos2, ...]} 段切换的 semantic token 位置
        """
        self.segment_positions = segment_positions
    
    def _detect_segment_switch(self, current_length: int, beam_idx: int) -> bool:
        """
        检测当前 beam 是否发生了段切换
        
        Args:
            current_length: 当前已生成的 semantic token 数量
            beam_idx: beam 索引
            
        Returns:
            是否发生了段切换
        """
        if self.segment_positions is None:
            return False
        
        if beam_idx not in self.segment_positions:
            return False
        
        positions = self.segment_positions[beam_idx]
        current_seg = self.current_segment_idx[beam_idx]
        
        # 检查是否到达下一段的切换点
        if current_seg < len(positions):
            switch_pos = positions[current_seg]
            if current_length >= switch_pos:
                return True
        
        return False
    
    def _handle_segment_switch(self, beam_idx: int):
        """
        处理段切换事件
        
        Args:
            beam_idx: beam 索引
        """
        old_seg = self.current_segment_idx[beam_idx]
        
        # 记录当前段的实际生成量
        self.actual_generated_per_segment[beam_idx][old_seg] = \
            self.generated_since_last_segment[beam_idx]
        
        # 切换到下一段
        self.current_segment_idx[beam_idx] += 1
        self.generated_since_last_segment[beam_idx] = 0
        
        if self.verbose:
            actual_len = self.actual_generated_per_segment[beam_idx][old_seg]
            target_len = self.target_tokens_per_segment[old_seg]
            ratio = actual_len / target_len if target_len > 0 else 0
            print(
                f"[RemainingBudgetEOS] Beam {beam_idx}: "
                f"Segment {old_seg} -> {self.current_segment_idx[beam_idx]} | "
                f"Actual: {actual_len} / Target: {target_len} ({ratio:.2f}x)"
            )
    
    def _get_current_segment_target(self, beam_idx: int) -> int:
        """
        获取当前段的目标长度
        
        Args:
            beam_idx: beam 索引
            
        Returns:
            当前段的目标 token 数
        """
        current_seg = self.current_segment_idx[beam_idx]
        
        # 如果已经超过最后一段，使用最后一段的目标
        if current_seg >= self.num_segments:
            return self.target_tokens_per_segment[-1]
        
        return self.target_tokens_per_segment[current_seg]
    
    def _compute_bias(self, progress: float, is_last_segment: bool) -> float:
        """
        根据 progress 计算 EOS bias
        
        使用分段线性函数：
        - 非最后一段：始终返回 max_negative_bias（强力抑制 EOS）
        - 最后一段：根据 progress 动态调整
          - progress < min_ratio: 强力抑制
          - min_ratio ~ neutral_start: 线性减弱抑制
          - neutral_start ~ neutral_end: 中性区（不干预）
          - neutral_end ~ max_ratio: 线性增强鼓励
          - progress > max_ratio: 强力鼓励
        
        Args:
            progress: 当前段的进度（当前段已生成 / 当前段目标）
            is_last_segment: 是否是最后一段
            
        Returns:
            EOS logit bias
        """
        if not is_last_segment:
            return self.max_negative_bias
        
        if progress < self.min_ratio:
            return self.max_negative_bias
        
        elif progress < self.neutral_start_ratio:
            # 线性插值：从 max_negative_bias 到 0
            t = (progress - self.min_ratio) / (self.neutral_start_ratio - self.min_ratio)
            return self.max_negative_bias * (1.0 - t)
        
        elif progress < self.neutral_end_ratio:
            return 0.0
        
        elif progress < self.max_ratio:
            t = (progress - self.neutral_end_ratio) / (self.max_ratio - self.neutral_end_ratio)
            return self.max_positive_bias * t
        
        else:
            return self.max_positive_bias
    
    def __call__(
        self, 
        input_ids: torch.LongTensor, 
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        应用 EOS bias
        
        Args:
            input_ids: [batch_size, seq_len]
            scores: [batch_size, vocab_size]
            
        Returns:
            修改后的 scores
        """
        batch_size = input_ids.shape[0]
        
        if self.initial_offset is None:
            self.initial_offset = input_ids.shape[1]
            self._initialize_state(batch_size)
            
            if self.verbose:
                print(f"[RemainingBudgetEOS] Detected offset: {self.initial_offset} tokens")
        
        total_length = input_ids.shape[1]
        
        for beam_idx in range(batch_size):
            current_length = total_length - self.initial_offset
            
            if self._detect_segment_switch(current_length, beam_idx):
                self._handle_segment_switch(beam_idx)
            
            self.generated_since_last_segment[beam_idx] = (
                current_length 
                - sum(self.actual_generated_per_segment[beam_idx])
            )
            
            current_seg = self.current_segment_idx[beam_idx]
            
            # 每段的 progress 基于自己的目标，不受前面误差影响
            current_seg_target = self._get_current_segment_target(beam_idx)
            generated_in_current = self.generated_since_last_segment[beam_idx]
            
            # Progress = 当前段已生成 / 当前段目标（而不是剩余预算）
            progress = generated_in_current / max(current_seg_target, 1e-6)
            
            # Clamp to reasonable range
            progress = max(0.0, min(progress, 2.0))
            
            is_last_segment = (current_seg >= self.num_segments - 1)
            
            bias = self._compute_bias(progress, is_last_segment)
            
            scores[beam_idx, self.stop_token_id] += bias
            
            if self.verbose and current_length % 50 == 0:
                seg_status = "LAST" if is_last_segment else f"{current_seg}"
                print(
                    f"[RemainingBudgetEOS] Beam {beam_idx} | "
                    f"Seg {seg_status}/{self.num_segments} | "
                    f"Generated: {generated_in_current} / Target: {current_seg_target} | "
                    f"Progress: {progress:.2f} | "
                    f"Bias: {bias:+.2f}"
                )
        
        return scores

class ResBlock(nn.Module):
    """
    Basic residual convolutional block that uses GroupNorm.
    """

    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(chan, chan, kernel_size=3, padding=1),
            nn.GroupNorm(chan // 8, chan),
            nn.ReLU(),
            nn.Conv1d(chan, chan, kernel_size=3, padding=1),
            nn.GroupNorm(chan // 8, chan)
        )

    def forward(self, x):
        return F.relu(self.net(x) + x)


class GPT2InferenceModel(GPT2PreTrainedModel):
    def __init__(self, config, gpt, text_pos_emb, embeddings, norm, linear, kv_cache=False):
        super().__init__(config)
        # Note: the argument named `text_pos_emb` here actually represents the mel position embedding
        self.transformer = gpt
        self.text_pos_embedding = text_pos_emb
        self.embeddings = embeddings
        self.final_norm = norm
        self.lm_head = nn.Sequential(norm, linear)
        self.kv_cache = kv_cache

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.cached_mel_emb = None
        self.last_attention_map = None
        
        # attn cache
        self.past_attn_cache = None
        self.past_attn_pos = None
        # MAS
        self.mas_mu = None
        # HMM
        self.hmm = None
        self.segment_positions = None  # 记录段切换位置
        self.duration_processor = None  # 存储 RemainingBudgetEOSProcessor 引用
        
        
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(max(1, torch.cuda.device_count())))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def store_mel_emb(self, mel_emb):
        self.cached_mel_emb = mel_emb

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        attention_phase = kwargs.get("attention_phase", None)
        phases_attn_mask_ids = kwargs.get("tokenwise_attention_mask", None)
        input_full_attention_mask = kwargs.get("input_full_attention_mask", False)
        input_attention_masks = kwargs.get("input_attention_masks", None)
        dynamic_cond_mask_idx = kwargs.get("dynamic_cond_mask_idx", None)
        trunc_index = input_attention_masks.shape[0] if input_attention_masks is not None else 0
        device = input_ids.device
        mask_dtype = next(self.parameters()).dtype
        if attention_phase is not None:
            attention_masks = kwargs.get("attention_masks")
            attention_mask_list = []
            for i, phase in enumerate(attention_phase):
                attention_mask_list.append(attention_masks[phase]) # [B, seq_len]
            attention_mask = torch.cat(attention_mask_list, dim=0) # [B*num_beams, seq_len]
            
            attention_mask = F.pad(attention_mask, (0, input_ids.shape[1] - attention_mask.shape[1]), value=1)
            kwargs["attention_mask"] = attention_mask
            # attention mask 右侧补1直到len(input_ids)
            
            if input_full_attention_mask==False:
                kwargs["_4d_attention_mask"] = construct_attn_mask(
                    attention_masks, phases_attn_mask_ids, trunc_index, input_attention_masks if input_full_attention_mask==False else None, dynamic_cond_mask_idx, attention_phase, device, mask_dtype
                )
                kwargs["attention_mask"] = torch.ones(input_ids.shape, device=device, dtype=mask_dtype)
            

        token_type_ids = kwargs.get("token_type_ids", None)  # usually None
        if not self.kv_cache:
            past_key_values = None
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
            
            if "_4d_attention_mask" in kwargs:
                _4d_attention_mask = kwargs["_4d_attention_mask"]
                kwargs["_4d_attention_mask"] = _4d_attention_mask[:, :, -1:, :]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
            
        output_path = kwargs.get("output_path", None)
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "_4d_attention_mask": kwargs.get("_4d_attention_mask", None),
            "attention_phase": attention_phase,
            "output_path": output_path,
        }
        
    def detect_emotion_twist(self, 
                        next_indices, 
                        next_tokens, 
                        emotion_final_phase, 
                        eos_token_id,
                        output_attentions,
                        **model_kwargs):
        """
        
        Args:
            next_indices (torch.Tensor): 下一个时间步的token索引.
            next_tokens (torch.Tensor): 下一个时间步的token id.
            emotion_final_phase (int): 情感的最终phase.
            eos_token_id (list): 结束token id列表.
            method (str): 
                判断情感转折头的方法.
                    - `eos`: 使用 eos token 位置判断.
                    - `hmm`: 使用HMM对齐判断.
                    - `max_head`: 消融选择最大头.
                    - `max_topk`: 消融选择top k头.
                    - `mas`: 使用MAS贪心.
                    
            output_attentions (`tuple(torch.FloatTensor)`): 模型输出的注意力权重.
            

        Returns:
            is_emotion_twist (`bool`): 
                是否发生情感转折.
        """
        method = model_kwargs.get("method", "eos")
        if method == "eos":
            indice_array = next_indices.detach().cpu().numpy()[0]
            token_array = next_tokens.detach().cpu().numpy()[0]
            # token_array 中eos_token_id的位置对应的indice_array
            eos_positions = [i for i, token_id in enumerate(token_array) if token_id in eos_token_id]
            emotion_end_indices = indice_array[eos_positions]
            # emotion_end_indices中，对应的phase尚未达到emotion_final_phase的index
            emotion_twist_indices = [i for i in emotion_end_indices if model_kwargs["attention_phase"][i] < emotion_final_phase]
            is_emotion_twist = len(emotion_twist_indices) > 0
            for i in emotion_twist_indices:
                model_kwargs["attention_phase"][i] += 1
                print(f"Beam {i} enters phase {model_kwargs['attention_phase'][i]}")
            return is_emotion_twist
        elif method == "hmm":
            all_attn = torch.stack(output_attentions, dim=0)  # [num_layers, beamsize, num_heads, seq_len, seq_len]
            
            text_last_token_position = model_kwargs.get('text_last_token_position', None)
            text_bos_idx = text_last_token_position[0]
            text_eos_idx = text_last_token_position[1][-1] + 1
            all_layer_attn = all_attn[:, :, :, -1, text_bos_idx:text_eos_idx] # [num_layers, beamsize, num_heads, seq_len], 只取最后一个token的attn，以及text部分
            
            L, B, H, Sk = all_layer_attn.shape  # L: 层数，H: 头，B: beamsize, Sk: text_len

            # # ==== 每一层选最大的头 ====
            # # 使用标准差
            # max_head_attn = torch.std(all_layer_attn, dim=-1)  # [L, B, H, ]
            # max_head = max_head_attn.argmax(dim=-1)  # [L, B]
            # # 取出结果
            # attn = all_layer_attn.permute(1, 0, 2, 3)  # [B, L, H, Sk ]
            # idx = max_head.permute(1, 0).unsqueeze(-1).unsqueeze(-1)  # [B, L, 1, 1]
            # idx = idx.expand(-1, -1, 1, attn.size(-1))  # [B, L, 1, Sk]
            # attn_seperate = attn.gather(2, idx).squeeze(2)  # [B, L, Sk]

            # # 相当于top1
            # # 对每个头，计算一个分数
            # # H = mean(attn_seperate, dim=-1)  # [B, H]
            # head_importance = torch.mean(attn_seperate, dim=-1)  # [B, H]
            # # 选出最大的头
            # max_head = head_importance.argmax(dim=-1)  # [B, ]
            # # 取出对应的注意力
            # max_head = max_head.view(-1, 1, 1).expand(B, 1, Sk)  # [B, 1, Sk]
            # attn_wo_head = attn_seperate.gather(1, max_head).squeeze(1)  # [B, Sk]
            
            # # ==== 仅输出使用 ====
                
            # # TODO: 输出情感转换内容。
            # self.attn_map_processor.process_emotion_twist_detect(
            #     method=method,
            #     output_path=model_kwargs.get("output_path", None),
            #     text_last_token_position=model_kwargs.get('text_last_token_position', None),
            # )
            
            # TODO: 以上可视化属于增强功能，下面才是核心逻辑。
            
            attention_phase = model_kwargs.get("attention_phase", None)
            text_last_position_array = torch.tensor(
                text_last_token_position[1] - text_last_token_position[0], 
                device=all_layer_attn.device)  # [B, ]
            text_last_pos_per_beam = text_last_position_array[attention_phase]  # [B, ]
            
            
            if self.hmm is None:
                self.hmm = StreamingHMMAligner(num_beams=B, num_text_tokens=Sk, device=all_layer_attn.device)
            
            # 只有在需要保存attention maps时才传递processor
            processor = self.attn_map_processor if model_kwargs.get("save_attention_maps", False) else None
            hmm_center, hmm_idx = self.hmm.step(all_layer_attn, attn_map_processor=processor, **model_kwargs)
            suspect_twist_pos = (hmm_center >= text_last_pos_per_beam + 0.5)  # bool tensor
            if suspect_twist_pos.any():
                indices = torch.where(suspect_twist_pos)[0]
                for i in indices:
                    i = i.item()
                    
                     # 调试：记录段切换位置（减去前置token）
                    current_total_len = all_attn.shape[-1]  # 包含前置token的总长度
                    trunc_index = self.trunc_index  # 从 self 读取前置token数量
                    current_sem_len = current_total_len - trunc_index  # 实际生成的semantic token位置
                    
                    # 根据attention_phase决定段号
                    if self.segment_positions is None:
                        self.segment_positions = dict()
                    if i not in self.segment_positions:
                        self.segment_positions[i] = [current_sem_len]
                    else:
                        if len(self.segment_positions[i]) > model_kwargs["attention_phase"][i]:
                            # 说明之前已经记录过该段切换位置，需要更新
                            self.segment_positions[i][model_kwargs["attention_phase"][i]] = current_sem_len
                        else:
                            self.segment_positions[i].append(current_sem_len)
                    
                    model_kwargs["attention_phase"][i] += 1
                    print(f"Beam {i} enters phase {model_kwargs['attention_phase'][i]} at sem {all_attn.shape[-1]} ")
                   
                # ===== 新增：通知 duration_processor 段切换信息 =====
                if self.duration_processor is not None:
                    self.duration_processor.update_segment_positions(self.segment_positions)
                return True
            return False
        elif method == "max_head":
            all_attn = torch.stack(output_attentions, dim=0)  # [num_layers, beamsize, num_heads, seq_len, seq_len]

            text_last_token_position = model_kwargs.get('text_last_token_position', (34, [34]))
            text_last_token_position = model_kwargs.get('text_last_token_position', None)
            text_bos_idx = text_last_token_position[0]
            text_eos_idx = text_last_token_position[1][-1] + 1
            all_layer_attn = all_attn[:, :, :, -1, text_bos_idx:text_eos_idx] # [num_layers, beamsize, num_heads, seq_len], 只取最后一个token的attn，以及text部分

            L, B, H, Sk = all_layer_attn.shape  # L: 层数，H: 头，B: beamsize, Sk: text_len
            
            # ==== 选最大的头 ====
            # 1. 计算所有层、所有头的标准差 (std)
            # 结果形状: [L, B, H]
            attn_std = torch.mean(all_layer_attn, dim=-1)

            # 2. 调整维度：把 Beam 放到最前，把 Layer 和 Head 展平
            # [L, B, H] -> [B, L, H]
            attn_std = attn_std.permute(1, 0, 2)
            # [B, L, H] -> [B, L*H]
            attn_std_flat = attn_std.reshape(B, -1)

            # 3. 对每个 Beam，在所有层所有头(L*H)中找到 std 最大的那个头的索引
            best_head_idx = attn_std_flat.argmax(dim=-1)  # [B, ]

            # 4. 取出对应的 attention 向量
            # 先把原始 attention 数据也变成 [B, L*H, Sk] 以便 gather
            attn_data = all_layer_attn.permute(1, 0, 2, 3).reshape(B, L*H, Sk)

            # 构造 gather 索引: [B, 1, Sk]
            gather_idx = best_head_idx.view(B, 1, 1).expand(-1, -1, Sk)

            # 提取结果 -> [B, 1, Sk] -> squeeze -> [B, Sk]
            attn_wo_head = attn_data.gather(1, gather_idx).squeeze(1)
            
            # 选出每个beam最高注意力
            # token_idx = attn_wo_head.argmax(dim=-1)  # [B, ]
            # token_value = attn_wo_head.gather(-1, token_idx.unsqueeze(-1)).squeeze(-1)  # [B, ]
            token_idx, token_value = self._update_attn_monotonic(attn_wo_head)

            # 比较的前置工作
            attention_phase = model_kwargs.get("attention_phase", None)
            text_last_position_array = torch.tensor(
                text_last_token_position[1] - text_last_token_position[0], 
                device=attn_wo_head.device)  # [B, ]
            text_last_pos_per_beam = text_last_position_array[attention_phase]  # [B, ]
            
            # TODO: 输出情感转换内容。
            # self.attn_map_processor.process_emotion_twist_detect(
            #     attn_wo_head=attn_wo_head,
            #     method=method,
            #     token_idx=token_idx,
            #     attention_phase=attention_phase,
            #     output_path=model_kwargs.get("output_path", None),
            #     text_last_token_position=model_kwargs.get('text_last_token_position', None),
            # )
            
            # 1. 先获取上一时刻的位置 (Handle step 0 or empty history)
            if self.past_attn_pos is not None and len(self.past_attn_pos) > 0:
                prev_token_idx = self.past_attn_pos[-1]
            else:
                # 如果没有历史记录（第一步），给一个不可能满足条件的值（比如 -100）
                prev_token_idx = torch.full_like(token_idx, -100)
            
            self._update_attn_cache(token_value, token_idx)
            
            suspect_twist_pos = (token_idx > text_last_pos_per_beam) & \
                                (prev_token_idx == token_idx - 1)
                                # 位置超过text最后token的位置, bool tensor
            if suspect_twist_pos.any():
                # 执行情感转折比较
                indices = torch.where(suspect_twist_pos)[0]
                for i in indices:
                    i = i.item()
                
                    # 调试：记录段切换位置（减去前置token）
                    current_total_len = all_attn.shape[-1]  # 包含前置token的总长度
                    trunc_index = self.trunc_index  # 从 self 读取前置token数量
                    current_sem_len = current_total_len - trunc_index  # 实际生成的semantic token位置
                    
                    # 根据attention_phase决定段号
                    if self.segment_positions is None:
                        self.segment_positions = dict()
                    if i not in self.segment_positions:
                        self.segment_positions[i] = [current_sem_len]
                    else:
                        if len(self.segment_positions[i]) > model_kwargs["attention_phase"][i]:
                            # 说明之前已经记录过该段切换位置，需要更新
                            self.segment_positions[i][model_kwargs["attention_phase"][i]] = current_sem_len
                        else:
                            self.segment_positions[i].append(current_sem_len)
                    
                    model_kwargs["attention_phase"][i] += 1
                    print(f"[Max head] Beam {i} enters phase {model_kwargs['attention_phase'][i]} at sem {all_attn.shape[-1]} ")
                # ===== 新增：通知 duration_processor 段切换信息 =====
                if self.duration_processor is not None:
                    self.duration_processor.update_segment_positions(self.segment_positions)
                return True
            return False
        elif method == "max_topk":
            all_attn = torch.stack(output_attentions, dim=0)  # [num_layers, beamsize, num_heads, seq_len, seq_len]

            text_last_token_position = model_kwargs.get('text_last_token_position', None)
            text_bos_idx = text_last_token_position[0]
            text_eos_idx = text_last_token_position[1][-1] + 1
            all_layer_attn = all_attn[:, :, :, -1, text_bos_idx:text_eos_idx] # [num_layers, beamsize, num_heads, seq_len], 只取最后一个token的attn，以及text部分

            L, B, H, Sk = all_layer_attn.shape  # L: 层数，H: 头，B: beamsize, Sk: text_len
            
            # ==== 选出topk个L*H ====
            all_layer_attn = all_layer_attn.permute(0,2,1,3) # [L, H, B, Sk]
            all_layer_attn = all_layer_attn.reshape(-1, B, Sk) # [L * H, B, Sk]
            scores = all_layer_attn.mean(dim=-1)
            topk_scores, topk_indices = torch.topk(scores, k=3, dim=0)
            topk_indices_exp = topk_indices.unsqueeze(-1).expand(-1, -1, Sk) # (topk, B, Sk)
            topk_attn_per_beam = torch.gather(all_layer_attn, 0, topk_indices_exp) #  (topk, B, Sk)
            attn_seperate = topk_attn_per_beam.permute(1, 0, 2)  # [B, topk, Sk]
            
            head_importance = torch.mean(attn_seperate, dim=-1)  # [B, H]
            # 选出top的头
            topk = 3
            topk_heads = torch.topk(head_importance, k=topk, dim=-1).indices  # [B, topk]
            topk_values = torch.topk(head_importance, k=topk, dim=-1).values  # [B, topk]
            # 归一化，使用min-max
            weight = (topk_values - topk_values.min(dim=-1, keepdim=True).values) / (topk_values.max(dim=-1, keepdim=True).values - topk_values.min(dim=-1, keepdim=True).values + 1e-8)
            weight = weight / weight.sum(dim=-1, keepdim=True)  # [B, topk]
            # 取出对应的注意力
            topk_heads = topk_heads.view(-1, topk, 1).expand(B, topk, Sk)  # [B, topk, Sk]
            topk_attn = attn_seperate.gather(1, topk_heads)  # [B, topk, Sk]
            
            attn_wo_head = (topk_attn * weight.unsqueeze(-1)).sum(dim=1)  # [B, Sk]
            
            # 选出每个beam最高注意力
            # token_idx = attn_wo_head.argmax(dim=-1)  # [B, ]
            # token_value = attn_wo_head.gather(-1, token_idx.unsqueeze(-1)).squeeze(-1)  # [B, ]
            token_idx, token_value = self._update_attn_monotonic(attn_wo_head)

            # 比较的前置工作
            attention_phase = model_kwargs.get("attention_phase", None)
            text_last_position_array = torch.tensor(
                text_last_token_position[1] - text_last_token_position[0], 
                device=attn_wo_head.device)  # [B, ]
            text_last_pos_per_beam = text_last_position_array[attention_phase]  # [B, ]
            
            # TODO: 输出情感转换内容。
            if model_kwargs.get("save_attention_maps", False):
                self.attn_map_processor.process_emotion_twist_detect(
                    attn_wo_head=attn_wo_head,
                    method=method,
                    token_idx=token_idx,
                    attention_phase=attention_phase,
                    output_path=model_kwargs.get("output_path", None),
                    text_last_token_position=model_kwargs.get('text_last_token_position', None),
                )
            
            # 1. 先获取上一时刻的位置 (Handle step 0 or empty history)
            if self.past_attn_pos is not None and len(self.past_attn_pos) > 0:
                prev_token_idx = self.past_attn_pos[-1]
            else:
                # 如果没有历史记录（第一步），给一个不可能满足条件的值（比如 -100）
                prev_token_idx = torch.full_like(token_idx, -100)
            
            self._update_attn_cache(token_value, token_idx)

            suspect_twist_pos = (token_idx > text_last_pos_per_beam) & \
                                (prev_token_idx == token_idx - 1)
                                # 位置超过text最后token的位置, bool tensor
            if suspect_twist_pos.any():
                # 执行情感转折比较
                indices = torch.where(suspect_twist_pos)[0]
                for i in indices:
                    i = i.item()
                
                    # 调试：记录段切换位置（减去前置token）
                    current_total_len = all_attn.shape[-1]  # 包含前置token的总长度
                    trunc_index = self.trunc_index  # 从 self 读取前置token数量
                    current_sem_len = current_total_len - trunc_index  # 实际生成的semantic token位置
                    
                    # 根据attention_phase决定段号
                    if self.segment_positions is None:
                        self.segment_positions = dict()
                    if i not in self.segment_positions:
                        self.segment_positions[i] = [current_sem_len]
                    else:
                        if len(self.segment_positions[i]) > model_kwargs["attention_phase"][i]:
                            # 说明之前已经记录过该段切换位置，需要更新
                            self.segment_positions[i][model_kwargs["attention_phase"][i]] = current_sem_len
                        else:
                            self.segment_positions[i].append(current_sem_len)
                    
                    model_kwargs["attention_phase"][i] += 1
                    print(f"[Max head topk] Beam {i} enters phase {model_kwargs['attention_phase'][i]} at sem {all_attn.shape[-1]} ")
                # ===== 新增：通知 duration_processor 段切换信息 =====
                if self.duration_processor is not None:
                    self.duration_processor.update_segment_positions(self.segment_positions)
                return True
            return False
        elif method == "mas":
            # 使用最大head
            all_attn = torch.stack(output_attentions, dim=0)  # [num_layers, beamsize, num_heads, seq_len, seq_len]

            text_last_token_position = model_kwargs.get('text_last_token_position', (34, [34]))
            text_last_token_position = model_kwargs.get('text_last_token_position', None)
            text_bos_idx = text_last_token_position[0]
            text_eos_idx = text_last_token_position[1][-1] + 1
            all_layer_attn = all_attn[:, :, :, -1, text_bos_idx:text_eos_idx] # [num_layers, beamsize, num_heads, seq_len], 只取最后一个token的attn，以及text部分

            L, B, H, Sk = all_layer_attn.shape  # L: 层数，H: 头，B: beamsize, Sk: text_len
            
            # ==== 选最大的头 ====
            # 1. 计算所有层、所有头的标准差 (std)
            # 结果形状: [L, B, H]
            attn_std = torch.mean(all_layer_attn, dim=-1)

            # 2. 调整维度：把 Beam 放到最前，把 Layer 和 Head 展平
            # [L, B, H] -> [B, L, H]
            attn_std = attn_std.permute(1, 0, 2)
            # [B, L, H] -> [B, L*H]
            attn_std_flat = attn_std.reshape(B, -1)

            # 3. 对每个 Beam，在所有层所有头(L*H)中找到 std 最大的那个头的索引
            best_head_idx = attn_std_flat.argmax(dim=-1)  # [B, ]

            # 4. 取出对应的 attention 向量
            # 先把原始 attention 数据也变成 [B, L*H, Sk] 以便 gather
            attn_data = all_layer_attn.permute(1, 0, 2, 3).reshape(B, L*H, Sk)

            # 构造 gather 索引: [B, 1, Sk]
            gather_idx = best_head_idx.view(B, 1, 1).expand(-1, -1, Sk)

            # 提取结果 -> [B, 1, Sk] -> squeeze -> [B, Sk]
            attn_wo_head = attn_data.gather(1, gather_idx).squeeze(1)
            
            # 比较的前置工作
            attention_phase = model_kwargs.get("attention_phase", None)
            text_last_position_array = torch.tensor(
                text_last_token_position[1] - text_last_token_position[0], 
                device=attn_wo_head.device)  # [B, ]
            text_last_pos_per_beam = text_last_position_array[attention_phase]  # [B, ]
            
            mas_mu = self._update_mas_mu_w2(attn_wo_head)
            
            # TODO: 输出情感转换内容。
            if model_kwargs.get("save_attention_maps", False):
                self.attn_map_processor.process_emotion_twist_detect(
                    attn_wo_head=attn_wo_head,
                    method=method,
                    token_idx=mas_mu,
                    attention_phase=attention_phase,
                    output_path=model_kwargs.get("output_path", None),
                    text_last_token_position=model_kwargs.get('text_last_token_position', None),
                )
            suspect_twist_pos = (mas_mu >= text_last_pos_per_beam + 1)  # bool tensor
            if suspect_twist_pos.any():
                indices = torch.where(suspect_twist_pos)[0]
                for i in indices:
                    i = i.item()
                    # 调试：记录段切换位置（减去前置token）
                    current_total_len = all_attn.shape[-1]  # 包含前置token的总长度
                    trunc_index = self.trunc_index  # 从 self 读取前置token数量
                    current_sem_len = current_total_len - trunc_index  # 实际生成的semantic token位置
                    
                    # 根据attention_phase决定段号
                    if self.segment_positions is None:
                        self.segment_positions = dict()
                    if i not in self.segment_positions:
                        self.segment_positions[i] = [current_sem_len]
                    else:
                        if len(self.segment_positions[i]) > model_kwargs["attention_phase"][i]:
                            # 说明之前已经记录过该段切换位置，需要更新
                            self.segment_positions[i][model_kwargs["attention_phase"][i]] = current_sem_len
                        else:
                            self.segment_positions[i].append(current_sem_len)
                    
                    model_kwargs["attention_phase"][i] += 1
                    print(f"[Mas] Beam {i} enters phase {model_kwargs['attention_phase'][i]} at sem {current_sem_len} ")
                # ===== 新增：通知 duration_processor 段切换信息 =====
                if self.duration_processor is not None:
                    self.duration_processor.update_segment_positions(self.segment_positions)
                return True
            return False
        elif method == "wo_align":
            # randomly pick a phase
            current_phases = model_kwargs.get("attention_phase", None)
            current_phases = np.array(current_phases)
            
            n_states = emotion_final_phase + 1
            should_jump = np.random.random(current_phases.shape) < 0.05
            
            # 3. 计算“乱跳”的目标值 (核心技巧)
            # 生成随机偏移量，范围是 [1, n_states)。注意 high 是 exclusive 的
            random_shifts = np.random.randint(1, n_states, size=current_phases.shape)

            #通过 (当前值 + 偏移量) % 总数，得到一个必定不等于当前值的新位置
            jump_targets = (current_phases + random_shifts) % n_states

            # 4. 应用变更：如果 should_jump 为 True 则取 target，否则保持原值
            new_phases = np.where(should_jump, jump_targets, current_phases)
            
            new_phases_list = new_phases.tolist()
            for i in range(len(new_phases_list)):
                if new_phases_list[i] != current_phases[i]:
                    print(f"[Wo align] Beam {i} jumps from phase {current_phases[i]} to {new_phases_list[i]}")
            model_kwargs["attention_phase"] = new_phases_list
            return False      
        else:
            raise ValueError(f"Unknown method {method} for emotion twist detection.")
                
                
    
    def _update_mas_mu(self, attn_wo_head):
        """
        更新mas_mu
        
        Args:
            attn_wo_head (torch.Tensor): 注意力权重. [B, Sk]
        """
        if self.mas_mu is None:
            # 选择最大的
            B, _ = attn_wo_head.shape
            device = attn_wo_head.device
            self.mas_mu = torch.ones(B, device=device) # 初始化为0.0
        else:
            B, S = attn_wo_head.shape
            sensitivity = 1.0  
            look_ahead = 2     # 看 [0, 1, 2]
        
            # 1. 当前锚点
            current_idx = torch.round(self.mas_mu).long()
            
            # 2. 构建前瞻窗口 [0, 1, 2]
            window_offsets = torch.arange(0, look_ahead + 1, device=attn_wo_head.device)
            window_indices = current_idx.unsqueeze(-1) + window_offsets.unsqueeze(0)
            window_indices = window_indices.clamp(max=S - 1)
            
            # 3. 取出原始权重 (不要做任何归一化操作！)
            # weights: [B, 3] -> [w_0, w_1, w_2]
            weights = attn_wo_head.gather(1, window_indices)
            
            # 4. 计算“推力” (Push Force)
            # 逻辑：
            # w_0 (当前位置权重): 贡献 0 推力 (不想动)
            # w_1 (下一格权重):   贡献 1 * w_1 的推力
            # w_2 (下下格权重):   贡献 2 * w_2 的推力
            
            # window_offsets 是 [0, 1, 2]
            # force = w_0*0 + w_1*1 + w_2*2
            push_force = (weights * window_offsets).sum(dim=-1) # [B, ]
            
            # 5. 更新
            # delta 直接等于推力。
            # 如果 w1=0.01, w2=0.01 -> force = 0.03 -> 基本不动
            # 如果 w1=0.8,  w2=0.1  -> force = 1.0  -> 大步向前
            delta = push_force
            
            self.mas_mu = self.mas_mu + sensitivity * delta
            
            # 6. 防越界
            self.mas_mu = self.mas_mu.clamp(max=S - 1)
        return self.mas_mu
    
    def _update_mas_mu_w2(self, attn_wo_head):
        """
        更新mas_mu (窗口改为 [0, 1])
        """
        if self.mas_mu is None:
            # 选择最大的
            B, _ = attn_wo_head.shape
            device = attn_wo_head.device
            self.mas_mu = torch.ones(B, device=device) # 初始化为0.0
        else:
            B, S = attn_wo_head.shape
            sensitivity = 1.0  
            look_ahead = 1     # <--- 修改点：只看 [0, 1]
        
            # 1. 当前锚点 (四舍五入取整)
            current_idx = torch.round(self.mas_mu).long()
            
            # 2. 构建前瞻窗口 [0, 1]
            # window_offsets -> [0, 1]
            window_offsets = torch.arange(0, look_ahead + 1, device=attn_wo_head.device)
            
            # 算出具体的 index: current, current+1
            window_indices = current_idx.unsqueeze(-1) + window_offsets.unsqueeze(0)
            window_indices = window_indices.clamp(max=S - 1)
            
            # 3. 取出权重
            # 先将attn_wo_head normalize
            log_attn = torch.log(attn_wo_head + 1e-8)
            attn_normalized = torch.softmax(log_attn, dim=-1)
            
            # weights: [B, 2] -> [w_0, w_1]
            weights = attn_normalized.gather(1, window_indices)
            
            # 4. 计算“推力”
            # 公式变更为: w_0 * 0 + w_1 * 1
            # 实际上推力就是 w_1 (下一个token的注意力权重)
            push_force = (weights * window_offsets).sum(dim=-1) # [B, ]
            
            # 5. 更新
            # 现在的 delta 就是 w_1。如果 w_1 很大，就会向前走接近 1 格。
            delta = push_force
            
            self.mas_mu = self.mas_mu + sensitivity * delta
            
            # 6. 防越界
            self.mas_mu = self.mas_mu.clamp(max=S - 1)
            
        return self.mas_mu
            
        
    def _update_attn_cache(self, token_value, token_idx):
        """
        _update_attn_cache 的 Docstring
        
        :param self: 说明
        :param token_values: 说明
        :param token_indices: 说明
        """
        N = 5
        if self.past_attn_cache is None:
            self.past_attn_cache = token_value.unsqueeze(0)  # [1, ]
        else:
            # 合并
            self.past_attn_cache = torch.cat([self.past_attn_cache, token_value.unsqueeze(0)], dim=0)  # [total_beams_so_far, ]
            if self.past_attn_cache.shape[0] > N:
                self.past_attn_cache = self.past_attn_cache[1:]  # 只保留N个
                
        if self.past_attn_pos is None:
            self.past_attn_pos = token_idx.unsqueeze(0)  # [1, ]
        else:
            # 合并
            self.past_attn_pos = torch.cat([self.past_attn_pos, token_idx.unsqueeze(0)], dim=0)  # [total_beams_so_far, ]
            if self.past_attn_pos.shape[0] > N:
                self.past_attn_pos = self.past_attn_pos[1:]  # 只保留N个
                
    def _update_attn_monotonic(self, attn_wo_head):
        if self.past_attn_pos is None:
            # 初始化为最大注意力的位置
            B = attn_wo_head.shape[0]
            token_idx = torch.ones(B, dtype=torch.long, device=attn_wo_head.device)
            token_value = attn_wo_head.gather(-1, token_idx.unsqueeze(-1)).squeeze(-1)
            return token_idx, token_value
        else:
            # 根据上一步做出选择
            # ================= [修改开始] =================
            # 1. 先获取上一时刻的位置
            B = attn_wo_head.shape[0]
            if self.past_attn_pos is not None and len(self.past_attn_pos) > 0:
                prev_token_idx = self.past_attn_pos[-1]

            # 2. 【需求】只比较 当前位置(prev) 和 下一位置(prev+1)
            Sk = attn_wo_head.shape[-1]
            
            val_curr = attn_wo_head.gather(1, prev_token_idx.unsqueeze(-1)).squeeze(-1)
            
            next_cand_idx = (prev_token_idx + 1).clamp(max=Sk - 1)
            val_next = attn_wo_head.gather(1, next_cand_idx.unsqueeze(-1)).squeeze(-1)
            
            move_mask = val_next > val_curr
            token_idx = torch.where(move_mask, next_cand_idx, prev_token_idx)
            token_value = torch.where(move_mask, val_next, val_curr)

            return token_idx, token_value
    
    @property
    def attn_map_processor(self):
        if not hasattr(self, "_attn_map_processor"):
            self._attn_map_processor = AttentionMapProcessor()
        return self._attn_map_processor

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            _4d_attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
    ):
        assert self.cached_mel_emb is not None
        assert inputs_embeds is None  # Not supported by this inference model.
        assert labels is None  # Training not supported by this inference model.
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # Create embedding
        mel_len = self.cached_mel_emb.shape[1]
        if input_ids.shape[1] != 1:
            text_inputs = input_ids[:, mel_len:]
            text_emb = self.embeddings(text_inputs)
            text_emb = text_emb + self.text_pos_embedding(text_emb)
            # print(f"text pos emb shape: {self.text_pos_embedding(text_emb).shape}")
            if self.cached_mel_emb.shape[0] != text_emb.shape[0]:
                mel_emb = self.cached_mel_emb.repeat_interleave(
                    text_emb.shape[0] // self.cached_mel_emb.shape[0], 0
                )
            else:  # this outcome only occurs once per loop in most cases
                mel_emb = self.cached_mel_emb
            emb = torch.cat([mel_emb, text_emb], dim=1)
        else:
            emb = self.embeddings(input_ids)
            emb = emb + self.text_pos_embedding.get_fixed_embedding(
                attention_mask.shape[1] - mel_len, attention_mask.device
            )
        transformer_outputs = self.transformer(
            inputs_embeds=emb,
            past_key_values=past_key_values,
            attention_mask=attention_mask if _4d_attention_mask is None else _4d_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=True,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        
        # 获取attention 并记录
        attention_phase = kwargs.get("attention_phase", None)
        output_path = kwargs.get("output_path", None)
        
        # self.attn_map_processor.process_attention_map(
        #     transformer_outputs.attentions,
        #     attention_phase,
        #     output_path,
        # )

        # Set device for model parallelism
        if self.model_parallel:
            if torch.backends.mps.is_available():
                self.to(self.transformer.first_device)
            else:
                torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + transformer_outputs[1:]

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past
        )


class ConditioningEncoder(nn.Module):
    def __init__(self,
                 spec_dim,
                 embedding_dim,
                 attn_blocks=6,
                 num_attn_heads=4,
                 do_checkpointing=False,
                 mean=False):
        super().__init__()
        attn = []
        self.init = nn.Conv1d(spec_dim, embedding_dim, kernel_size=1)
        for a in range(attn_blocks):
            attn.append(AttentionBlock(embedding_dim, num_attn_heads))
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim
        self.do_checkpointing = do_checkpointing
        self.mean = mean

    def forward(self, x):
        h = self.init(x)
        h = self.attn(h)
        if self.mean:
            return h.mean(dim=2)
        else:
            return h
            # return h[:, :, 0]


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x):
        sl = x.shape[1]
        return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)


def build_hf_gpt_transformer(layers, model_dim, heads, max_mel_seq_len, max_text_seq_len, checkpointing):
    """
    GPT-2 implemented by the HuggingFace library.
    """
    from transformers import GPT2Config, GPT2Model
    gpt_config = GPT2Config(vocab_size=256,  # Unused.
                            n_positions=max_mel_seq_len + max_text_seq_len,
                            n_ctx=max_mel_seq_len + max_text_seq_len,
                            n_embd=model_dim,
                            n_layer=layers,
                            n_head=heads,
                            gradient_checkpointing=checkpointing,
                            use_cache=not checkpointing)
    gpt = GPT2Model(gpt_config)
    # Override the built in positional embeddings
    del gpt.wpe
    gpt.wpe = functools.partial(null_position_embeddings, dim=model_dim)
    # Built-in token embeddings are unused.
    del gpt.wte
    return gpt, LearnedPositionEmbeddings(max_mel_seq_len, model_dim), LearnedPositionEmbeddings(max_text_seq_len, model_dim), \
        None, None


class MelEncoder(nn.Module):
    def __init__(self, channels, mel_channels=80, resblocks_per_reduction=2):
        super().__init__()
        self.channels = channels
        self.encoder = nn.Sequential(nn.Conv1d(mel_channels, channels // 4, kernel_size=3, padding=1),
                                     nn.Sequential(*[ResBlock(channels // 4) for _ in range(resblocks_per_reduction)]),
                                     nn.Conv1d(channels // 4, channels // 2, kernel_size=3, stride=2, padding=1),
                                     nn.GroupNorm(channels // 16, channels // 2),
                                     nn.ReLU(),
                                     nn.Sequential(*[ResBlock(channels // 2) for _ in range(resblocks_per_reduction)]),
                                     nn.Conv1d(channels // 2, channels, kernel_size=3, stride=2, padding=1),
                                     nn.GroupNorm(channels // 8, channels),
                                     nn.ReLU(),
                                     nn.Sequential(*[ResBlock(channels) for _ in range(resblocks_per_reduction)]),
                                     )
        self.reduction = 4

    def forward(self, x):
        for e in self.encoder:
            x = e(x)
        return x.permute(0, 2, 1)


class UnifiedVoice(nn.Module):
    def __init__(self, layers=8, model_dim=512, heads=8, max_text_tokens=120, max_mel_tokens=250, max_conditioning_inputs=1,
                 mel_length_compression=1024, number_text_tokens=256,
                 start_text_token=0, stop_text_token=1, number_mel_codes=8194, start_mel_token=8192, stop_mel_token=8193,
                 train_solo_embeddings=False, use_mel_codes_as_input=True,
                 checkpointing=True, types=1,
                 condition_num_latent=32, condition_type="perceiver", condition_module=None, emo_condition_module=None):
        """
        Args:
            layers: Number of layers in transformer stack.
            model_dim: Operating dimensions of the transformer
            heads: Number of transformer heads. Must be divisible by model_dim. Recommend model_dim//64
            max_text_tokens: Maximum number of text tokens that will be encountered by model.
            max_mel_tokens: Maximum number of MEL tokens that will be encountered by model.
            max_conditioning_inputs: Maximum number of conditioning inputs provided to the model. If (1), conditioning input can be of format (b,80,s), otherwise (b,n,80,s).
            mel_length_compression: The factor between <number_input_samples> and <mel_tokens>. Used to compute MEL code padding given wav input length.
            number_text_tokens:
            start_text_token:
            stop_text_token:
            number_mel_codes:
            start_mel_token:
            stop_mel_token:
            train_solo_embeddings:
            use_mel_codes_as_input:
            checkpointing:
            condition_type: perceiver, gst or default encoder
        """
        super().__init__()
        self.number_text_tokens = number_text_tokens
        self.start_text_token = start_text_token
        self.stop_text_token = stop_text_token
        self.number_mel_codes = number_mel_codes
        self.start_mel_token = start_mel_token
        self.stop_mel_token = stop_mel_token
        self.layers = layers
        self.heads = heads
        self.max_mel_tokens = max_mel_tokens
        self.max_text_tokens = max_text_tokens
        self.model_dim = model_dim
        self.max_conditioning_inputs = max_conditioning_inputs
        self.mel_length_compression = mel_length_compression
        self.condition_type = condition_type
        self.cond_num = condition_num_latent
        self.cond_mask_pad = nn.ConstantPad1d((self.cond_num, 0), True)
        self.emo_cond_mask_pad = nn.ConstantPad1d((1, 0), True)
        if condition_type == "perceiver":
            self.conditioning_encoder = ConditioningEncoder(1024, model_dim, num_attn_heads=heads)
            self.perceiver_encoder = PerceiverResampler(model_dim, dim_context=model_dim, num_latents=self.cond_num)
        elif condition_type == "conformer_perceiver" or condition_type == "conformer_encoder":
            self.conditioning_encoder = ConformerEncoder(input_size=1024,
                                                         output_size=condition_module['output_size'],
                                                         linear_units=condition_module['linear_units'],
                                                         attention_heads=condition_module['attention_heads'],
                                                         num_blocks=condition_module['num_blocks'],
                                                         input_layer=condition_module['input_layer'])
            if condition_type == "conformer_perceiver":
                self.perceiver_encoder = PerceiverResampler(model_dim, dim_context=condition_module['output_size'],
                                                            ff_mult=condition_module['perceiver_mult'],
                                                            heads=condition_module['attention_heads'],
                                                            num_latents=self.cond_num)
        else:
            self.conditioning_encoder = ConditioningEncoder(1024, model_dim, num_attn_heads=heads, mean=True)

        self.emo_conditioning_encoder = ConformerEncoder(input_size=1024,
                                                         output_size=emo_condition_module['output_size'],
                                                         linear_units=emo_condition_module['linear_units'],
                                                         attention_heads=emo_condition_module['attention_heads'],
                                                         num_blocks=emo_condition_module['num_blocks'],
                                                         input_layer=emo_condition_module['input_layer'])
        self.emo_perceiver_encoder = PerceiverResampler(1024, dim_context=emo_condition_module['output_size'],
                                                            ff_mult=emo_condition_module['perceiver_mult'],
                                                            heads=emo_condition_module['attention_heads'],
                                                            num_latents=1)



        self.text_embedding = nn.Embedding(self.number_text_tokens * types + 1, model_dim)
        self.emo_layer = nn.Linear(model_dim, model_dim)
        self.emovec_layer = nn.Linear(1024, model_dim)

        if use_mel_codes_as_input:
            self.mel_embedding = nn.Embedding(self.number_mel_codes, model_dim)
        else:
            self.mel_embedding = MelEncoder(model_dim, resblocks_per_reduction=1)
        self.gpt, self.mel_pos_embedding, self.text_pos_embedding, self.mel_layer_pos_embedding, self.text_layer_pos_embedding = \
            build_hf_gpt_transformer(layers, model_dim, heads, self.max_mel_tokens + 2 + self.max_conditioning_inputs,
                                     self.max_text_tokens + 2, checkpointing)
        if train_solo_embeddings:
            self.mel_solo_embedding = nn.Parameter(torch.randn(1, 1, model_dim) * .02, requires_grad=True)
            self.text_solo_embedding = nn.Parameter(torch.randn(1, 1, model_dim) * .02, requires_grad=True)
        else:
            self.mel_solo_embedding = 0
            self.text_solo_embedding = 0

        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, self.number_text_tokens * types + 1)
        self.mel_head = nn.Linear(model_dim, self.number_mel_codes)

        self.speed_emb = nn.Embedding(2, model_dim)
        self.speed_emb.weight.data.normal_(mean=0.0, std=0.0)

        # Initialize the embeddings per the GPT-2 scheme
        embeddings = [self.text_embedding]
        if use_mel_codes_as_input:
            embeddings.append(self.mel_embedding)
        for module in embeddings:
            module.weight.data.normal_(mean=0.0, std=.02)

    def post_init_gpt2_config(self, use_deepspeed=False, kv_cache=False, half=False):
        seq_length = self.max_mel_tokens + self.max_text_tokens + 2
        gpt_config = GPT2Config(
            vocab_size=self.number_mel_codes,
            n_positions=seq_length,
            n_ctx=seq_length,
            n_embd=self.model_dim,
            n_layer=self.layers,
            n_head=self.heads,
            gradient_checkpointing=False,
            use_cache=True,
        )
        self.inference_model = GPT2InferenceModel(
            gpt_config,
            self.gpt,
            self.mel_pos_embedding,
            self.mel_embedding,
            self.final_norm,
            self.mel_head,
            kv_cache=kv_cache,
        )
        if use_deepspeed and half and torch.cuda.is_available():
            import deepspeed
            self.ds_engine = deepspeed.init_inference(model=self.inference_model,
                                                      mp_size=1,
                                                      replace_with_kernel_inject=True,
                                                      dtype=torch.float16)
            self.inference_model = self.ds_engine.module.eval()
        elif use_deepspeed and torch.cuda.is_available():
            import deepspeed
            self.ds_engine = deepspeed.init_inference(model=self.inference_model,
                                                      mp_size=1,
                                                      replace_with_kernel_inject=True,
                                                      dtype=torch.float32)
            self.inference_model = self.ds_engine.module.eval()
        else:
            self.inference_model = self.inference_model.eval()

        # self.inference_model = PrunedGPT2InferenceModel(gpt_config, self.gpt, self.mel_pos_embedding, self.mel_embedding, self.final_norm, self.mel_head)
        self.gpt.wte = self.mel_embedding

    def build_aligned_inputs_and_targets(self, input, start_token, stop_token):
        inp = F.pad(input, (1, 0), value=start_token)
        tar = F.pad(input, (0, 1), value=stop_token)
        return inp, tar

    def set_mel_padding(self, mel_input_tokens, mel_lengths):
        """
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with STOP_MEL_TOKEN in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        for b in range(len(mel_lengths)):
            # Due to the convolutional nature of how these tokens are generated,
            # it would be best if the model predicts a token past the actual last token.
            actual_end = mel_lengths[b]
            if actual_end < mel_input_tokens.shape[-1]:
                mel_input_tokens[b, actual_end:] = self.stop_mel_token
        return mel_input_tokens

    def set_text_padding(self, text_input_tokens, text_lengths):
        """
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with STOP_MEL_TOKEN in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        for b in range(len(text_lengths)):
            # Due to the convolutional nature of how these tokens are generated,
            # it would be best if the model predicts a token past the actual last token.
            actual_end = text_lengths[b]
            if actual_end < text_input_tokens.shape[-1]:
                text_input_tokens[b, actual_end:] = self.stop_text_token
        return text_input_tokens

    def get_logits(self, speech_conditioning_inputs, first_inputs, first_head, second_inputs=None, second_head=None, get_attns=False, return_latent=False, attention_mask=None):
        if second_inputs is not None:
            emb = torch.cat([speech_conditioning_inputs, first_inputs, second_inputs], dim=1)
        else:
            emb = torch.cat([speech_conditioning_inputs, first_inputs], dim=1)

        gpt_out = self.gpt(inputs_embeds=emb, return_dict=True, output_attentions=get_attns, attention_mask=attention_mask)
        if get_attns:
            return gpt_out.attentions

        offset = speech_conditioning_inputs.shape[1]
        enc = gpt_out.last_hidden_state[:, offset:]
        enc = self.final_norm(enc)

        if return_latent:
            return enc[:, :first_inputs.shape[1]], enc[:, -second_inputs.shape[1]:]

        first_logits = enc[:, :first_inputs.shape[1]]
        first_logits = first_head(first_logits)
        first_logits = first_logits.permute(0, 2, 1)
        if second_inputs is not None:
            second_logits = enc[:, -second_inputs.shape[1]:]
            second_logits = second_head(second_logits)
            second_logits = second_logits.permute(0, 2, 1)
            return first_logits, second_logits
        else:
            return first_logits

    def get_conditioning(self, speech_conditioning_input, cond_mel_lengths=None):
        if self.condition_type == "perceiver":
            if speech_conditioning_input.ndim == 4:
                speech_conditioning_input = speech_conditioning_input.squeeze(1)
            speech_conditioning_input = self.conditioning_encoder(speech_conditioning_input)  # (b, d, s)
            conds = self.perceiver_encoder(speech_conditioning_input.transpose(1, 2))  # (b, 32, d)
        elif self.condition_type == "conformer_perceiver":
            speech_conditioning_input, mask = self.conditioning_encoder(speech_conditioning_input.transpose(1, 2),
                                                                        cond_mel_lengths)  # (b, s, d), (b, 1, s)
            if self.condition_type == "conformer_perceiver":
                # conds_mask = torch.cat([torch.ones((mask.shape[0], self.cond_num), dtype=torch.bool), mask.squeeze(1)], dim=1)
                conds_mask = self.cond_mask_pad(mask.squeeze(1))
                conds = self.perceiver_encoder(speech_conditioning_input, conds_mask)  # (b, 32, d)
        elif self.condition_type == "gst":
            if speech_conditioning_input.ndim == 4:
                speech_conditioning_input = speech_conditioning_input.squeeze(1)
            conds = self.gst_encoder(speech_conditioning_input.transpose(1, 2))  # (b, 1, d)
        else:
            speech_conditioning_input = (
                speech_conditioning_input.unsqueeze(1)
                if len(speech_conditioning_input.shape) == 3
                else speech_conditioning_input
            )
            conds = []
            for j in range(speech_conditioning_input.shape[1]):
                conds.append(self.conditioning_encoder(speech_conditioning_input[:, j]))
            conds = torch.stack(conds, dim=1)
            conds = conds.mean(dim=1)
            conds = conds.unsqueeze(1)
        return conds


    def get_emo_conditioning(self, speech_conditioning_input, cond_mel_lengths=None):
        speech_conditioning_input, mask = self.emo_conditioning_encoder(speech_conditioning_input.transpose(1, 2),
                                                                        cond_mel_lengths)  # (b, s, d), (b, 1, s)
        conds_mask = self.emo_cond_mask_pad(mask.squeeze(1))
        conds = self.emo_perceiver_encoder(speech_conditioning_input, conds_mask)  # (b, 1, d)
        return conds.squeeze(1)
    
    def get_duration_embeddings(self, lengths: torch.Tensor, check: bool = False):
        """
        Thanks to https://github.com/JarodMica/index-tts/commit/a9f0125531124eccb8b3e8568d1b9c711a1e7564#diff-456187d8672e8d50cdf6ffbbe9e0f139242e20fad9ea52b3ce32e2cfe41677c1R7
        """
        max_index = self.mel_pos_embedding.emb.num_embeddings - 1
        clamped = lengths.clamp(max=max_index).long()
        
        import os
        if check and os.path.exists("mel_pos_embedding.csv") == False:
            # 将所有位置embedding输出csv
            import pandas as pd
            all_pos_emb = self.mel_pos_embedding.emb.weight.data.cpu().numpy()
            df = pd.DataFrame(all_pos_emb)
            df.to_csv("mel_pos_embedding.csv", index=False, header=False)
        return self.mel_pos_embedding.emb(clamped)
        
    def forward(self, speech_conditioning_latent, text_inputs, text_lengths, mel_codes, mel_codes_lengths, emo_speech_conditioning_latent,
                cond_mel_lengths=None, emo_cond_mel_lengths=None, emo_vecs=None, use_speed=None, do_spk_cond=False, attention_mask=None):
        """
        Forward pass that uses both text and voice in either text conditioning mode or voice conditioning mode

        Args:
            speech_conditioning_input: MEL float tensor, (b,1024); or a `list` of such tensors
            text_inputs: long tensor, (b,t)
            text_lengths: long tensor, (b,)
            mel_inputs:  long tensor, (b,m)
            wav_lengths: long tensor, (b,)

        Returns:
            If return_attentions is specified, only logits are returned.
            If return_latent is specified, loss & logits are not computed or returned. Only the predicted latents are returned.
        """

        if do_spk_cond:
            speech_conditioning_latent = self.get_conditioning(speech_conditioning_latent.transpose(1,2), cond_mel_lengths)
        else:
            speech_conditioning_latent = speech_conditioning_latent

        # if emo_vec is None:
        #     emo_vec_syn_ori = self.get_emo_conditioning(emo_speech_conditioning_latent.transpose(1,2), emo_cond_mel_lengths)
        #     emo_vec_syn = self.emovec_layer(emo_vec_syn_ori)
        #     emo_vec = self.emo_layer(emo_vec_syn)

        text_inputs = self.set_text_padding(text_inputs, text_lengths)
        text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)

        mel_codes = self.set_mel_padding(mel_codes, mel_codes_lengths)
        mel_codes = F.pad(mel_codes, (0, 1), value=self.stop_mel_token)

        duration_emb = self.speed_emb(torch.zeros_like(use_speed))
        duration_emb_half = self.speed_emb(torch.ones_like(use_speed))
        # conds = torch.cat((speech_conditioning_latent + emo_vec.unsqueeze(1), duration_emb_half.unsqueeze(1), duration_emb.unsqueeze(1)), 1)
        # conds = torch.cat((speech_conditioning_latent, duration_emb_half.unsqueeze(1), duration_emb.unsqueeze(1)), 1)
        conds_latents = []
        for i, emo_vec in enumerate(emo_vecs):
            if isinstance(speech_conditioning_latent, list):
                conds_latent = torch.cat(
                    (speech_conditioning_latent[i] + emo_vec.unsqueeze(1), 
                     duration_emb_half.unsqueeze(1), 
                     duration_emb.unsqueeze(1)), 1)
            else:
                conds_latent = torch.cat((speech_conditioning_latent + emo_vec.unsqueeze(1), duration_emb_half.unsqueeze(1), duration_emb.unsqueeze(1)), 1)
            conds_latents.append(conds_latent)
        conds = torch.cat(conds_latents, dim=1)

        text_inputs, text_targets = self.build_aligned_inputs_and_targets(text_inputs, self.start_text_token, self.stop_text_token)
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)
        mel_codes, mel_targets = self.build_aligned_inputs_and_targets(mel_codes, self.start_mel_token, self.stop_mel_token)

        mel_emb = self.mel_embedding(mel_codes)
        mel_emb = mel_emb + self.mel_pos_embedding(mel_codes)

        text_logits, mel_logits = self.get_logits(conds, text_emb, self.text_head, mel_emb, self.mel_head, get_attns=False, return_latent=True, attention_mask = attention_mask)
        return mel_logits[:, :-2]  # Despite the name, these are not logits. Strip off the two tokens added by this forward pass.

    def prepare_gpt_inputs(
        self,
        conditional_latents: torch.Tensor,
        text_inputs_list: torch.Tensor,
        full_text: bool = False,
    ):
        
        """
        Prepare the inputs for the GPT2InferenceModel to generate.
        Args:
            conds_latent: (b, 32, dim) audio conditioning embedding by `get_conditioning()`
            text_inputs: (b, L)
            full_text(`bool`): whether to use full text or only each segment
        Returns:
            input_ids: (b, s+1) the input ids for the GPT2InferenceModel.generate()
            inputs_embeds: (b, s+1, dim) the input embeddings for the GPT2InferenceModel.forward()
            attention_mask: (b, s+1) the attention mask for the GPT2InferenceModel.generate()
        """
        b = text_inputs_list[0].shape[0]
        L = sum([t.shape[1] for t in text_inputs_list])
        device = text_inputs_list[0].device
        # single_cond = conditional_latents[0].ndim == 3 and conditional_latents[0].shape[0] == 1
        # if not single_cond:
        #     assert conditional_latents.shape[0] == b, f"batch size mismatch: {conditional_latents.shape[0]} vs {b}"
        batched_mel_emb = []
        attention_masks = []
        attention_masks_full_view = []
        cond_lengths = [t.shape[1] for t in conditional_latents]
        # 每个cond长度相同
        assert all([l == cond_lengths[0] for l in cond_lengths]), f"cond lengths not equal: {cond_lengths}"
        cond_len = cond_lengths[0]
        cond_cum_lengths = [0]
        for length in cond_lengths:
            cond_cum_lengths.append(cond_cum_lengths[-1] + length)
        total_cond_len = cond_cum_lengths[-1]
        target_len = total_cond_len + L + 2 # 留2个给
        print(f"target_len: {target_len}, cond_lens: {cond_lengths}, L: {L}")
        


        # valid_mask = (text_inputs[i] != self.stop_text_token) & (text_inputs[i] != self.start_text_token)
        # text_input = text_inputs[i][valid_mask]
        text_input = torch.cat(text_inputs_list, dim=1)
        text_input = F.pad(text_input, (1, 0), value=self.start_text_token)
        text_input = F.pad(text_input, (0, 1), value=self.stop_text_token)
        text_input_pos = torch.arange(0, text_input.size(-1), device=device)
        text_emb = self.text_embedding(text_input) + self.text_pos_embedding.emb(text_input_pos)
        # concatenate [conditional latents][text embeddings]
        print(f"conditional_latents shape: {conditional_latents[0].shape}, text_emb shape: {text_emb.shape}")
        conds_text_emb = [c.squeeze(0) for c in conditional_latents] + [
            text_emb.squeeze(0),
        ]

        switching_points = [] # 情绪转换点
        progress = 0
        for idx, t in enumerate(text_inputs_list):
            progress += t.shape[1]
            switching_points.append(progress)
        # +1 for the start_mel_token
        attention_mask = torch.ones(target_len+1, dtype=torch.long, device=device)
        # check this text input is padded
        padding: int = L + 2 - text_input.size(-1)
        
        # pad left of [cond][text] -> [pad][cond][text]
        if padding > 0:
            pad = torch.zeros((padding, conditional_latents.size(-1)), dtype=text_emb.dtype, device=device) # [p, dim]
            conds_text_emb.insert(0, pad)
            attention_mask[:padding] = 0
        # if len(conditional_latents) > 1: # 默认先只关注第一个cond，以及第一段text
        for i, sp in enumerate(switching_points):
            new_attention_mask = attention_mask.clone()
            if i > 0:
                new_attention_mask[padding:padding+cond_cum_lengths[i]] = 0
            if i < len(switching_points) - 1:
                new_attention_mask[padding+cond_cum_lengths[i+1]:padding+total_cond_len] = 0
            # attention_mask[padding+cond_len:padding+cond_len*len(conditional_latents)] = 0
            attention_masks_full_view.append(new_attention_mask.clone().unsqueeze(0))
            if not full_text:
                new_attention_mask[padding+total_cond_len+sp+1:-2] = 0
            attention_masks.append(new_attention_mask.unsqueeze(0))
        
        mel_emb = torch.cat(conds_text_emb) #[s, dim]
        assert mel_emb.shape[0] == target_len, f"mel_emb.shape: {mel_emb.shape}, target_len: {target_len}"
        batched_mel_emb.append(mel_emb)
        
        input_masks = []
        dynamic_cond_mask_idx = []
        for i in range(len(conditional_latents)):
            input_mask = attention_mask.clone()
            if i > 0:
                input_mask[padding:padding+cond_len*i] = 0
            if i < len(conditional_latents) - 1:
                input_mask[padding+cond_len*(i+1):padding+cond_len*len(conditional_latents)] = 0
            for _ in range(cond_len):
                input_masks.append(input_mask.unsqueeze(0))

        dynamic_cond_mask_idx.append(len(input_masks))
        input_masks.append(attention_mask.unsqueeze(0))  # for the start token
        for i, sp in enumerate(switching_points):
            input_mask = attention_mask.clone()
            if i > 0:
                input_mask[padding:padding+cond_len*i] = 0
            if i < len(switching_points) - 1:
                input_mask[padding+cond_len*(i+1):padding+cond_len*len(conditional_latents)] = 0
            length = sp
            if i != 0:
                length -= switching_points[i-1]
            for _ in range(length):
                input_masks.append(input_mask.unsqueeze(0))
        dynamic_cond_mask_idx.append(len(input_masks))
        input_masks.append(attention_mask.unsqueeze(0))  # for the stop token
        dynamic_cond_mask_idx.append(len(input_masks))
        input_masks.append(attention_mask.unsqueeze(0))  # for the mel_start token
        input_attention_masks = torch.cat(input_masks, dim=0)
                

        # [b, s, dim]
        batched_mel_emb = torch.stack(batched_mel_emb, dim=0)
        # [b, s+1]
        # attention_mask = torch.stack(attention_masks, dim=0)
        # [b, s+1]
        fake_inputs = torch.ones(
            (
                batched_mel_emb.shape[0],
                batched_mel_emb.shape[1] + 1,  # +1 for the start_mel_token
            ),
            dtype=torch.long,
            device=device,
        )
        fake_inputs[:, -1] = self.start_mel_token
        
        text_last_token_position = padding + total_cond_len + np.array(switching_points)
        text_last_token_position = (total_cond_len, text_last_token_position)
        
        return fake_inputs, batched_mel_emb, attention_masks, attention_masks_full_view, text_last_token_position, input_attention_masks, dynamic_cond_mask_idx

    def inference_speech(self, speech_condition, text_inputs_list, emo_speech_condition=None, cond_lengths=None, emo_cond_lengths=None, emo_vecs=None, use_speed=False, input_tokens=None, num_return_sequences=1,
                         max_generate_length=None, typical_sampling=False, typical_mass=.9, input_full_attention_mask=False, target_duration_tokens=None, **hf_generate_kwargs):
        """
        Args:
            speech_condition: (b, d, frames) or (d, frames)
            text_inputs: (b, L)
            cond_mel_lengths: lengths of the conditioning mel spectrograms in shape (b,) or (1,)
            input_tokens: additional tokens for generation in shape (b, s) or (s,)
            max_generate_length: limit the number of generated tokens
            hf_generate_kwargs: kwargs for `GPT2InferenceModel.generate(**hf_generate_kwargs)`
        """

        if not isinstance(speech_condition, list) and speech_condition.ndim == 2:
            speech_condition = speech_condition.unsqueeze(0)
            
        if emo_speech_condition is None:
            emo_speech_condition = speech_condition
            
        if cond_lengths is None:
            cond_lengths = torch.tensor([speech_condition.shape[-1]], device=speech_condition.device)
            
        if emo_cond_lengths is None:
            emo_cond_lengths = torch.tensor([emo_speech_condition.shape[-1]], device=speech_condition.device) 

        # speech cond latent part
        speech_conditioning_latent_list = None
        if isinstance(speech_condition, list) and isinstance(cond_lengths, list):
            speech_conditioning_latent_list = []
            for i, speech_cond in enumerate(speech_condition):
                latent = self.get_conditioning(speech_cond.transpose(1,2), cond_lengths[i])
                speech_conditioning_latent_list.append(latent)
        else:
            speech_conditioning_latent = self.get_conditioning(speech_condition.transpose(1,2), cond_lengths)
        
        if emo_vecs is None:
            print('compute emo vec')
            emo_vec = self.get_emo_conditioning(emo_speech_condition.transpose(1,2), emo_cond_lengths)
            emo_vec = self.emovec_layer(emo_vec)
            emo_vec = self.emo_layer(emo_vec)
            emo_vecs = [emo_vec]
        else:
            print('Use the specified emotion vector')
        for text_inputs in text_inputs_list:
            print("text input shape:", text_inputs.shape)


        tmp = torch.zeros(text_inputs_list[0].size(0)).to(text_inputs_list[0].device)
        duration_emb =  self.speed_emb(torch.zeros_like(tmp).long())
        duration_emb_half = self.speed_emb(torch.ones_like(tmp).long())
        
        conds_latents = []
        global_duration_cursor = 0
        print("[DEBUG] target_duration_tokens (model side) =", target_duration_tokens)

        for i, emo_vec in enumerate(emo_vecs):
            tmp = torch.zeros(text_inputs.size(0), device=text_inputs.device)
            duration_free = self.speed_emb(torch.zeros_like(tmp).long())

            if target_duration_tokens is not None:
                seg_len = int(target_duration_tokens[i])
                global_duration_cursor += seg_len
                if i < 10: 
                    print(
                        f"[DurationCtrl] seg={i} | "
                        f"seg_len={seg_len} | "
                        f"global_cursor={global_duration_cursor}"
                    )
                t = max(1, min(global_duration_cursor, self.max_mel_tokens - 1))
                duration_idx = torch.full(
                    (text_inputs.size(0),),
                    t,
                    device=text_inputs.device,
                    dtype=torch.long
                )
                duration_ctrl = self.get_duration_embeddings(duration_idx, check=True)
            else:
                duration_ctrl = self.speed_emb(torch.ones_like(tmp).long())

            if speech_conditioning_latent_list is not None:
                conds_latent = torch.cat(
                    (
                        speech_conditioning_latent_list[i] + emo_vec.unsqueeze(1),
                        duration_emb.unsqueeze(1),
                        duration_ctrl.unsqueeze(1)
                    ),
                    dim=1
                )
            else:
                conds_latent = torch.cat(
                    (
                        speech_conditioning_latent + emo_vec.unsqueeze(1),
                        duration_emb.unsqueeze(1),
                        duration_ctrl.unsqueeze(1)
                    ),
                    dim=1
                )
            conds_latents.append(conds_latent)
            
        print(f"Emo cond length: {emo_vecs[0].shape}")
        full_text = not (hf_generate_kwargs.get("method", "eos") == "eos")
        input_ids, inputs_embeds, attention_masks, attention_masks_full_view, text_last_token_position, input_attention_masks, dynamic_cond_mask_idx = self.prepare_gpt_inputs(conds_latents, text_inputs_list, full_text=full_text)
        print(f"attention_mask: {attention_masks}")
        self.inference_model.store_mel_emb(inputs_embeds)
        # if input_tokens is None:
        inputs = input_ids
        # else:
        #     if input_tokens.ndim == 1:
        #         input_tokens = input_tokens.unsqueeze(0)
        #     assert num_return_sequences % input_tokens.shape[0] == 0, \
        #             "The num_return_sequences must be divisible by the batch number of input_tokens"
        #     assert num_return_sequences % text_inputs.shape[0] == 0, \
        #             "The num_return_sequences must be divisible by the batch number of text_inputs"
        #     b = num_return_sequences // input_ids.shape[0]
        #     if b > 1:
        #         input_ids = input_ids.repeat(b, 1)
        #         attention_mask = attention_mask.repeat(b, 1)
        #     input_tokens = input_tokens.repeat(num_return_sequences // input_tokens.shape[0], 1)
        #     inputs = torch.cat([input_ids, input_tokens], dim=1)
        #     attention_mask = F.pad(attention_mask, (0, input_tokens.shape[1]), value=1)
        
        trunc_index = inputs.shape[1]
        # 将 trunc_index 存储到 inference_model 中，供段切换时使用
        self.inference_model.trunc_index = trunc_index
        logits_processor = LogitsProcessorList()
        if typical_sampling:
            # employ custom typical sampling
            if not (typical_mass > 0.0 and typical_mass < 1.0):
                raise ValueError(f"`typical_mass` has to be a float > 0 and < 1, but is {typical_mass}")
            min_tokens_to_keep = 2 if hf_generate_kwargs.get("num_beams", 1) > 1 else 1
            logits_processor.append(TypicalLogitsWarper(mass=typical_mass, min_tokens_to_keep=min_tokens_to_keep))

        # ========== Duration Control via RemainingBudgetEOS ==========
        if target_duration_tokens is not None:
            # 确保是列表格式
            if not isinstance(target_duration_tokens, list):
                target_duration_tokens = [target_duration_tokens]
            
            # 使用新的 RemainingBudgetEOSProcessor
            duration_processor = RemainingBudgetEOSProcessor(
                target_tokens_per_segment=target_duration_tokens,  # 直接传入列表
                stop_token_id=self.stop_mel_token,
                verbose=True,
                # 超参数（根据需要调整）
                min_ratio=0.5,              # 当前段进度 < 50% 时强力抑制 EOS
                neutral_ratio=(0.7, 1.0),   # 80%-120% 不干预
                max_ratio=1.2,              # 超过 200% 时强力鼓励 EOS
                max_negative_bias=-5.0,     # 抑制强度（可以调整）
                max_positive_bias=15.0      # 鼓励强度（可以调整）
            )
            
            # 重要：将 processor 存储到 inference_model 中
            self.inference_model.duration_processor = duration_processor
            
            logits_processor.append(duration_processor)
            
            print(f"\n>> [RemainingBudgetEOS] Enabled")
            print(f"   Num segments: {len(target_duration_tokens)}")
            print(f"   Target per segment: {target_duration_tokens}")
            print(f"   Total target: {sum(target_duration_tokens)} semantic tokens")

        # 从generate kwargs中提取save_attention_maps参数（不传给generate）
        save_attention_maps = hf_generate_kwargs.pop("save_attention_maps", False)
        
        max_length = (trunc_index + self.max_mel_tokens - 1) if max_generate_length is None else trunc_index + max_generate_length
        output = self.inference_model.generate(inputs, 
                                            bos_token_id=self.start_mel_token, pad_token_id=self.stop_mel_token,
                                            eos_token_id=self.stop_mel_token, attention_masks=attention_masks,
                                            max_length=max_length, logits_processor=logits_processor,
                                            num_return_sequences=num_return_sequences,
                                            text_last_token_position=text_last_token_position,
                                            input_full_attention_mask=input_full_attention_mask,
                                            input_attention_masks=input_attention_masks,
                                            dynamic_cond_mask_idx=dynamic_cond_mask_idx,
                                            save_attention_maps=save_attention_maps,
                                            **hf_generate_kwargs)
        
        # 只有在需要时才保存attention maps
        if save_attention_maps:
            self.inference_model.attn_map_processor.save_attention_maps(input_embeds_len=inputs_embeds.shape[1],)
        # self.inference_model.hmm = None   # 清除hmm实例
        
        output.sequences = output.sequences[:, trunc_index:]  # remove the input part
        print(f"Generated output shape: {output.sequences.shape}, inputs shape: {inputs.shape}")
    
        # min_dtype = torch.finfo(mask_dtype).min
        # causal_masks = []
        # for i in range(len(output.attention_mask_ids)):
        #     output_len = len(output.attention_mask_ids[i]) + trunc_index
            
            
        #     if input_full_attention_mask:
        #         print("Using full attention mask")
        #         attention_mask_list = [torch.ones((1, output_len), dtype=torch.long, device=output.sequences.device)] * trunc_index
        #     else:
        #         print("Using input attention masks")
        #         attention_mask_list = [ torch.nn.functional.pad(input_attention_masks[i], (0, output_len - input_attention_masks[i].shape[0]), value=1).unsqueeze(0) for i in range(trunc_index)]
            
        #     for mask_id in output.attention_mask_ids[i]:
        #         attention_mask_list.append(torch.nn.functional.pad(attention_masks_full_view[mask_id], (0, output_len - attention_masks_full_view[mask_id].shape[1]), value=1))
        #     attention_mask = torch.cat(attention_mask_list, dim=0).unsqueeze(0).unsqueeze(0)
        #     print("attention_mask shape:", attention_mask.shape)

        #     mask_dtype = next(self.gpt.parameters()).dtype
        #     causal_mask = torch.full(
        #         (output_len, output_len), fill_value=min_dtype, dtype=mask_dtype, device=output.sequences.device
        #     )
        #     causal_mask = torch.triu(causal_mask, diagonal=1)
        #     # causal_mask *= torch.arange(output_len, device=output.sequences.device) > cache_position.reshape(-1, 1)
        #     causal_mask = causal_mask[None, None, :, :]
        #     padding_mask = causal_mask + attention_mask
        #     padding_mask = padding_mask == 0
        #     causal_mask = causal_mask.masked_fill(
        #         padding_mask, min_dtype
        #     )
        #     causal_masks.append(causal_mask)
        causal_mask = construct_attn_mask(attention_masks_full_view, output.attention_mask_ids, trunc_index, input_attention_masks if input_full_attention_mask==False else None, None, None, output.sequences.device, next(self.gpt.parameters()).dtype)
        print("causal_mask shape:", causal_mask.shape)

        # ========== 调试：统计每段的 semantic token 数量 ==========
        print("\n" + "="*70)
        print("Multi-segment Semantic Token:")
        
        total_semantic_tokens = output.sequences.shape[1] - 1
        selected_beam = output.beam_indices[0][0]
        seg_lens = []
        print(f"Total semantic tokens: {total_semantic_tokens}")
        
        # 从 HMM 的段切换记录中获取位置
        if hasattr(self.inference_model, 'segment_positions') and self.inference_model.segment_positions is not None:
            # 使用之前记录的段切换位置
            for beam_id, positions in self.inference_model.segment_positions.items():
                print(f"\nBeam {beam_id}:")
                if beam_id == selected_beam:
                    print(f"* selected beam *")
                all_positions = [0] + positions + [total_semantic_tokens]
                
                for seg_idx in range(len(all_positions) - 1):
                    start_pos = all_positions[seg_idx]
                    end_pos = all_positions[seg_idx + 1]
                    seg_len = end_pos - start_pos
                    percentage = (seg_len / total_semantic_tokens * 100) if total_semantic_tokens > 0 else 0
                    
                    print(f"  Segment {seg_idx + 1}: {seg_len} tokens ({percentage:.1f}%) - location [{start_pos}, {end_pos})")
                    
                    if beam_id == selected_beam:
                        seg_lens.append(seg_len)
                
                print(f"  Total: {total_semantic_tokens} tokens")
            
            # 清空记录
            if self.inference_model.duration_processor is not None:
                self.inference_model.duration_processor = None
            self.inference_model.segment_positions = None
        else:
            print("No switching record")
            print(f"  hasattr check: {hasattr(self.inference_model, 'segment_positions')}")
            if hasattr(self.inference_model, 'segment_positions'):
                print(f"  segment_positions value: {self.inference_model.segment_positions}")
        
        print("="*70 + "\n")
        # ========== 调试输出结束 ==========
        
        # if isinstance(output, torch.Tensor):
        #     print(f"Generated output shape: {output[:, trunc_index:].shape}")
        #     return output[:, trunc_index:], speech_conditioning_latent, attention_mask
        # GenerateOutput
        # 临时保存一下causal mask， 保存到mask.csv中
        # import pandas as pd
        # pd.DataFrame(causal_mask[0,0].cpu().numpy()).to_csv('mask.csv', index=False, header=False)
        if speech_conditioning_latent_list is not None:
            speech_conditioning_latent = speech_conditioning_latent_list
        return output.sequences, speech_conditioning_latent, causal_mask, seg_lens

    def get_emovec(self, emo_speech_conditioning_latent, emo_cond_lengths):
        emo_vec_syn_ori = self.get_emo_conditioning(emo_speech_conditioning_latent.transpose(1,2), emo_cond_lengths)
        emo_vec_syn = self.emovec_layer(emo_vec_syn_ori)
        emo_vec = self.emo_layer(emo_vec_syn)
        return emo_vec

    def merge_emovec(self, speech_conditioning_latent, emo_speech_conditioning_latent, cond_lengths, emo_cond_lengths, alpha = 1.0):
        emo_vec = self.get_emovec(emo_speech_conditioning_latent, emo_cond_lengths)
        base_vec = self.get_emovec(speech_conditioning_latent, cond_lengths)

        out = base_vec + alpha * (emo_vec - base_vec)
        return out
