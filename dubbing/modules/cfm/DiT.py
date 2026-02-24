import torch
import torch.nn as nn
import math
from indextts.s2mel.modules.gpt_fast.model import ModelArgs, Transformer
from indextts.s2mel.modules.commons import sequence_mask

class LipSyncDiT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_dim = args.DiT.hidden_dim
        self.num_heads = args.DiT.num_heads
        self.depth = args.DiT.depth
        self.mel_dim = args.DiT.in_channels  # 通常是 80
        
        # -------------------------------------------------------
        # 1. 条件特征提取器
        # -------------------------------------------------------
        
        # A. 音素 Embedding
        # 假设 phoneme_vocab_size 比如 256
        self.phoneme_embedder = nn.Embedding(args.DiT.phoneme_vocab_size, self.hidden_dim)
        
        # B. 唇形 Projection
        # 假设 lip_dim 是 512 或其他维度，映射到 hidden_dim 以便融合
        self.lip_projector = nn.Linear(args.DiT.lip_dim, self.hidden_dim)

        # C. 时间步 Embedding
        self.t_embedder = TimestepEmbedder(self.hidden_dim)

        # -------------------------------------------------------
        # 2. 融合层 (Fusion Layer)
        # -------------------------------------------------------
        # 输入通道计算:
        # x (Noisy State) [80] + Stretched Mel [80] + Phoneme [Hidden] + Lip [Hidden]
        fusion_input_dim = self.mel_dim + self.mel_dim + self.hidden_dim + self.hidden_dim
        
        # 负责将拼接后的超级向量压缩回 Transformer 的 hidden_dim
        self.input_projection = nn.Linear(fusion_input_dim, self.hidden_dim)

        # -------------------------------------------------------
        # 3. Transformer Backbone (保持原样)
        # -------------------------------------------------------
        model_args = ModelArgs(
            block_size=4096, 
            n_layer=self.depth,
            n_head=self.num_heads,
            dim=self.hidden_dim,
            head_dim=self.hidden_dim // self.num_heads,
            vocab_size=1, # 不使用 token embedding，直接用 continuous input
            uvit_skip_connection=False
        )
        self.transformer = Transformer(model_args)
        
        # 位置编码 buffer
        self.register_buffer("input_pos", torch.arange(4096))

        # -------------------------------------------------------
        # 4. 输出层
        # -------------------------------------------------------
        # 预测速度场 vector field，维度与 mel_dim 一致
        self.final_layer = nn.Linear(self.hidden_dim, self.mel_dim)

    def forward(self, x, t, stretched_mel, phoneme_ids, lip_embedding, x_lens):
        """
        Args:
            x: [B, 80, T] - 当前流的状态 (Noisy Input)
            t: [B] - 时间步
            stretched_mel: [B, 80, T] - 拉伸的 Mel (Condition/Prior)
            phoneme_ids: [B, T] - 音素 ID
            lip_embedding: [B, T, Lip_Dim] - 唇形特征
            x_lens: [B] - 长度
        """
        B, _, T = x.size()

        # 1. 维度调整：全部转为 [B, T, C]
        x = x.transpose(1, 2)              # [B, T, 80]
        stretched_mel = stretched_mel.transpose(1, 2) # [B, T, 80]

        # 2. 处理条件特征
        t_emb = self.t_embedder(t)         # [B, Hidden]
        ph_feat = self.phoneme_embedder(phoneme_ids) # [B, T, Hidden]
        lip_feat = self.lip_projector(lip_embedding) # [B, T, Hidden]

        # 3. 特征拼接 (Early Fusion)
        # 将所有对齐的时序特征拼在一起
        # x_in shape: [B, T, 80+80+Hidden+Hidden]
        x_in = torch.cat([x, stretched_mel, ph_feat, lip_feat], dim=-1)

        # 4. 投影到 Transformer 维度
        x_in = self.input_projection(x_in) # [B, T, Hidden]

        # 5. 注入时间步信息 (作为 Token 或者是 AdaLN，这里演示作为 Token 拼在最前面)
        # 也可以像原来的 DiT 代码一样通过 AdaLN 注入，这里为了简化逻辑使用 concat token 方式
        # 或者直接加到 x_in 上 (Broadcasting)
        x_in = x_in + t_emb.unsqueeze(1) 

        # 6. Transformer Forward
        # 创建 mask
        mask = sequence_mask(x_lens, max_len=T).to(x.device).unsqueeze(1).unsqueeze(1) # [B, 1, 1, T]
        input_pos = self.input_pos[:T]
        
        # 这里的 mask 处理可能需要根据你的 Transformer 实现调整
        # 如果是 causal mask (GPT)，需要 causal=True；如果是 DiT (Bi-directional)，causal=False
        x_out = self.transformer(x_in, input_pos=input_pos, mask=mask if args.DiT.is_causal else None)

        # 7. 输出投影
        output = self.final_layer(x_out) # [B, T, 80]
        
        return output.transpose(1, 2) # [B, 80, T]

# 需要包含之前的 TimestepEmbedder 类定义
class TimestepEmbedder(nn.Module):
    # ... (保持原代码不变) ...
    pass