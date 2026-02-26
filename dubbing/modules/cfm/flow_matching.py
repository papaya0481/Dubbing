import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Any
from transformers import set_seed
from .DiT import LipSyncDiT


@dataclass
class DiTConfig:
    in_channels: int
    hidden_dim: int
    num_heads: int
    depth: int
    cond_dim: int | None = None
    mu_dim: int | None = None
    dropout: float = 0.1
    ff_mult: int = 4
    long_skip_connection: bool = False
    phoneme_vocab_size: int = 8194
    lip_dim: int = 512
    spk_dim: int | None = None
    out_channels: int | None = None
    static_chunk_size: int = 50
    num_decoding_left_chunks: int = 2


@dataclass
class CFMConfig:
    t_scheduler: str = "linear"
    training_cfg_rate: float = 0.1
    inference_cfg_rate: float = 0.5


@dataclass
class LipSyncCFMConfig:
    DiT: DiTConfig
    CFM: CFMConfig = field(default_factory=CFMConfig)

    @classmethod
    def from_args(cls, args: Any) -> "LipSyncCFMConfig":
        dit_args = args.DiT
        dit_cfg = DiTConfig(
            in_channels=dit_args.in_channels,
            hidden_dim=dit_args.hidden_dim,
            num_heads=dit_args.num_heads,
            depth=dit_args.depth,
            cond_dim=getattr(dit_args, "cond_dim", None) or dit_args.hidden_dim,
            mu_dim=getattr(dit_args, "mu_dim", None),
            dropout=getattr(dit_args, "dropout", 0.1),
            ff_mult=getattr(dit_args, "ff_mult", 4),
            long_skip_connection=getattr(dit_args, "long_skip_connection", False),
            phoneme_vocab_size=getattr(dit_args, "phoneme_vocab_size", 72),
            lip_dim=getattr(dit_args, "lip_dim", 512),
            spk_dim=getattr(dit_args, "spk_dim", None),
            out_channels=getattr(dit_args, "out_channels", None),
            static_chunk_size=getattr(dit_args, "static_chunk_size", 50),
            num_decoding_left_chunks=getattr(dit_args, "num_decoding_left_chunks", 2),
        )

        cfm_args = getattr(args, "CFM", None)
        cfm_cfg = CFMConfig(
            t_scheduler=getattr(cfm_args, "t_scheduler", "linear"),
            training_cfg_rate=getattr(cfm_args, "training_cfg_rate", 0.1),
            inference_cfg_rate=getattr(cfm_args, "inference_cfg_rate", 0.5),
        )

        return cls(DiT=dit_cfg, CFM=cfm_cfg)

class LipSyncCFM(nn.Module):
    def __init__(self, args: LipSyncCFMConfig | Any):
        super().__init__()
        cfg = args if isinstance(args, LipSyncCFMConfig) else LipSyncCFMConfig.from_args(args)

        self.sigma_min = 1e-6
        dit_cfg = cfg.DiT

        cond_dim = getattr(dit_cfg, "cond_dim", dit_cfg.hidden_dim)
        self.estimator = LipSyncDiT(args=dit_cfg)

        # Used only for temporary fallback cond construction (phoneme+lip -> cond_dim).
        # Output must match cond_dim (not necessarily hidden_dim).
        self.cond_adapter = nn.Conv1d(2 * dit_cfg.hidden_dim, cond_dim, kernel_size=1)

        cfm_cfg = cfg.CFM
        self.t_scheduler = cfm_cfg.t_scheduler
        self.training_cfg_rate = cfm_cfg.training_cfg_rate
        self.inference_cfg_rate = cfm_cfg.inference_cfg_rate

        self.criterion = nn.MSELoss()

    def _build_condition(self, stretched_mel, phoneme_ids=None, lip_embedding=None, cond=None):
        # TODO: 设计更合理的条件融合策略，而不是简单的优先级覆盖
        if cond is not None:
            return cond
        
        if phoneme_ids is None:
            raise ValueError("缺少条件输入: phoneme_ids")
        
        phoneme_feat = self.estimator.phoneme_embed(phoneme_ids)  # [B, T, D]
        B, T, D = phoneme_feat.shape
        
        if lip_embedding is not None:
            lip_feat = self.estimator.lip_proj(lip_embedding)  # [B, T, D]
        else:
            lip_feat = torch.zeros_like(phoneme_feat)
            
        fused_cond = torch.cat(
            [phoneme_feat, lip_feat], dim=-1
        )   # [B, T, D1+D2]
        
        fused_cond = fused_cond.transpose(1, 2)  # [B, 2D, T]
        cond_out = self.cond_adapter(fused_cond)  # [B, cond_dim, T]

        return cond_out

    def forward_estimator(self, x, mask, mu, t, spks=None, cond=None, streaming=False):
        return self.estimator(x=x, mask=mask, mu=mu, t=t, spks=spks, cond=cond, streaming=streaming)
        
    def forward(self, clean_mel, stretched_mel, phoneme_ids, lip_embedding, x_lens, cond=None, spks=None):
        """
        训练阶段 Forward
        x0 (Source): Stretched Mel (Noisy Prior)
        x1 (Target): Clean Mel (Ground Truth)
        """
        B, _, T = clean_mel.shape
        
        # -------------------------------------------------------
        # 1. 定义流的起点 x0 (Source Distribution)
        # -------------------------------------------------------
        # IndexTTS2 策略：起点不是纯噪声，而是 "拉伸Mel + 噪声"
        # 这样模型是从一个“大概正确”的地方开始修，而不是从虚无开始
        epsilon = torch.randn_like(clean_mel)
        x0 = stretched_mel + epsilon 
        
        # 目标 x1 就是 clean_mel
        x1 = clean_mel

        # -------------------------------------------------------
        # 2. 随机采样时间步 t
        # -------------------------------------------------------
        t = torch.rand(B, 1, 1, device=clean_mel.device)
        # 训练时同步 t_scheduler，避免 train/infer 分布不一致
        if self.t_scheduler == "cosine":
            t = 1 - torch.cos(t * 0.5 * torch.pi)
        
        # -------------------------------------------------------
        # 3. 构建中间状态 x_t (Interpolation)
        # -------------------------------------------------------
        # Optimal Transport Path (直线路径)
        # x_t = (1 - t) * x0 + t * x1
        x_t = (1 - (1 - self.sigma_min) * t) * x0 + t * x1
        
        # 理想的速度场 vector field (Target)
        # u_t = d(x_t)/dt = x1 - (1 - sigma_min) * x0
        # 简单理解就是指向目标的向量
        target_velocity = x1 - (1 - self.sigma_min) * x0

        # -------------------------------------------------------
        # 4. Condition Masking (关键技巧！)
        # -------------------------------------------------------
        # 随机丢弃 stretched_mel 条件，强迫模型学会看音素和唇形
        # 否则模型可能会学会恒等映射 (Identity Mapping)
        drop_prob = self.training_cfg_rate
        if self.training:
            drop_mask = (torch.rand(B, device=clean_mel.device) < drop_prob)    # [B] bool
            drop_mask = drop_mask[:, None, None]  # [B,1,1] for broadcasting
        else:
            drop_mask = torch.zeros(B, device=clean_mel.device).bool()  # 推理阶段不丢条件
            drop_mask = drop_mask[:, None, None]
            
        # drop_mel = drop_mask.float()   # [B,1,1] for [B,C,T]
        # input_stretched_mel = stretched_mel * (1.0 - drop_mel)
        input_stretched_mel = stretched_mel

        cond_input = self._build_condition(
            stretched_mel=input_stretched_mel,
            phoneme_ids=phoneme_ids,
            lip_embedding=lip_embedding,
            cond=cond,
        )

        # CFG: also drop cond for the same samples (match inference unconditional branch)
        if drop_mask is not None:
            drop_cond = drop_mask.float()  # [B,1,1]
            cond_input = cond_input * (1.0 - drop_cond)

        # -------------------------------------------------------
        # 5. 模型预测
        # -------------------------------------------------------
        # 注意：这里输入给网络的 input_stretched_mel 是作为“参考图纸”
        # x_t 是“正在雕刻的石头”
        pred_velocity = self.estimator(
            x=x_t,
            mask=x_lens,
            mu=input_stretched_mel,
            t=t.view(clean_mel.shape[0]),  # (B,) – squeeze is fragile for B=1
            spks=spks,
            cond=cond_input,
        )

        # -------------------------------------------------------
        # 6. 计算 Loss
        # -------------------------------------------------------
        # 只在有效长度内计算 loss
        loss = 0
        for i in range(B):
            length = x_lens[i]
            loss += self.criterion(
                pred_velocity[i, :, :length], 
                target_velocity[i, :, :length]
            )
        loss /= B
        
        return loss

    @torch.inference_mode()
    def inference(
        self,
        stretched_mel,
        phoneme_ids,
        lip_embedding,
        x_lens,
        steps=10,
        temperature=1.0,
        cfg_scale=None,
        cond=None,
        spks=None,
        seed=None,
        streaming=False,
    ):
        """
        推理阶段
        """
        if seed is not None:
            set_seed(int(seed))

        B, _, _ = stretched_mel.size()
        device = stretched_mel.device
        
        # -------------------------------------------------------
        # 1. 构造起点 (Initial Noise)
        # -------------------------------------------------------
        # 同样从 "拉伸Mel + 噪声" 开始
        # temperature 控制随机性：越小越接近原始拉伸Mel，越大越自由
        noise = torch.randn_like(stretched_mel) * temperature
        x = stretched_mel + noise
        
        cond_input = self._build_condition(
            stretched_mel=stretched_mel,
            phoneme_ids=phoneme_ids,
            lip_embedding=lip_embedding,
            cond=cond,
        )

        # 时间步列表 (0 -> 1)
        t_span = torch.linspace(0, 1, steps + 1, device=device, dtype=stretched_mel.dtype)
        if self.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

        cfg_scale = self.inference_cfg_rate if cfg_scale is None else cfg_scale
        return self.solve_euler(
            x,
            t_span=t_span,
            mu=stretched_mel,
            mask=x_lens,
            cond=cond_input,
            spks=spks,
            cfg_scale=cfg_scale,
            streaming=streaming,
        )

    def solve_euler(self, x, t_span, mu, mask, cond, spks=None, cfg_scale=1.5, streaming=False):
        B = x.size(0)

        for step in range(1, len(t_span)):
            # 直接从预计算的 t_span 读取当前评估点和步长，避免浮点累积误差
            t_val = t_span[step - 1]
            dt = t_span[step] - t_val
            t_batch = t_val.unsqueeze(0).repeat(B)

            if cfg_scale > 0:
                x_in = torch.cat([x, x], dim=0)
                t_in = torch.cat([t_batch, t_batch], dim=0)

                if mask.dim() == 1:
                    mask_in = torch.cat([mask, mask], dim=0)
                else:
                    mask_in = torch.cat([mask, mask], dim=0)

                mu_in = torch.cat([mu, mu], dim=0)
                cond_in = torch.cat([cond, torch.zeros_like(cond)], dim=0)

                if spks is not None:
                    spks_in = torch.cat([spks, torch.zeros_like(spks)], dim=0)
                else:
                    spks_in = None

                dphi_dt = self.estimator(
                    x=x_in,
                    mask=mask_in,
                    mu=mu_in,
                    t=t_in,
                    spks=spks_in,
                    cond=cond_in,
                    streaming=streaming,
                )
                dphi_dt, cfg_dphi_dt = dphi_dt.chunk(2, dim=0)
                dphi_dt = (1.0 + cfg_scale) * dphi_dt - cfg_scale * cfg_dphi_dt
            else:
                dphi_dt = self.estimator(
                    x=x,
                    mask=mask,
                    mu=mu,
                    t=t_batch,
                    spks=spks,
                    cond=cond,
                    streaming=streaming,
                )

            x = x + dt * dphi_dt

        return x