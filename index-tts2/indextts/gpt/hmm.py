import torch
import torch.nn.functional as F
from .attn_map import AttentionMapProcessor

class StreamingHMMAligner:
    def __init__(self,
                 num_beams,
                 num_text_tokens, 
                 transition_prob=0.1,
                 sigma=1.2,           
                 window_size=None, 
                 device='cpu'):
        """
        初始化对齐器
        
        Args:
            num_text_tokens: Text 序列的长度 (N)
            transition_prob: p, 转移概率 (0.3 表示有 30% 概率向后走一步)
            sigma: 高斯分布的标准
            window_size: 高斯核的窗口大小
        """
        self.Sk = num_text_tokens
        self.B = num_beams
        self.p = transition_prob
        self.device = device
        
        # 初始化隐状态 belief state
        self.belief = torch.zeros((self.B, self.Sk), device=self.device)
        self.belief[:, 1] = 1.0  # 初始时刻，全部概率集中在第一个位置
        
        # 用于记录历史对齐轨迹
        self.history = []
        self.transition_matrix = self._build_transition_matrix().to(self.device)
        
        # 预计算高斯卷积核 (用于 Step B 和 Step C)
        self.kernel_radius = int(3 * sigma) if window_size is None else window_size // 2
        self.gaussian_kernel = self._build_gaussian_kernel(self.kernel_radius, sigma).to(self.device)
        
        # 优化: 预分配张量
        self.indices = torch.arange(self.Sk, device=self.device, dtype=self.belief.dtype)
        
    @staticmethod
    def _normalize(tensor, dim=-1, eps=1e-30):
        """
        利用softmax，使用log后的数值归一化
        """
        log_tensor = torch.log(tensor + eps)
        normed = F.softmax(log_tensor, dim=dim)
        return normed
        
    
    def _build_gaussian_kernel(self, radius, sigma):
        """
        构建离散高斯卷积核 g
        """
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        # 高斯公式 exp(-x^2 / 2sigma^2)
        kernel = torch.exp(- (x ** 2) / (2 * sigma ** 2))
        # 归一化，保证核的积分为 1
        kernel = self._normalize(kernel, dim=0)
        # Reshape for conv1d: [Out_Channel, In_Channel, Kernel_Size] -> [1, 1, K]
        return kernel.view(1, 1, -1)
    
    def _build_transition_matrix(self):
        """
        构建状态转移矩阵
        """
        transition_matrix = torch.zeros((self.Sk, self.Sk), device=self.device)
        
        # 主对角线
        transition_matrix = torch.eye(self.Sk, device=self.device) * (1 - self.p)  # 保持原地的概率
        # 上对角线 (向前走一步)
        upper_diag = torch.full((self.Sk - 1,), self.p, device=self.device)
        transition_matrix += torch.diag(upper_diag, diagonal=1)  # 添加上对角线
        
        transition_matrix[-1, -1] = 1.0
        return transition_matrix
    
    def _apply_gaussian(self, tensor_bn):
        """
        对输入的分布进行高斯平滑 (Convolution)
        Args:
            tensor_bn: [B, N]
        Returns:
            smoothed: [B, N]
        """
        B, N = tensor_bn.shape
        # Conv1d 需要输入 [B, Channels, Length]
        x = tensor_bn.unsqueeze(1) 
        
        # Padding: 'same' padding 效果，左右各补 radius
        x_pad = F.pad(x, (self.kernel_radius, self.kernel_radius), mode='replicate')
        
        # 卷积
        out = F.conv1d(x_pad, self.gaussian_kernel)
        
        # 这里的 padding 处理可能导致边缘极微小的误差，但在流式对齐中可忽略
        # 截取回 [B, N]
        return out.squeeze(1)

    def _predict_prior(self):
        """
        步骤 1: 状态转移 (Time Update)
        根据物理规律，推测当前可能在哪。
        规则：要么停在原地 (1-p)，要么向前走一步 (p)。
        """
        # 构造移位后的分布 (向前走一步)
        # 也就是 belief[i] 移动到 belief[i+1]
        prior = torch.matmul(self.belief, self.transition_matrix)
        return prior

    def _select_best_head(self, prior, attention_heads):
        """
        步骤 2: 观测值筛选 (Observation Selection)
        从 L*H 个头中，选出最符合 prior 的那个。
        
        Args:
            attention_heads: shape [num_layers * num_heads, B, N]
        """
        # 将 prior 广播或通过循环计算与每个 head 的相似度
        # Metric: 点积 (Dot Product) 最能衡量重叠程度
        
        # attention_heads 形状假设为 (K, B, N)
        # prior 形状为 (B, N)
        # scores: (K, B)
        scores = torch.sum(attention_heads * prior.unsqueeze(0), dim=-1)
        
        # best_idx: (B,)
        best_idx = torch.argmax(scores, dim=0)
        
        # Gather best_head: (B, N)
        b_indices = torch.arange(self.B, device=self.device)
        best_head = attention_heads[best_idx, b_indices, :]
        
        return best_head, best_idx
    
    def _select_best_head_loglik(self, prior, attention_heads):
        """
        步骤 2: 使用对数似然选择最佳头
        Score = sum( head[j] * log(prior[j]) )
        这实际上是计算 Cross Entropy 的负数部分。
        
        Return:
            best_head_norm: 归一化后的最佳头 (B, N)
            best_idx: 最佳头的索引 (B,)
            best_head: 原始最佳头 (B, N)
        """
        # 防止 log(0)
        epsilon = 1e-10
        log_prior = torch.log(prior + epsilon) # (B, N)
        
        attn_norm = self._normalize(attention_heads, dim=-1) # (K, B, N)
        # 计算所有头的得分: shape (L*H, B)
        # attention_heads: (K, B, N), log_prior: (B, N)
        # dot product: sum(head_i * log_prior)
        scores = torch.sum(attn_norm * log_prior.unsqueeze(0), dim=-1)
        
        # center < 1的不考虑，
        center = torch.sum(self.indices.unsqueeze(0) * attn_norm, dim=-1) # (K, B)
        mask = center < 1.0
        scores = scores.masked_fill(mask, float('-inf'))
        
        best_idx = torch.argmax(scores, dim=0) # (B,)
        
        # 调试用，直接返回最大的那个头
        avg = torch.mean(attention_heads, dim=-1)
        avg_best_idx = torch.argmax(avg, dim=0)
        
        # Gather best_head: (B, N)
        b_indices = torch.arange(self.B, device=self.device)
        best_head_norm = attn_norm[best_idx, b_indices, :]  # shape (B, N)
        if (torch.sum(self.indices.unsqueeze(0) * best_head_norm, dim=-1) < 1.0).any():
            import pdb; pdb.set_trace()
        best_head = attention_heads[avg_best_idx, b_indices, :]
        
        return best_head_norm, best_idx, best_head

    def step(self, 
             attention_matrix_stack,
             attn_map_processor: AttentionMapProcessor = None,
             **model_kwargs):
        """
        流式处理一步
        
        Args:
            attention_matrix_stack ([L*H, B, Sk] 或 [L, B, H, Sk]): 当前 sem token 对所有 text token 的注意力。
            attn_map_processor(`AttentionMapProcessor`): 用于记录attn_map 
                                       
        Return:
            current_alignment (Tensor [B]), selected_head_idx (Tensor [B])
        """
        # 0. 数据展平，方便处理
        if attention_matrix_stack.ndim == 4:
            # [L, B, H, N] -> [L, H, B, N] -> [L*H, B, N]
            L, B, H, N = attention_matrix_stack.shape
            candidates = attention_matrix_stack.permute(0, 2, 1, 3).reshape(L * H, B, N)
        else:
            candidates = attention_matrix_stack

        # Ensure candidates are on the correct device
        if candidates.device != self.device:
            candidates = candidates.to(self.device)

        # 1. 预测 (Prior)
        prior = self._predict_prior()
        
        # ---------------------------------------------------
        # 步骤 B: 观测预测与选择 (Prediction & Selection)
        # 1. 预测观测分布: \hat{o} = prior * Gaussian
        #    这步增加了对预测位置不确定性的容忍
        # ---------------------------------------------------
        predicted_obs = self._apply_gaussian(prior)
        best_obs, best_idx, best_head_or = self._select_best_head_loglik(predicted_obs, candidates)
        
        # ---------------------------------------------------
        # 步骤 C: 贝叶斯更新 (Update)
        # 1. 计算似然 (Likelihood): P(o|z)
        #    根据高斯观测模型，Likelihood 等价于对观测值进行高斯平滑
        #    \lambda_t = best_obs * Gaussian
        # ---------------------------------------------------
        likelihood = self._apply_gaussian(best_obs)
        # 将likelihood mask掉
        # 上一时刻的对齐位置
        last_alignment = torch.sum(self.indices.unsqueeze(0) * self.belief, dim=1) # (B,)
        
        # mask掉比last_alignment取整后还小的位置
        mask = self.indices.unsqueeze(0) < last_alignment.floor().unsqueeze(1)   # (B, N)
        likelihood = likelihood.masked_fill(mask, 0.0)
        
        # 2. 计算后验 (Posterior)
        #    Posterior = Prior * Likelihood
        posterior = prior * (likelihood + 1e-12) # 防止乘零
        
        # 归一化
        norm_posterior = self._normalize(posterior, dim=1)  # 防止除零
        
        self.belief = norm_posterior

        # 4. 计算非整数重心 (Expectation)
        # index: [0, 1, 2, ..., N-1]
        alignment_center = torch.sum(self.indices.unsqueeze(0) * norm_posterior, dim=1) # (B,)
        
        # # 在alignment_center >= last_alignment的地方，更新belief
        # update_mask = alignment_center >= last_alignment
        # self.belief[update_mask] = norm_posterior[update_mask]
        # # 同时只有alignment_center >= last_alignment的地方，更新alignment_center
        # alignment_center[update_mask == False] = last_alignment[update_mask == False]
        
        self.history.append(alignment_center.cpu().tolist())
        
        if attn_map_processor:
            attn_map_processor.prcess_hmm(
                alignment_center, 
                best_head_or, 
                best_obs,
                self.belief,
                output_path=model_kwargs.get('output_path', None),
                attn_phase=model_kwargs.get('attention_phase', None),
                text_last_token_position=model_kwargs.get('text_last_token_position', None),
            )
        
        return alignment_center, best_idx
    
    def reorder(self, beam_indices):
        """
        重排序
        """
        self.belief = self.belief.index_select(0, beam_indices)