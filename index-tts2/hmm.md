## 定义

### 符号定义 (Notation)

* **状态空间**：令 $\mathcal{Z} = \{0, 1, \dots, N-1\}$ 为 Text Token 的索引集合，$N$ 为 Text 长度。
* **时间步**：$t = 1, 2, \dots, T$ 代表 Sem Token（流式输入）。
* **隐状态分布 (Belief State)**：$\boldsymbol{\pi}_t \in \mathbb{R}^N$，其中 $\pi_t(j) = P(z_t = j | \text{history})$ 表示时刻 $t$ 对齐到第 $j$ 个 Text Token 的后验概率。满足 $\sum_{j} \pi_t(j) = 1$。
* **候选观测集合**：令 $\mathcal{A}_t = \{ \mathbf{a}_t^{(k)} \}_{k=1}^M$ 为 $t$ 时刻所有注意力头的集合，其中 $M = L \times H$，表示$L$层和$H$个注意力头，且每个 $\mathbf{a}_t^{(k)} \in \mathbb{R}^N$ 为一个非负的注意力分布向量。
* **转移概率**：$p \in [0, 1]$ 为向后移动一步的概率。
  
为了实现高斯观测模型，我们需要引入一个**高斯扩散核 (Gaussian Kernel)**。

* **高斯核向量**：令 $\mathbf{g} \in \mathbb{R}^{2W+1}$ 为一个离散化的截断高斯向量，窗口半径为 $W$（例如 $W=3\sigma$）。
    对于相对位置 $\delta \in \{-W, \dots, W\}$：
    $$
    g(\delta) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{\delta^2}{2\sigma^2}\right)
    $$
* **观测模型假设**：$P(o|z=j) \sim \mathcal{N}(j, \sigma^2)$。这意味着，如果当前真实对齐在 $j$，我们期望看到的 Attention 分布应该是一个以 $j$ 为中心的高斯包络。

---

### 算法流程 (Algorithm Formulation)

#### 1. 初始化 (Initialization)
在 $t=0$ 时刻，假设对齐从第一个 Token 开始：
$$
\boldsymbol{\pi}_0 = [1, 0, \dots, 0]^\top
$$

#### 2. 递归推断 (Recursive Inference)
对于每个时间步 $t = 1, \dots, T$，执行以下三个子步骤：

**步骤 A: 先验预测 (Time Update / Prediction)**
利用马尔可夫转移性质，计算先验分布 $\tilde{\boldsymbol{\pi}}_t$（对应代码中的 `prior`）：

$$
\tilde{\pi}_t(j) = (1-p) \cdot \pi_{t-1}(j) + p \cdot \pi_{t-1}(j-1)
$$

*注：对于边界条件，当 $j=0$ 时第二项为0；当 $j=N-1$ 时，通常处理为累积或截断。写成向量形式即：*
$$
\tilde{\boldsymbol{\pi}}_t = (1-p)\boldsymbol{\pi}_{t-1} + p (\mathbf{S} \cdot \boldsymbol{\pi}_{t-1})
$$
*其中 $\mathbf{S}$ 为下移位矩阵 (Lower Shift Matrix)。*

**步骤 B: 观测分布预测与选择 (Observation Prediction & Selection)**
我们不直接用状态先验 $\tilde{\boldsymbol{\pi}}_t$ 去找头，而是先计算**“预测观测分布” (Predicted Observation Distribution)**。

1.  **计算预测观测 $\hat{\mathbf{o}}_t$**：
    利用全概率公式 $P(o) = \sum_z P(o|z)P(z)$。在离散网格上，这等价于将状态先验与高斯核进行**卷积 (Convolution)**：
    $$
    \hat{\mathbf{o}}_t = \tilde{\boldsymbol{\pi}}_t * \mathbf{g}
    $$
    * $\hat{\mathbf{o}}_t$ 回答了：*“考虑到位置的不确定性 $\sigma$，理想的 Attention Map 长什么样？”*（它通常比 $\tilde{\boldsymbol{\pi}}_t$ 更平滑、更宽）。
    
    

2.  **选择最佳观测头**：
    在候选集合 $\mathcal{A}_t$ 中，寻找与预测分布 $\hat{\mathbf{o}}_t$ 最相似的头。通常使用 **KL散度 (KL Divergence)** 或 **交叉熵**：
    $$
    k^*_t = \underset{k}{\arg\min} \ D_{KL}(\hat{\mathbf{o}}_t \ \| \ \mathbf{a}_t^{(k)} + \epsilon)
    $$
    或者为了计算简便（假设 $\hat{\mathbf{o}}_t$ 是目标），最大化似然：
    $$
    k^*_t = \underset{k}{\arg\max} \sum_{j=0}^{N-1} \hat{\mathbf{o}}_t[j] \cdot \log(\mathbf{a}_t^{(k)}[j] + \epsilon)
    $$
    
    *令选中的头为 $\mathbf{a}_{sel} = \mathbf{a}_t^{(k^*_t)}$。*

**步骤 C: 贝叶斯更新 (Update with Gaussian Emission)**
利用选中的头更新状态。此时我们需要计算**似然向量 (Likelihood Vector)** $\boldsymbol{\lambda}_t$。

根据 $P(o|z=j) \sim \mathcal{N}(j, \sigma^2)$，如果观测到了分布 $\mathbf{a}_{sel}$，那么状态 $j$ 的似然度等于 $\mathbf{a}_{sel}$ 在 $j$ 附近的高斯加权和（即对观测值也做一次高斯平滑）：

$$
\lambda_t(j) = \sum_{\delta=-W}^{W} \mathbf{a}_{sel}[j+\delta] \cdot g(\delta)
$$

写成向量卷积形式：
$$
\boldsymbol{\lambda}_t = \mathbf{a}_{sel} * \mathbf{g}
$$

最后执行贝叶斯乘法更新：
$$
\boldsymbol{\pi}_t = \frac{\tilde{\boldsymbol{\pi}}_t \odot \boldsymbol{\lambda}_t}{\| \tilde{\boldsymbol{\pi}}_t \odot \boldsymbol{\lambda}_t \|_1}
$$



#### 3. 状态估计输出 (Estimation)
计算当前时刻的对齐中心（期望值）：

$$
\hat{y}_t = \mathbb{E}[z_t] = \sum_{j=0}^{N-1} j \cdot \pi_t(j)
$$

---

### 总结性描述

这是一个非常标准的离散隐马尔可夫模型 (Discrete Hidden Markov Model, HMM) 实现，具体来说，它是基于 网格的贝叶斯滤波器 (Grid-based Bayesian Filter)，用于流式（Streaming）的状态估计。

更准确地定义它，这是一种带有观测选择机制和高斯平滑的流式 HMM 滤波器 (Streaming HMM Filter with Observation Selection and Gaussian Smoothing)。

---

## 实现

### 潜在的优化方向 (For future)

1.  **Top-K 融合**：现在只选了 Top-1 的头。如果模型不稳定，可以选 Top-3 个头取平均，这样更稳健。
2.  **动态转移概率**：现在的 `p` 是固定的。如果某个 Sem Token 对应的音素很长，应该降低 `p`；如果是短元音，提高 `p`。
3.  **窗口限制**：为了加速，计算 dot product 时可以只算 belief 非零区域附近的 text token，不必全量计算。
4.  
这个修改让算法从“直接信任观测”变成了“带有不确定性建模的概率关联”，这是一个非常符合控制理论（Control Theory）和统计信号处理的改进。

这种修改的核心思想是：**无论是“预测”还是“观测”，都应当包含高斯噪声（即位置的不确定性）。**

以下是根据你的要求（高斯观测模型 + 预测观测分布进行选择）重写的完整算法定义。

---
### 关键改进点分析

1.  **双重平滑 (Double Smoothing)**：
    * 在 **Step B** 中，你用高斯核平滑了**先验** ($\tilde{\boldsymbol{\pi}} * \mathbf{g}$)，这意味着：即使先验认为只能在位置 5，我们也允许找峰值在 4 或 6 的 Attention 头。这增加了对“预测偏差”的容忍度。
    * 在 **Step C** 中，你用高斯核平滑了**观测** ($\mathbf{a}_{sel} * \mathbf{g}$)，这意味着：即使 Attention 头是一个非常尖锐的 One-hot 向量，它也会通过高斯扩散影响周围的状态概率。

2.  **更强的鲁棒性**：
    这种方法被称为 **Probabilistic Data Association Filter (PDAF)** 的变体。通过引入 $\sigma$（方差），你明确告诉模型：“我不完全信任我的运动模型（Step A），也不完全信任 Attention 头的像素级精度（Step C），我允许它们在 $\sigma$ 范围内波动。”

### 总结公式
$$
\begin{aligned}
\text{Prior:} & \quad \tilde{\boldsymbol{\pi}} = \text{Shift}(\boldsymbol{\pi}_{t-1}) \\
\text{Predict Obs:} & \quad \hat{\mathbf{o}} = \tilde{\boldsymbol{\pi}} * \mathbf{g}_\sigma \\
\text{Select:} & \quad \mathbf{a}_{sel} = \arg\max_k \langle \hat{\mathbf{o}}, \log \mathbf{a}^{(k)} \rangle \\
\text{Likelihood:} & \quad \boldsymbol{\lambda} = \mathbf{a}_{sel} * \mathbf{g}_\sigma \\
\text{Posterior:} & \quad \boldsymbol{\pi}_t \propto \tilde{\boldsymbol{\pi}} \odot \boldsymbol{\lambda}
\end{aligned}
$$


