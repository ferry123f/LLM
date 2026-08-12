# Hyper-Connections（超连接）：残差连接的可学习替代

> 一句话：Hyper-Connections（HC）是 ByteDance Seed 团队提出的**残差连接替代方案**——把输入扩展成 *n* 份，用可学习的**深度连接 + 宽度连接**让网络自主决定层间连接强度，一举化解 Pre-Norm / Post-Norm 在「梯度消失 ↔ 表示崩塌」之间的跷跷板困境。DeepSeek-V4 的 **mHC** 正是它的流形约束改进版。

## 核心论点

残差连接（He et al., 2016）是现代深度网络的基石，但它有个未解的老问题——两大变体各自偏科：

- **Pre-Norm**：在残差块前归一化，解决梯度消失，但深层**表示崩塌**（相邻层特征高度相似，多加的层几乎没贡献）。
- **Post-Norm**：在残差块后归一化，缓解表示崩塌，但**重新引入梯度消失**。

二者像跷跷板（seesaw）的两端，各做一次取舍。**根本症结**：残差连接**预先固定**了层内输入与输出的连接强度。HC 的问题意识由此而来——*能不能让网络自己学习最优的连接强度？*

## 核心机制：深度连接 + 宽度连接

**图解——HC 与残差连接的对比（论文 Figure 2，扩展率 n=2）：** (a) 标准残差；(b) 完整 HC（绿色 β 深度权重 + 蓝色 α 宽度权重）；(c) 深度连接（层输出与 h₁ 加权求和）；(d) 宽度连接（h₁、h₂ 横向交换信息）：

![[hc-connections.png]]

HC 把网络输入**复制 *n* 份**（扩展率 expansion rate = *n*），形成 hyper hidden matrix：

$$H_0 = (h^0\ h^0\ \cdots\ h^0)^\top \in \mathbb{R}^{n\times d}$$

层间连接用一个 $(n{+}1)\times(n{+}1)$ 矩阵表示，元素是可学习的连接权重：

$$\mathrm{HC} = \begin{pmatrix} 0 & \mathcal{B} \\ \mathcal{A}_m & \mathcal{A}_r \end{pmatrix} \in \mathbb{R}^{(n+1)\times(n+1)}$$

对一个层 $\mathcal{T}$（attention 或 FFN），HC 的输出为：

$$\hat{H} = \mathcal{B}^\top\, \mathcal{T}(H^\top \mathcal{A}_m)^\top + \mathcal{A}_r^\top H$$

它可拆成两类连接（对应图 c / d）：

- **深度连接（Depth-connections）**：广义残差——用 $\mathcal{A}_m$ 加权求和得到当前层输入 $h_0^\top=\mathcal{A}_m^\top H$，再用 $\mathcal{B}$ 给层输出加权。相当于"给每层的输入/输出连接都赋一个可学习权重"。
- **宽度连接（Width-connections）**：用 $\mathcal{A}_r$ 让 *n* 个 hidden vector 之间**横向交换信息**（同层内的信息流动）。

论文论证 **n>1 是必要的**：n=1 时跷跷板依旧存在、实验无提升；只有 n>1 时 HC 才不仅能调残差强度，还能**重排层**（见下）。

## 静态 vs 动态（SHC / DHC）

- **SHC（Static）**：矩阵元素是可训练标量，训练完固定。
- **DHC（Dynamic）**：连接权重**按输入 token 动态生成**，更灵活。动态部分经 norm → 线性变换 → tanh → 小初始可学习缩放因子 $s$ 得到，叠加在静态矩阵上：

$$\mathcal{B}(H)=s_\beta\circ\tanh(HW_\beta)^\top+\mathcal{B},\qquad \mathcal{A}_m(H)=s_\alpha\circ\tanh(HW_m)+\mathcal{A}_m$$

**初始化**：动态权重 $W_\beta,W_m,W_r$ 全部初始化为 0、静态矩阵按特定 pattern 初始化，使 HC **初始时严格等价于 Pre-Norm**——保证训练起点不劣化，再逐步学出更优连接。

## 为什么有效

1. **残差是 HC 的特例**：Pre-Norm / Post-Norm 都能写成 n=1 的**不可训练** HC 矩阵，例如
$$\mathrm{HC}_{\text{PreNorm}}=\begin{pmatrix}0&1\\1&1\end{pmatrix}$$
HC 只是把它推广到 $(n{+}1)\times(n{+}1)$、且权重可训练甚至输入相关。

2. **序列-并行对偶（Sequential-Parallel Duality）**：不同的 HC 矩阵对应把层排成**串行、并行、或两者的软混合**。静态 HC 训练后排布固定；**DHC 甚至能为每个 token 动态选择排布**。这是 HC 超越固定架构的关键。

3. **缓解表示崩塌**：可视化显示 Pre-Norm 相邻层特征余弦相似度极高（崩塌），HC 显著拉低相似度、放大每层的独立贡献。

## 实验结果

- 设置：OLMo（dense）与 OLMoE（MoE）预训练，500B tokens，Pre-Norm 为基线；默认配置 **DHC×4**（n=4 + tanh）。
- **OLMoE-1B-7B-DHC×4**：收敛**快约 1.8 倍**，ARC-Challenge **+6 分**（相对基线，500B tokens）。
- 扩展率消融：n=1 不如基线；**n=4 最优**；n=8 收益微小。
- DHC（n≥2）训练损失下降更陡，且**更稳、全程无 loss spike**。
- 消融显示**宽度连接 WC 的可训练性最关键**（去掉后掉点明显）。
- **参数与算力开销可忽略**（虽然名义上把宽度扩大了 n 倍）。
- 优于同类方法 Altup、ResiDual；视觉生成 / 分类任务也有类似提升。

## 与 DeepSeek-V4 的 mHC 的关系

DeepSeek-V4 的 **mHC（Manifold-Constrained Hyper-Connections）** 直接建立在本文的 HC 之上，并针对"HC 深层堆叠时数值不稳定"做了关键改进：把残差映射矩阵**约束到双随机矩阵流形（Birkhoff 多面体）**，保证谱范数 ≤ 1（映射非扩张），用 **Sinkhorn-Knopp** 迭代投影到该流形。一句话对照：

> **HC（本文）= 让层间连接强度可学习；mHC（V4）= 在此基础上给连接矩阵加流形约束，换取深层堆叠的数值稳定。**

详见 [DeepSeek-V4 架构与百万上下文效率](DeepSeek-V4%20架构与百万上下文效率.md) 的 mHC 章节。

## See Also

- [DeepSeek-V4 架构与百万上下文效率](DeepSeek-V4%20架构与百万上下文效率.md) — V4 用 mHC（HC 的流形约束版）替代残差连接；本文是其理论来源。

## Sources

- [Hyper-Connections](../../raw/deepseek/HC.pdf)（Zhu et al., ByteDance Seed，ICLR 2025，arXiv:2409.19606v3）
