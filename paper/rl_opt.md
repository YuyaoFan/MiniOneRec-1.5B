# GDPO 论文分析与RL阶段改进方案

> **论文**: GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization
> **发表**: ICML 2026 | **作者**: Shih-Yang Liu et al. (NVIDIA)
> **源码参考**: [reference/GDPO/](../reference/GDPO/)

---

## 1. GDPO 核心思想

### 1.1 问题动机：GRPO在多奖励场景下的奖励信号坍塌

GRPO（Group Relative Policy Optimization）当前主要用于单目标奖励优化（如准确率）。在多奖励RL中，现有方法采用的策略是**先求和再组内归一化**：

$$r^{(i,j)}_{sum} = r^{(i,j)}_1 + \cdots + r^{(i,j)}_n$$
$$A^{(i,j)}_{sum} = \frac{r^{(i,j)}_{sum} - \text{mean}\{r^{(i,1)}_{sum}, \ldots, r^{(i,G)}_{sum}\}}{\text{std}\{r^{(i,1)}_{sum}, \ldots, r^{(i,G)}_{sum}\}}$$

**GRPO的根本问题**：这种"先求和再归一化"的策略导致了推荐信号的信息坍塌。

#### 具体示例（二值双奖励, G=2 rollout）

考虑两个二值奖励 $r_1, r_2 \in \{0, 1\}$，总分 $\in \{0, 1, 2\}$，每个prompt采样2个response。

| Rollout组合 | (总分1, 总分2) | GRPO归一化优势 | 分组 |
|------------|---------------|---------------|------|
| (0,1) | (0,1) | (-0.7071, 0.7071) | A |
| (0,2) | (0,2) | (-0.7071, 0.7071) | A |
| (1,2) | (1,2) | (-0.7071, 0.7071) | A |
| (0,0) | (0,0) | (0, 0) | B |
| (1,1) | (1,1) | (0, 0) | B |
| (2,2) | (2,2) | (0, 0) | B |

**关键问题**: (0,2)和(0,1)在语义上有本质区别：
- (0,2)表示一个rollout同时满足两个奖励（更强的正信号）
- (0,1)表示只满足一个奖励（较弱正信号）

但GRPO将它们映射到**完全相同的优势值** $(-0.7071, 0.7071)$，丢失了关键的区分信息。

### 1.2 GDPO的解决方案

GDPO的核心思想是**解耦每个奖励的组内归一化**，然后再聚合：

```
GRPO: 先求和 → 再组归一化 → 优势值 (2种)
GDPO: 每个奖励独立组归一化 → 加权聚合 → 批归一化 → 优势值 (3种)
```

**GDPO的优势计算流程**:
1. 对每个奖励函数 $i$，独立进行组内归一化：
   $$A_i = \frac{r_i - \text{mean}_g(r_i)}{\text{std}_g(r_i) + \epsilon}$$
2. 加权聚合各奖励的优势：
   $$A_{pre} = \sum_i w_i \cdot A_i$$
3. 批级归一化（可选，稳定数值范围）：
   $$A_{final} = \frac{A_{pre} - \text{mean}_{batch}(A_{pre})}{\text{std}_{batch}(A_{pre}) + \epsilon}$$

---

## 2. 方法详解

### 2.1 GRPO reward collapse的数学分析

GRPO的组合奖励归一化在数学上等价于对奖励空间进行了一个**低分辨率投影**。对于n个奖励函数，每个有$r_i$种可能取值，组合空间有$\prod_i r_i$种可能状态。但GRPO的group-wise normalization只保留了**排序信息**（相对顺序），丢失了**幅度信息**（差异大小）。

当G固定，n增加时：
- GRPO的distinct advantage groups增长缓慢（饱和效应）
- GDPO的解耦归一化保持了对每个奖励维度的完整分辨率

### 2.2 GDPO在MiniOneRec场景的适配

MiniOneRec的RL阶段使用两种奖励：
1. **Rule Reward** ($r_{rule} \in \{0, 1\}$): 二值正确性
2. **NDCG Ranking Reward** ($r_{rank} \in [-1, 0]$): 排序位置惩罚

这两种奖励的**数值范围和语义**完全不同：
- Rule reward: 二元、稀疏（大多数负样本得0）
- NDCG reward: 连续、密集（每个负样本有不同的惩罚值）

在这种情况下，GRPO的"先求和再归一化"会：
- NDCG的排序信息被rule reward的0/1信号稀释
- 不同排序位置的负样本难以区分（都被压缩到类似优势值）

GDPO的解耦归一化保留了：
- Rule reward的二值区分度
- NDCG reward的细粒度排序梯度

### 2.3 批归一化的作用

GDPO在组内解耦归一化后增加了批归一化步骤。这一步的作用：
- **稳定数值范围**: 不管有多少个reward，advantage的分布保持稳定
- **跨batch的可比性**: 不同batch样本的优势值在同一尺度上
- **训练稳定性**: 避免多奖励优势值累加导致的梯度爆炸

---

## 3. 代码实现差异分析

### 3.1 GRPO原始实现（当前MiniOneRec代码中的逻辑）

```python
# minionerec_trainer.py: line ~960-980
# 步骤:
# 1. Gather rewards
rewards_per_func = gather(rewards_per_func)  # shape: [B*G, n_rewards]

# 2. 加权求和
rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

# 3. 组内归一化（先求和再归一化）
mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
```

### 3.2 GDPO实现（需要修改为）

```python
# 参考: reference/GDPO/trl-GDPO 中的实现
# 步骤:
# 1. Gather rewards
rewards_per_func = gather(rewards_per_func)  # shape: [B*G, n_rewards]

# 2. 每个奖励独立组内归一化
all_reward_advantage = []
for i in range(len(self.reward_weights)):
    reward_i = rewards_per_func[:, i]
    each_reward_mean = reward_i.view(-1, self.num_generations).mean(dim=1)
    each_reward_std = reward_i.view(-1, self.num_generations).std(dim=1)
    each_reward_mean = each_reward_mean.repeat_interleave(self.num_generations, dim=0)
    each_reward_std = each_reward_std.repeat_interleave(self.num_generations, dim=0)
    each_advantage = (reward_i - each_reward_mean) / (each_reward_std + 1e-4)
    all_reward_advantage.append(each_advantage)

# 3. 加权聚合
combined_advantage = torch.stack(all_reward_advantage, dim=1)
pre_bn_advantages = (combined_advantage * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

# 4. 批归一化
bn_mean = pre_bn_advantages.mean()
bn_std = pre_bn_advantages.std()
advantages = (pre_bn_advantages - bn_mean) / (bn_std + 1e-4)
```

### 3.3 修改的核心差异

| 操作 | GRPO (当前) | GDPO (改进) |
|------|-----------|-----------|
| 归一化时机 | 求和后归一化 | 各奖励独立归一化 |
| 奖励区分度 | 低（坍塌效应） | 高（保留各维度信息） |
| 批归一化 | 无 | 有（稳定训练） |
| 代码改动量 | - | 约20行 |

---

## 4. 优势分析

### 4.1 训练稳定性

GDPO在论文中展示了更强的训练稳定性：
- **Tool Calling任务**: GDPO同时达到更高的correctness和format reward
- **GRPO训练崩溃**: Fig. 5显示GRPO在约400步后correctness reward开始下降（reward hacking）
- **5次独立运行**: GDPO的IQR带更窄（更稳定），中位数更高

### 4.2 下游性能

| 任务 | 模型 | GRPO | GDPO | 提升 |
|------|------|------|------|------|
| Math (AIME) | DeepSeek-R1-1.5B | baseline | +6.3% | 显著 |
| Math (AIME) | Qwen3-4B-Instruct | baseline | +2.3% | 中等 |
| Tool Calling | Qwen2.5-1.5B | baseline | 更高 | 显著 |
| Code | - | baseline | 更好 | 中等 |

### 4.3 实现简洁性

- **即插即用**: 仅需修改advantage计算部分，不需要改变reward函数、模型结构或其他训练逻辑
- **无额外超参数**: 不需要新的学习率、损失权重等
- **向后兼容**: 单奖励时GDPO退化为GRPO

---

## 5. 劣势与局限性

### 5.1 多奖励前提

- GDPO的收益在**多奖励场景**（≥2个reward函数）下才显著
- 单奖励时GDPO与GRPO等价
- MiniOneRec默认识别两种情况：`reward_type == "rule"`（单reward）vs `reward_type == "ranking"`（双reward）

### 5.2 额外计算开销

- 每个奖励独立计算group-wise statistics（微小的额外开销）
- Batch-wise normalization增加一次全局统计（可忽略）
- 总体计算开销相比GRPO增加 < 1%

### 5.3 奖励独立性假设

- GDPO假设各奖励之间可以独立归一化
- 如果奖励之间存在强相关性或对抗关系，解耦归一化可能不是最优策略
- 在MiniOneRec中，rule reward和NDCG reward是互补的（正确性+排序质量），独立性假设合理

### 5.4 权重敏感性

- 奖励权重 $w_i$ 的设置影响最终优势值分布
- 论文建议通过reward shaping和权重调整来对齐不同优先级
- 需要一定的调参经验

---

## 6. 实现流程

### 6.1 总体修改方案

MiniOneRec中RL训练的advantage计算在 `minionerec_trainer.py` 的 `_prepare_inputs()` 方法中（约960-980行）。需要修改的只是advantage计算逻辑。

### 6.2 需要修改的文件

1. **minionerec_trainer.py**: 核心advantage计算逻辑修改
2. **rl.py**: 添加GDPO选项（如 `--gdpo` flag）

### 6.3 具体实施步骤

**Step 1**: 在 `rl.py` 中添加GDPO开关
```python
# rl.py train() 函数中添加参数
gdpo: bool = False,  # 新增GDPO flag
```

**Step 2**: 传递GDPO flag到ReReTrainer
```python
# rl.py
trainer = ReReTrainer(
    ...
    gdpo=gdpo,  # 新增
    ...
)
```

**Step 3**: 在ReReTrainer中存储flag
```python
# minionerec_trainer.py __init__
self.gdpo = gdpo
```

**Step 4**: 修改 `_prepare_inputs()` 中的advantage计算
```python
# minionerec_trainer.py _prepare_inputs()
# 原代码: rewards → advantages (GRPO)
# 新代码: if self.gdpo and n_rewards > 1 → GDPO else → GRPO

if self.gdpo and len(self.reward_funcs) > 1:
    # GDPO: 解耦归一化 + 批归一化
    all_reward_advantage = []
    for i in range(len(self.reward_weights)):
        reward_i = rewards_per_func[:, i]
        mean_g = reward_i.view(-1, self.num_generations).mean(dim=1)
        std_g = reward_i.view(-1, self.num_generations).std(dim=1)
        mean_g = mean_g.repeat_interleave(self.num_generations, dim=0)
        std_g = std_g.repeat_interleave(self.num_generations, dim=0)
        adv_i = (reward_i - mean_g) / (std_g + 1e-4)
        all_reward_advantage.append(adv_i)
    
    combined = torch.stack(all_reward_advantage, dim=1)
    pre_bn = (combined * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)
    advantages = (pre_bn - pre_bn.mean()) / (pre_bn.std() + 1e-4)
else:
    # 原始GRPO逻辑（保持不变）
    rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)
    mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
    std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
```

### 6.4 与现有DAPO/GSPO的兼容性

MiniOneRec已有DAPO/GSPO支持（在`compute_loss`中）：
```python
if self.dapo:    # per-token average loss
    loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
elif self.gspo:  # sequence-level importance ratio  
    ...
else:            # standard GRPO loss
    ...
```

GDPO的优势值可以直接替代GRPO的优势值输入到这些loss变体中，完全兼容。

---

## 7. MiniOneRec场景的预期收益

### 7.1 为什么GDPO适合推荐场景

1. **推荐奖励本质多模态**: 正确性（二值）+ 排序质量（连续）具有不同的统计特性
2. **奖励分布非对称**: 正样本极少（1/|I|），负样本占绝大多数
3. **排序感知需求**: 需要区分"好的错误"（top-2推荐）和"差的错误"（top-50推荐）

### 7.2 预期改进

- **HR@K提升**: GDPO保留的细粒度排序信号应转化为更好的top-K准确率
- **NDCG@K提升**: 排序感知优势被更好地保留，NDCG应有更显著改善
- **训练稳定性**: 避免GRPO可能出现的奖励信号衰减
- **多样性**: GDPO保留的多奖励信号可能促进更丰富的采样策略

---

## 8. 参考文献

- Liu et al., "GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimization," ICML 2026. arXiv:2601.05242
- Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models," arXiv:2402.03300, 2024.
- Yu et al., "DAPO: An Open-Source LLM Reinforcement Learning System at Scale," arXiv:2503.14476, 2025.
- Schulman et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.
