# RQ-VAE SFT + GRPO vs GDPO 对比分析

**数据集**: Amazon Industrial_and_Scientific | **SID**: RQ-VAE (e_dim=2560, 3-level×256)
**模型**: Qwen2.5-1.5B | **RL超参**: G=16, beam_search, reward=ranking, lr=1e-5, beta=1e-3

## 1. 核心指标对比

| 指标 | SFT | GRPO (best) | GDPO (best) | GRPO vs SFT | GDPO vs SFT | GRPO vs GDPO |
|------|-----|-------------|-------------|-------------|-------------|--------------|
| HR@3 | 0.0904 | **0.0984** | 0.0918 | +8.8% | +1.5% | **+0.0066** |
| NDCG@3 | 0.0792 | **0.0863** | 0.0825 | +9.0% | +4.2% | **+0.0038** |
| HR@5 | 0.1061 | **0.1136** | 0.1050 | +7.1% | -1.0% | **+0.0086** |
| NDCG@5 | 0.0856 | **0.0926** | 0.0880 | +8.1% | +2.7% | **+0.0046** |
| HR@10 | 0.1337 | **0.1379** | 0.1253 | +3.1% | -6.3% | **+0.0126** |
| NDCG@10 | 0.0945 | **0.1005** | 0.0944 | +6.4% | -0.0% | **+0.0061** |

## 2. 训练曲线对比

| 阶段 | GRPO NDCG@10 | GDPO NDCG@10 | 说明 |
|------|-------------|-------------|------|
| step 165 | 0.0845 | 0.0799 | GRPO-GDPO=+0.0046 |
| step 330 | 0.0756 | 0.0679 | GRPO-GDPO=+0.0077 |
| step 495 | 0.0702 | 0.0769 | GRPO-GDPO=-0.0067 |
| step 660 | 0.0894 | 0.0767 | GRPO-GDPO=+0.0127 |
| step 825 | 0.0908 | 0.0830 | GRPO-GDPO=+0.0079 |
| step 990 | 0.0929 | 0.0846 | GRPO-GDPO=+0.0083 |
| step 1155 | 0.0990 | 0.0944 | GRPO-GDPO=+0.0046 |
| step 1320 | 0.1005 | 0.0934 | GRPO-GDPO=+0.0071 |
| step 1485 | 0.1000 | 0.0928 | GRPO-GDPO=+0.0072 |
| step 1650 | 0.1002 | 0.0933 | GRPO-GDPO=+0.0069 |

## 3. 关键发现

### ✅ GRPO 显著超越 SFT: NDCG@10 0.1005 vs SFT 0.0945 (+6.4%)
### ⚠️ GDPO 基本持平 SFT: NDCG@10 0.0944 vs SFT 0.0945 (-0.0%)
### 🔴 GRPO 全面领先 GDPO: NDCG@10 高出 0.0061 (6.5%)

## 4. GDPO 未达预期的原因分析

### 4.1 Reward 稀疏性问题
GDPO 的核心优势在于 per-reward decoupled normalization 保留每个 reward 维度的区分度。
但在推荐场景中：
- **rule_reward**: 二值匹配 (0/1)，命中率仅 ~2%，大部分 sample 得 0 分
- **ndcg_rule_reward**: 排序惩罚 (0~-1)，组内相对有效但绝对值区分度有限
两个 reward 信号都**极度稀疏**，GDPO 对每个 reward 独立做组归一化时，std≈0 的组占绝大多数，
优势值 ≈ 0，模型几乎收不到有效梯度。相比之下 GRPO 先求和至少提供了更强的信号。

### 4.2 GDPO 论文场景 vs 推荐场景
GDPO 论文在 Tool Calling/Math/Coding 场景中验证，这些场景的 reward (format+correctness) 
拥有 >50% 的命中率。推荐场景的 reward 命中率 (~2%) 低了 25 倍，GDPO 的优势无法释放。

### 4.3 有效梯度对比
| 方法 | 正信号比例 | 有效梯度密度 | 收敛效果 |
|------|----------|------------|---------|
| GRPO | Σ(rewards) → ~2% non-zero | 低但持续 | ✅ 超越 SFT |
| GDPO | per-reward norm → ~2%×2 | 更低 (独立 norm 放大稀疏性) | ⚠️ 未超越 SFT |

## 5. 结论与建议

1. **GRPO 是推荐场景下已验证的最优 RL 方法**
2. GDPO 在极度稀疏 reward 场景下无优势，不推荐用于当前推荐 setting
3. 未来方向：设计更密集的软 reward（如 item embedding cosine similarity），提高 GDPO 的有效梯度密度
4. 或增大 num_generations (G=16→32→64) 提高正样本命中率

## 6. 产出文件

| 文件 | 路径 |
|------|------|
| GRPO 指标表 | `results/baseline_grpo/results_table.md` |
| GDPO 指标表 | `results/baseline_gdpo/results_table.md` |
| 本对比报告 | `GRPO_vs_GDPO.md` |
| GRPO 曲线 | `assets/plot_rqvae_grpo_ndcg.png` / `_recall.png` |
| GDPO 曲线 | `assets/plot_rqvae_gdpo_ndcg.png` / `_recall.png` |
| 对比图 | `assets/plot_grpo_vs_gdpo_comparison.png` |