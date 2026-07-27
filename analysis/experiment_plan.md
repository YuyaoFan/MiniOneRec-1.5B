# MiniOneRec 改进方法实验流程

> **基线**: 1.5B RQ-VAE + SFT + RL(GRPO) — 已有开源结果
> **改进**: 1.5B RQ-MoE + SFT + RL(GDPO) — 本实验方案
> **数据集**: Amazon Review - Industrial_and_Scientific

---

## 实验设计原则

1. **控制变量**: 除SID方法和RL方法外，所有参数与baseline保持一致
2. **公平对比**: RQ-MoE vs RQ-VAE（相同码本大小K=256, M=3层）
3. **GDPO vs GRPO**（相同reward function、训练步数、RL超参）
4. **单变量改动**: 每次只改变一个模块，便于归因分析

---

## 实验流程总览

```
Stage 1: SID构建
  Baseline: RQ-VAE → 训练 → best_collision_model.pth → generate_indices → index.json
  Improved:  RQ-MoE → 训练 → best_collision_model.pth → generate_indices_rqmoe → index.json

Stage 2: SFT
  使用对应的index.json + item.json → convert_dataset → SFT训练 → SFT checkpoint

Stage 3: RL
  Baseline: 加载 SFT ckpt → RL(GRPO) → eval → 指标
  Improved:  加载 SFT ckpt → RL(GDPO) → eval → 指标

对比: 最终推荐指标 (HR@K, NDCG@K) + 中间指标 (碰撞率, 重构误差, SFT loss)
```

---

## Stage 1: SID码本构建（改进：RQ-VAE → RQ-MoE）

### 1.1 基线实验（已完成）

```bash
# RQ-VAE (已有结果)
cd rq
bash rqkmeans_constrained.sh      # Constrained K-means 初始化
bash rqkmeans_plus.sh             # RQ-KMeans++ 训练
bash generate_indices_plus.sh     # SID 生成
# 产出: data/Amazon/index/Industrial_and_Scientific.index.json
```

### 1.2 改进实验（RQ-MoE）

```bash
# Step 1: RQ-MoE 训练
cd rq
bash train_rqmoe.sh
# 超参保持一致:
#   --num_emb_list 256 256 256  (与RQ-VAE相同)
#   --e_dim 64                   (与RQ-VAE相同)
#   --layers 2048 1024 512 256 128 64  (与RQ-VAE相同)
#   --lr 1e-3                    (与RQ-VAE相同)
#   --batch_size 20480           (与RQ-VAE相同)
#   --epochs 10000               (与RQ-VAE相同)
#   --warmup_epochs 5
#   --eval_step 10
#   新增:
#   --moe_N 2    (每步2个专家)
#   --moe_L 4    (每个专家4层)
#   --moe_H 256  (专家隐藏维度)
#   --moe_dropout 0.1
# 产出: checkpoints/rqmoe/best_collision_model.pth

# Step 2: SID 生成
bash generate_indices_rqmoe.sh
# 产出: data/Amazon/index/Industrial_and_Scientific.index.json (覆盖或另存)
```

### 1.3 Stage 1 对比指标

| 指标 | RQ-VAE (baseline) | RQ-MoE (改进) | 说明 |
|------|-------------------|---------------|------|
| Train Loss (收敛) | TBD | TBD | 收敛时训练损失 |
| Collision Rate | TBD | TBD | 码本碰撞率, 越低越好 |
| Reconstruction MSE | TBD | TBD | 重构误差 |
| Unique SID Count | TBD | TBD | 唯一SID数量 |
| 训练时间 | TBD | TBD | 单epoch训练时间 |

---

## Stage 2: 监督微调（SFT）

### 2.1 基线实验（已完成）

```bash
# 使用 RQ-VAE 生成的 index.json
bash convert_dataset.sh           # 数据转换
bash sft.sh                       # SFT 训练
# 产出: output/sft/Industrial_and_Scientific_plus/
```

### 2.2 改进实验（使用 RQ-MoE 生成的 index.json）

```bash
# 使用 RQ-MoE 生成的 index.json 进行数据转换
# 如果 index.json 路径不同, 需更新 convert_dataset.sh 中的路径
bash convert_dataset.sh           # 数据转换

# SFT 训练（超参与baseline完全一致）
bash sft.sh
# 关键超参 (严格保持一致):
#   base_model: Qwen2.5-1.5B-Instruct (相同)
#   batch_size: 128, micro_batch_size: 4 (相同)
#   num_epochs: 10 (相同, early stopping patience=3)
#   learning_rate: 3e-4 (相同)
#   cutoff_len: 512 (相同)
# 产出: output/sft/Industrial_and_Scientific_rqmoe/
```

### 2.3 Stage 2 对比指标

| 指标 | RQ-VAE SFT | RQ-MoE SFT | 说明 |
|------|-----------|-----------|------|
| Final Eval Loss | 待提取 | TBD | SFT评估损失 |
| Best Epoch | 待提取 | TBD | 早停时的最佳epoch |
| 训练时间 | TBD | TBD | 总训练时间 |

---

## Stage 3: 强化学习优化（改进：GRPO → GDPO）

### 3.1 基线实验（使用 RQ-VAE SID + GRPO）

```bash
# Baseline: RQ-VAE + SFT + RL(GRPO)
bash rl.sh
# 关键超参:
#   model_path: output/sft/Industrial_and_Scientific_plus/
#   reward_type: ranking  (rule + ndcg)
#   num_generations: 16
#   learning_rate: 1e-6
#   beta: 0.04
#   per_device_train_batch_size: 32
#   gradient_accumulation_steps: 32
#   temperature: 1.0
#   beam_search: True
# 产出: output/rl/Industrial_and_Scientific_plus/
```

### 3.2 改进实验（使用 RQ-MoE SID + GDPO）

```bash
# Improved: RQ-MoE + SFT + RL(GDPO)
# 修改 rl.sh: 添加 --gdpo True 参数, 更新 model_path
python rl.py \
    --model_path "output/sft/Industrial_and_Scientific_rqmoe/final_checkpoint" \
    --train_file "data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
    --eval_file "data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
    --info_file "data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt" \
    --category "Industrial_and_Scientific" \
    --output_dir "output/rl/Industrial_and_Scientific_rqmoe_gdpo" \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --gradient_accumulation_steps 32 \
    --num_generations 16 \
    --num_train_epochs 1 \
    --learning_rate 1e-6 \
    --beta 0.04 \
    --temperature 1.0 \
    --beam_search True \
    --test_during_training True \
    --test_beam 20 \
    --reward_type "ranking" \
    --eval_step 0.1 \
    --gdpo True \
    --sid_index_path "data/Amazon/index/Industrial_and_Scientific.index.json" \
    --item_meta_path "data/Amazon/index/Industrial_and_Scientific.item.json" \
    --wandb_project "" \
    --wandb_run_name "rqmoe_gdpo"
# 产出: output/rl/Industrial_and_Scientific_rqmoe_gdpo/
```

### 3.3 Stage 3 对比指标

| 指标 | RQ-VAE+GRPO | RQ-MoE+GDPO | 说明 |
|------|------------|-------------|------|
| HR@1 | TBD | TBD | |
| HR@3 | 0.0938 | TBD | |
| HR@5 | 0.1112 | TBD | |
| HR@10 | 0.1401 | TBD | |
| NDCG@3 | 0.0817 | TBD | |
| NDCG@5 | 0.0889 | TBD | |
| NDCG@10 | 0.0981 | TBD | |
| Categorical Diversity | TBD | TBD | 生成多样性 |
| Token Diversity | TBD | TBD | Token多样性 |
| Avg Completion Length | TBD | TBD | 平均生成长度 |
| KL Divergence | TBD | TBD | KL散度 |

### 3.4 补充消融实验（可选）

为进一步归因改进来源，可增加以下消融实验：

| 实验 | SID方法 | RL方法 | 目的 |
|------|---------|--------|------|
| Exp-A (基线) | RQ-VAE | GRPO | 已有结果 |
| Exp-B | RQ-MoE | GRPO | 单独评估RQ-MoE贡献 |
| Exp-C | RQ-VAE | GDPO | 单独评估GDPO贡献 |
| Exp-D (主实验) | RQ-MoE | GDPO | 评估组合改进效果 |

---

## 评测

### 单模型评测

```bash
# 对每个RL checkpoint分别评测
python evaluate.py \
    --base_model "output/rl/Industrial_and_Scientific_rqmoe_gdpo/eval_snapshots/checkpoint-N" \
    --info_file "data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt" \
    --category "Industrial_and_Scientific" \
    --test_data_path "data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv" \
    --result_json_data "results/eval/checkpoint-N.json" \
    --batch_size 4 \
    --num_beams 50
```

### 批量评测

参考原有 `tools/eval_all.py` 进行批量评测和可视化。

---

## 参数对齐检查清单

### Stage 1 参数对齐

| 参数 | RQ-VAE | RQ-MoE | 一致? |
|------|--------|--------|-------|
| 输入维度 | 768 | 768 | ✅ |
| 潜空间维度 e_dim | 64 | 64 | ✅ |
| 码本层数 M | 3 | 3 | ✅ |
| 码本大小 K | 256 | 256 | ✅ |
| Encoder结构 | [2048,1024,512,256,128,64] | [2048,1024,512,256,128,64] | ✅ |
| Decoder结构 | [64,128,256,512,1024,2048,768] | [64,128,256,512,1024,2048,768] | ✅ |
| 学习率 | 1e-3 | 1e-3 | ✅ |
| Batch size | 20480 | 20480 | ✅ |
| Epochs | 10000 | 10000 | ✅ |
| 优化器 | Adam | Adam | ✅ |
| 梯度裁剪 | 1.0 | 1.0 | ✅ |
| SID去重策略 | Polars dedup | Polars dedup | ✅ |
| Token格式 | `<a_X><b_Y><c_Z>` | `<a_X><b_Y><c_Z>` | ✅ |

### Stage 2 参数对齐

| 参数 | Baseline | Improved | 一致? |
|------|----------|----------|-------|
| 基础模型 | Qwen2.5-1.5B | Qwen2.5-1.5B | ✅ |
| SFT epoch | 10 (early stop=3) | 10 (early stop=3) | ✅ |
| 学习率 | 3e-4 | 3e-4 | ✅ |
| Batch size | 128 | 128 | ✅ |
| Micro batch | 4 | 4 | ✅ |
| Max seq len | 512 | 512 | ✅ |
| 训练任务 | SidSFT + ItemFeat + FusionSeqRec | 同左 | ✅ |
| 优化器 | AdamW | AdamW | ✅ |
| 学习率调度 | cosine with warmup | cosine with warmup | ✅ |

### Stage 3 参数对齐

| 参数 | GRPO (baseline) | GDPO (improved) | 一致? |
|------|----------------|-----------------|-------|
| SFT checkpoint | RQ-VAE SFT | RQ-MoE SFT | ✅ (各自对应的SFT) |
| Generations G | 16 | 16 | ✅ |
| 学习率 | 1e-6 | 1e-6 | ✅ |
| KL β | 0.04 | 0.04 | ✅ |
| Temperature | 1.0 | 1.0 | ✅ |
| Beam search | True | True | ✅ |
| Reward type | ranking | ranking | ✅ |
| Train batch size | 32 | 32 | ✅ |
| Grad accum | 32 | 32 | ✅ |
| Max grad norm | 0.3 | 0.3 | ✅ |
| LR scheduler | cosine | cosine | ✅ |
| Warmup ratio | 0.03 | 0.03 | ✅ |
| 评测策略 | Beam50, constraint | Beam50, constraint | ✅ |

---

## 预期结果

基于两篇论文的实验结果：

### RQ-MoE 预期改进
- 重构MSE降低: 10-30% (参考: BigANN上RQ-MoE vs RQ的MSE差距)
- 碰撞率降低: 更好的重构保持 → 更少的SID碰撞
- SFT收敛改善: 更高质量的SID → 更稳定的语义空间
- 传导到最终指标: HR@K提升1-3个百分点

### GDPO 预期改进
- 训练更稳定: reward曲线波动更小
- 排序质量提升: NDCG@K提升应比HR@K更明显（GDPO保持排序信号分辨率）
- 多样性: GDPO保留的多奖励结构可能增加采样多样性
- 传导到最终指标: NDCG@K提升2-5个百分点

### 组合预期
- RQ-MoE + GDPO 的改进应当具有**叠加效应**
- RQ-MoE改善表示空间 → 更准确的SID映射
- GDPO改善RL信号 → 更精确的排序优化
- 两者从不同角度提升，理论上互不冲突

---

## 运行时间估算

基于单卡48G环境：

| 阶段 | 预估时间 | 备注 |
|------|---------|------|
| Stage 1 (RQ-MoE训练) | ~12-24h | 取决于epoch数和数据集大小 |
| Stage 1 (SID生成) | ~5min | 推理速度较快 |
| Stage 2 (SFT) | ~4-8h | 约3-5个epoch早停 |
| Stage 3 (RL) | ~8-16h | 1650 steps × 1024 completions |
| 评测 | ~2-4h | Beam50×全部checkpoint |
| **总计** | **~26-52h** | 适合单卡48G周末运行 |

---

*本实验方案确保改进方法（RQ-MoE + GDPO）与基线方法（RQ-VAE + GRPO）的参数完全对齐，仅在SID构建方法和RL advantage计算方法上存在差异，保证实验结论的可信度和可复现性。*
