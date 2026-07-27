# MiniOneRec 运行指南

> 基线：Qwen2.5-1.5B、单卡48G、Amazon (Industrial_and_Scientific)
> 改进方法：RQ-MoE SID构建 + GDPO强化学习

---

## 目录

- [0. 环境准备](#0-环境准备)
- [1. 快速启动（一键全流程）](#1-快速启动一键全流程)
- [2. 分阶段运行](#2-分阶段运行)
  - [Stage 1: SID码本构建](#stage-1-sid码本构建)
  - [Stage 2: 数据转换与SFT](#stage-2-数据转换与sft)
  - [Stage 3: 强化学习（RL）](#stage-3-强化学习rl)
  - [Stage 4: 评测](#stage-4-评测)
- [3. 后台运行（screen/tmux）](#3-后台运行screentmux)
- [4. 训练过程可视化](#4-训练过程可视化)
- [5. 改进方法实验](#5-改进方法实验)
- [6. 断点续训与故障恢复](#6-断点续训与故障恢复)
- [7. 常用命令速查](#7-常用命令速查)

---

## 0. 环境准备

```bash
# 1. 确认环境
cd /root/autodl-tmp/MiniOneRec
conda activate MiniOneRec

# 2. 检查关键依赖
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
python -c "import transformers; print(f'Transformers {transformers.__version__}')"
python -c "import trl; print(f'TRL {trl.__version__}')"
nvidia-smi

# 3. 确认数据完整性
ls data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy   # 文本嵌入 (~55M)
ls data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv
ls data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv
ls data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv
ls data/Amazon/info/Industrial_and_Scientific_5_2016-10-2018-11.txt
ls data/Amazon/index/Industrial_and_Scientific.item.json

# 4. 确认模型权重
ls models/Qwen2.5-1.5B/   # 需预先下载（或从 HuggingFace 自动拉取）

# 5. 创建工作目录
mkdir -p checkpoints/rqmoe output/sft output/rl results/eval logs/tensorboard
```

### SFT / RL 训练中用到的 Baseline 产物

| 文件 | 说明 | 是否必须 |
|------|------|---------|
| `data/Amazon/index/*.emb-qwen-td.npy` | item 文本嵌入向量 | Stage 1 必须 |
| `data/Amazon/index/*.item.json` | item 元数据（标题/描述） | Stage 2/3/4 必须 |
| `data/Amazon/index/*.index.json` | item → SID 映射 | Stage 2/3/4 必须 |
| `data/Amazon/index/*.codebooks_constrained.npz` | RQ-VAE 初始化码本（仅 RQ-VAE 路线需要） | Stage 1 RQ-VAE 路线 |

---

## 1. 快速启动（一键全流程）

### Baseline 路线：RQ-VAE + SFT + RL(GRPO)

```bash
cd /root/autodl-tmp/MiniOneRec
bash scripts/run_baseline.sh
```

### 改进路线：RQ-MoE + SFT + RL(GDPO)

```bash
cd /root/autodl-tmp/MiniOneRec
bash scripts/run_improved.sh
```

两个脚本自动依次执行 Stage 1→2→3→4，每个 stage 结束后自动进入下一阶段。
如需在后台运行，请见 [§3. 后台运行](#3-后台运行screentmux)。

---

## 2. 分阶段运行

### Stage 1: SID码本构建

SID 是整个 pipeline 的地基，将 item 文本嵌入压缩为三级离散编码。

#### Baseline: RQ-VAE（已完成，可跳过）

```bash
cd rq

# Step 1: Constrained K-means 初始化码本
bash rqkmeans_constrained.sh
# 产出: data/Amazon/index/Industrial_and_Scientific.codebooks_constrained.npz

# Step 2: RQ-KMeans++ 训练
bash rqkmeans_plus.sh
# 产出: rq/Jun-24-2026_11-38-28/best_collision_model.pth

# Step 3: 生成 SID index.json
bash generate_indices_plus.sh
# 产出: data/Amazon/index/Industrial_and_Scientific.index.json
```

#### Improved: RQ-MoE

```bash
cd rq

# Step 1: 训练 RQ-MoE 模型
# 参配：M=3层，K=256，N=2专家，L=4层/专家，H=256隐藏维
bash train_rqmoe.sh

# 训练完成后，记下 best_collision_model.pth 路径，例如：
# checkpoints/rqmoe/Mar-15-2026_14-30-00/best_collision_model.pth

# Step 2: 用训练好的 RQ-MoE 生成 SID
# 修改 generate_indices_rqmoe.sh 中的 --ckpt_path 为上面记下的路径
vim generate_indices_rqmoe.sh   # 更新 CKPT_PATH
bash generate_indices_rqmoe.sh
# 产出: data/Amazon/index/Industrial_and_Scientific.index.json
```

**Stage 1 关键监控指标**：

| 指标 | 查看方式 | 含义 |
|------|---------|------|
| Train Loss | 终端输出 | 越低越好，RQ-MoE 的 NRL loss 在 0.01-0.1 区间 |
| Collision Rate | 终端输出 | SID 碰撞率，< 5% 为佳 |
| 唯一 SID 数量 | generate_indices 输出 | 应接近 item 总数 |

---

### Stage 2: 数据转换与SFT

```bash
cd /root/autodl-tmp/MiniOneRec

# Step 1: 数据转换（CSV + index.json → SFT训练样本）
bash convert_dataset.sh
# 产出: data/Amazon/train/ 和 valid/ 下的 .csv 文件

# Step 2: SFT 训练
# sft.sh 默认使用 RQ-VAE 生成的 index.json
# 如果用 RQ-MoE 的 index.json，请确认 --sid_index_path 指向正确文件
bash sft.sh

# 关键超参（sft.sh 内）:
#   --base_model ./models/Qwen2.5-1.5B
#   --batch_size 1024 --micro_batch_size 16
#   --num_epochs 10 (EarlyStopping patience=3)
#   --learning_rate 3e-4
#   --seed 42
# 产出: output/sft/Industrial_and_Scientific_plus/
```

**SFT 监控**：

```bash
# 查看训练日志（screen 模式下）
tail -f logs/sft_$(date +%Y%m%d).log

# TensorBoard 可视化（另开终端）
tensorboard --logdir logs/tensorboard/sft --port 6006 --bind_all
```

---

### Stage 3: 强化学习（RL）

```bash
cd /root/autodl-tmp/MiniOneRec

# Baseline: GRPO
bash rl.sh

# Improved: GDPO（在 rl.sh 中添加 --gdpo True）
# 直接使用改进脚本：
bash scripts/run_rl_gdpo.sh

# 关键超参:
#   --model_path ./output/sft/Industrial_and_Scientific_plus
#   --train_batch_size 32 --gradient_accumulation_steps 32
#   --num_generations 16
#   --num_train_epochs 2
#   --learning_rate 1e-5
#   --beta 1e-3
#   --reward_type ranking
#   --beam_search True
# 产出: output/rl/Industrial_and_Scientific_plus/
#       ├── checkpoint-*/              # 完整 checkpoint（含优化器，用于续训）
#       └── eval_snapshots/checkpoint-*/ # 评测快照（~3G/个，不含优化器）
```

**RL 监控**：

```bash
# TensorBoard（推荐）
tensorboard --logdir logs/tensorboard/rl --port 6007 --bind_all

# 查看 checkpoint 列表
ls output/rl/Industrial_and_Scientific_plus/eval_snapshots/

# 实时查看指标
watch -n 5 'ls -t output/rl/Industrial_and_Scientific_plus/eval_snapshots/ | head -5'
```

---

### Stage 4: 评测

#### 4a. 单模型评测

```bash
cd /root/autodl-tmp/MiniOneRec

# 编辑 evaluate.sh：修改 exp_name 为待评模型路径
exp_name="output/rl/Industrial_and_Scientific_plus/eval_snapshots/checkpoint-1650"

bash evaluate.sh
# 产出: results/<exp_name>/final_result_Industrial_and_Scientific.json
```

#### 4b. 批量评测（SFT基线 + 所有RL快照 + 自动绘图）

```bash
cd /root/autodl-tmp/MiniOneRec

# 评测 SFT + 所有 RL eval_snapshots
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0 \
python tools/eval_all.py --batch_size 8

# 产物:
#   results/eval/pred_sft.json
#   results/eval/pred_rl-*.json
#   results/eval/metrics_all.json     # 所有模型指标汇总
#   assets/plot_ndcg.png              # NDCG 曲线
#   assets/plot_recall.png            # Recall 曲线
```

#### 4c. 只看已保存指标（无需重新推理）

```bash
cat results/eval/metrics_all.json | python -m json.tool
```

---

## 3. 后台运行（screen/tmux）

### 3.1 使用 screen（推荐，简单稳定）

```bash
# 安装（如未安装）
apt-get install -y screen

# === 创建 session ===
screen -S stage1     # Stage 1: SID 构建
screen -S sft        # Stage 2: SFT
screen -S rl         # Stage 3: RL

# === 在 session 内运行 ===
# screen -S stage1
cd /root/autodl-tmp/MiniOneRec/rq
bash train_rqmoe.sh 2>&1 | tee ../logs/stage1_$(date +%Y%m%d_%H%M%S).log

# screen -S sft
cd /root/autodl-tmp/MiniOneRec
bash sft.sh 2>&1 | tee logs/sft_$(date +%Y%m%d_%H%M%S).log

# screen -S rl
cd /root/autodl-tmp/MiniOneRec
bash rl.sh 2>&1 | tee logs/rl_$(date +%Y%m%d_%H%M%S).log

# === 离开 session（不中断任务）===
# 按 Ctrl+A，然后按 D

# === 重新连接 ===
screen -r stage1     # 重连指定 session
screen -ls           # 查看所有 session
screen -r            # 只有一个 session 时直接重连

# === 结束 session ===
# 在 session 内: exit 或 Ctrl+D
```

### 3.2 使用 tmux（功能更强，分屏方便）

```bash
# 安装
apt-get install -y tmux

# === 创建 session ===
tmux new -s minionerec

# === 分屏（可选，同时监控多阶段）===
# Ctrl+B, %    垂直分屏
# Ctrl+B, "    水平分屏
# Ctrl+B, 方向键  切换窗格

# === 在窗格内运行任务 ===
cd /root/autodl-tmp/MiniOneRec

# 窗格1: Stage 1
cd rq && bash train_rqmoe.sh 2>&1 | tee ../logs/stage1.log

# 窗格2: 监控（等待 Stage 1 完成后再启动 Stage 2）
watch -n 30 'echo "Waiting for Stage 1..."'

# === 离开 ===
# Ctrl+B, D

# === 重新连接 ===
tmux attach -t minionerec

# === 常用 tmux 命令 ===
tmux ls                    # 列出所有 session
tmux kill-session -t name  # 删除 session
```

### 3.3 一键后台启动脚本

```bash
# Baseline 全流程（screen）
bash scripts/screen_run_baseline.sh

# 改进方法全流程（screen）
bash scripts/screen_run_improved.sh

# 改进方法全流程（tmux，带分屏监控）
bash scripts/tmux_run_improved.sh
```

---

## 4. 训练过程可视化

### 4.1 TensorBoard（推荐）

每个 stage 都自动写入 TensorBoard 日志到 `logs/tensorboard/<stage>/`：

```bash
# 启动 TensorBoard（监控所有 stage）
tensorboard --logdir logs/tensorboard --port 6006 --bind_all

# 浏览器访问: http://<服务器IP>:6006
# 如果是本地: http://localhost:6006

# 后台运行 TensorBoard
screen -S tensorboard
tensorboard --logdir logs/tensorboard --port 6006 --bind_all
# Ctrl+A, D 离开

# TensorBoard 可监控:
# - Stage 1: train/loss, train/recon_loss, eval/collision_rate
# - Stage 2: train/loss, eval/loss, train/learning_rate
# - Stage 3: rewards/rule_reward, rewards/ndcg_rule_reward,
#             categorical_diversity, token_diversity, kl,
#             NDCG@3/5/10, HR@3/5/10
```

### 4.2 WandB（备选，需 API key）

```bash
# 登录（一次性）
wandb login <your-api-key>

# 启用 WandB（修改对应 .sh 脚本）
# 将 --report_to none 改为 --report_to wandb
# 设置 --wandb_project "MiniOneRec"
# 设置 --wandb_run_name "experiment-name"

# 浏览器访问: https://wandb.ai/<your-username>/MiniOneRec
```

### 4.3 终端实时监控

```bash
# GPU使用率（推荐 nvitop）
pip install nvitop
nvitop                         # 交互式 GPU 监控

# 或使用 nvidia-smi
watch -n 1 nvidia-smi

# 显存使用
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# 实时查看训练日志
tail -f logs/stage1_*.log      # Stage 1
tail -f logs/sft_*.log         # Stage 2
tail -f logs/rl_*.log          # Stage 3

# 查看最新 RL checkpoint
watch -n 10 'echo "Latest checkpoints:"; ls -lt output/rl/Industrial_and_Scientific_plus/eval_snapshots/ | head -5'
```

### 4.4 自定义监控面板

```bash
# 安装 tmux + 自定义布局一键监控
bash scripts/monitor_panel.sh

# 自动打开4分屏 tmux 布局:
# ┌─────────────────┬─────────────────┐
# │   nvitop GPU    │  TensorBoard    │
# │   监控面板       │  localhost:6006 │
# ├─────────────────┼─────────────────┤
# │  Training Log   │  Eval Metrics   │
# │  tail -f         │  watch metrics  │
# └─────────────────┴─────────────────┘
```

---

## 5. 改进方法实验

### 5.1 实验矩阵

| 实验 | SID方法 | RL方法 | 启动命令 | 状态 |
|------|---------|--------|---------|------|
| Exp-A (基线) | RQ-VAE | GRPO | `bash scripts/run_baseline.sh` | ✅ 已完成 |
| Exp-B | RQ-MoE | GRPO | `bash scripts/run_rqmoe_grpo.sh` | 🔲 可选 |
| Exp-C | RQ-VAE | GDPO | `bash scripts/run_rqvae_gdpo.sh` | 🔲 可选 |
| **Exp-D** | **RQ-MoE** | **GDPO** | `bash scripts/run_improved.sh` | 🔲 **主实验** |

### 5.2 主实验：RQ-MoE + SFT + RL(GDPO)

```bash
cd /root/autodl-tmp/MiniOneRec

# 一步启动（screen 后台）
bash scripts/screen_run_improved.sh

# 或手动分步（见 §2）
```

### 5.3 结果对比

```bash
# 生成对比报告
python tools/compare_results.py \
    --baseline results/eval/rqvae_grpo_metrics.json \
    --improved results/eval/rqmoe_gdpo_metrics.json \
    --output results/comparison_report.md

# 关键表格自动生成（Markdown 格式，可直接写入论文/简历）
```

---

## 6. 断点续训与故障恢复

### 6.1 SFT 续训

```bash
# SFT 使用 HuggingFace Trainer，默认 load_best_model_at_end=True
# 重新运行 sft.sh 会从头开始
# 如需从 checkpoint 续训：
python sft.py \
    --resume_from_checkpoint output/sft/Industrial_and_Scientific_plus/checkpoint-XXX \
    ... (其他参数不变)
```

### 6.2 RL 续训（自动）

```bash
# RL 已内置断点续训：崩溃后重跑同一条命令即可自动续训
# 检查是否有可用的 checkpoint：
ls output/rl/Industrial_and_Scientific_plus/checkpoint-*

# 如有 checkpoint，直接重跑 rl.sh:
bash rl.sh
# 输出: [resume] found N checkpoint(s) in ..., resuming from latest
```

### 6.3 常见问题

| 问题 | 解决方案 |
|------|---------|
| CUDA OOM | 减小 `micro_batch_size` 或 `per_device_train_batch_size` |
| 训练 loss=NaN | 降低 `learning_rate`，检查数据是否有 NaN |
| 碰撞率>10% | 增加 `epochs` 或调整码本初始化 |
| SFT eval loss 不降 | 检查 index.json 格式是否正确 |
| RL reward 不升 | 调整 `beta` (KL penalty)，增加 `num_generations` |

---

## 7. 常用命令速查

```bash
# === 环境 ===
conda activate MiniOneRec
nvidia-smi

# === 数据 ===
ls data/Amazon/index/Industrial_and_Scientific.*
wc -l data/Amazon/train/Industrial_and_Scientific*.csv

# === SID 构建 ===
cd rq
bash train_rqmoe.sh                              # RQ-MoE 训练
bash generate_indices_rqmoe.sh                    # SID 生成
cd ..

# === SFT ===
bash convert_dataset.sh                           # 数据转换
bash sft.sh                                       # SFT 训练

# === RL ===
bash rl.sh                                        # Baseline: GRPO
bash scripts/run_rl_gdpo.sh                       # Improved: GDPO

# === 评测 ===
bash evaluate.sh                                  # 单模型评测
python tools/eval_all.py --batch_size 8           # 批量评测+画图

# === 监控 ===
nvitop                                            # GPU 监控
tensorboard --logdir logs/tensorboard --port 6006 # TensorBoard
screen -ls                                        # 查看 screen session
tmux ls                                           # 查看 tmux session

# === 日志 ===
tail -f logs/stage1_*.log
tail -f logs/sft_*.log
tail -f logs/rl_*.log

# === 结果 ===
cat results/eval/metrics_all.json | python -m json.tool
ls results/eval/pred_*.json
```

---

## 文件结构速查

```
MiniOneRec/
├── RUN.md                          # ← 本文件
├── experiment_plan.md              # 实验设计文档
├── MiniOneRec.md                   # 技术报告
├── scripts/                        # 一键启动脚本
│   ├── run_baseline.sh             # Baseline 全流程
│   ├── run_improved.sh             # 改进方法全流程
│   ├── screen_run_improved.sh      # 改进方法（screen 后台）
│   ├── tmux_run_improved.sh        # 改进方法（tmux 后台+监控）
│   ├── run_rl_gdpo.sh              # RL(GDPO)单独启动
│   └── monitor_panel.sh            # tmux监控面板
├── logs/                           # 运行日志
│   ├── stage1_*.log
│   ├── sft_*.log
│   ├── rl_*.log
│   └── tensorboard/                # TensorBoard 事件文件
├── checkpoints/                    # Stage 1 模型权重
│   └── rqmoe/
├── output/                         # Stage 2/3 产出
│   ├── sft/
│   └── rl/
└── results/                        # 评测结果
    └── eval/
```

---

*更多技术细节见 [MiniOneRec.md](MiniOneRec.md)（技术报告）、[experiment_plan.md](experiment_plan.md)（实验设计）、[paper/sid_opt.md](paper/sid_opt.md)（RQ-MoE分析）、[paper/rl_opt.md](paper/rl_opt.md)（GDPO分析）。*
