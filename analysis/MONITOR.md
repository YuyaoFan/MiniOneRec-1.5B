# MiniOneRec 训练监控与验证指南

> 训练已通过 screen 在后台启动 → `screen -r minionerec_0719_1409`

---

## 一、快速状态检查

```bash
# 1. 确认 screen session 存活
screen -ls | grep minionerec

# 2. 确认 GPU 被占用
nvidia-smi

# 3. 查看最新训练日志
tail -30 /root/autodl-tmp/MiniOneRec/logs/pipeline_20260719_140900/stage1_train.log

# 4. 进入 screen 实时查看
screen -r minionerec_0719_1409
# (Ctrl+A, D 离开)
```

---

## 二、Stage 1: RQ-MoE SID 构建 — 监控与验证

### 2.1 当前状态

```
数据: 3686 embeddings (2560-dim)
模型: 16,764,480 params (RQ-MoE, M=3, K=256, N=2, L=4, H=256)
训练: ~0.6s/epoch, 预计 10000 epochs 约 100 分钟
GPU: RTX 3090 (48GB)
日志: logs/pipeline_20260719_140900/stage1_train.log
TensorBoard: logs/tensorboard/stage1/Jul-19-2026_14-09-08/
```

### 2.2 实时监控命令

```bash
# 查看最新几行日志
tail -5 /root/autodl-tmp/MiniOneRec/logs/pipeline_20260719_140900/stage1_train.log

# 持续滚动查看日志
tail -f /root/autodl-tmp/MiniOneRec/logs/pipeline_20260719_140900/stage1_train.log

# 查看 GPU 使用情况
nvidia-smi
# 或交互式：nvitop

# 从日志中提取 loss 趋势
grep "epoch.*training" /root/autodl-tmp/MiniOneRec/logs/pipeline_20260719_140900/stage1_train.log | tail -20

# 从日志中提取 collision rate
grep "collision_rate" /root/autodl-tmp/MiniOneRec/logs/pipeline_20260719_140900/stage1_train.log | tail -10
```

### 2.3 训练正常标志

| 指标 | 正常范围 | 说明 |
|------|---------|------|
| Train loss | **持续下降** | 初始 ~6.0, 目标 ~0.5-1.0 |
| Recon loss | **持续下降** | 初始 ~3.3, 目标 ~0.4-0.8 |
| Collision rate | **持续下降** | 从高开始, 目标 < 10% |
| GPU 显存 | **稳定** | 预期 ~2-5 GB |
| GPU 利用率 | **> 50%** | 训练时应维持较高 |

### 2.4 异常检测

```bash
# 检查是否有 NaN
grep -i "nan" /root/autodl-tmp/MiniOneRec/logs/pipeline_20260719_140900/stage1_train.log

# 检查是否有 Error/CUDA OOM
grep -iE "error|exception|oom|killed" /root/autodl-tmp/MiniOneRec/logs/pipeline_20260719_140900/stage1_train.log

# 检查 loss 是否在下降（应逐渐减小）
grep "train loss" /root/autodl-tmp/MiniOneRec/logs/pipeline_20260719_140900/stage1_train.log | \
  awk '{print $NF}' | tail -50
# 去掉逗号后观察趋势
```

### 2.5 Stage 1 完成标志

当看到以下输出时, Stage 1 完成:
```
Training complete. Best loss: X.XXXXX, Best collision: Y.YYYYY
```
随后会自动进入 Stage 1b (SID 生成), 输出:
```
[Success] SID file generated at: data/Amazon/index/Industrial_and_Scientific.index.json
```

### 2.6 Stage 1 验证方法

```bash
# 训练中/结束后验证 SID 质量
source /root/miniconda3/etc/profile.d/conda.sh && conda activate MiniOneRec && python -c "
import json, numpy as np

# 1. 检查 index.json
with open('data/Amazon/index/Industrial_and_Scientific.index.json') as f:
    idx = json.load(f)

n_items = len(idx)
# 统计唯一 SID 数量
unique_sids = set(''.join(v) for v in idx.values())
collision_rate = (n_items - len(unique_sids)) / n_items

print(f'Total items: {n_items}')
print(f'Unique SIDs: {len(unique_sids)}')
print(f'Collision rate: {collision_rate:.4f} ({collision_rate*100:.2f}%)')

# 2. 检查 SID 格式
sample_key = list(idx.keys())[0]
sample_sid = idx[sample_key]
print(f'Sample SID: {sample_sid}')

# 3. 验证所有 SID 为三级格式
errors = 0
for k, v in idx.items():
    if len(v) < 3:
        errors += 1
        if errors <= 3: print(f'  Bad format: {k} -> {v}')
print(f'Format errors: {errors}')

# 4. 正常: collision_rate < 0.10 (10%), format_errors = 0
if collision_rate < 0.10 and errors == 0:
    print('[PASS] Stage 1 SID quality OK!')
else:
    print(f'[WARN] Stage 1 SID may need attention (collision={collision_rate:.4f}, errors={errors})')
"
```

---

## 三、Stage 2: SFT — 监控与验证

### 3.1 启动确认

Stage 1 完成后自动进入 Stage 2。日志输出:
```
=== Stage 2/3: Supervised Fine-Tuning ===
```

### 3.2 实时监控命令

```bash
# 查看 SFT 日志
tail -f /root/autodl-tmp/MiniOneRec/logs/pipeline_20260719_140900/stage2_sft.log

# 查看 SFT 训练进度 (HuggingFace Trainer 格式)
grep -E "'loss'|'eval_loss'|'epoch'" /root/autodl-tmp/MiniOneRec/logs/pipeline_20260719_140900/stage2_sft.log | tail -20
```

### 3.3 训练正常标志

| 指标 | 正常范围 | 说明 |
|------|---------|------|
| Train loss | 持续下降 | 初始 ~3-4, 收敛 ~0.3-0.8 |
| Eval loss | 持续下降 | 应与 train loss 趋势一致 |
| GPU 显存 | ~24-35 GB | SFT 占用较多显存 |
| Epoch 数 | ≤ 10 | 通常 3-5 epoch 早停 |

### 3.4 Stage 2 完成标志

日志出现:
```
[OK] Stage 2 complete! SFT model saved to output/sft/Industrial_and_Scientific_rqmoe
```

### 3.5 Stage 2 验证方法

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate MiniOneRec && python -c "
import os, json

# 1. 确认模型文件存在
sft_path = 'output/sft/Industrial_and_Scientific_rqmoe/final_checkpoint'
files = ['config.json', 'model.safetensors', 'tokenizer_config.json']
for f in files:
    fp = os.path.join(sft_path, f)
    exists = os.path.exists(fp)
    size = os.path.getsize(fp)/1024/1024 if exists else 0
    print(f'  [{\"OK\" if exists else \"MISSING\"}] {f} ({size:.1f} MB)')

# 2. 加载 config 验证
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(sft_path)
print(f'  Model: {cfg.model_type}, hidden={cfg.hidden_size}, vocab={cfg.vocab_size}')

# 3. 验证 tokenizer
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sft_path)
print(f'  Tokenizer vocab: {len(tok)}')

# 4. 检查 trainer_state (训练信息)
state_path = os.path.join('output/sft/Industrial_and_Scientific_rqmoe', 'checkpoint-*', 'trainer_state.json')
import glob
state_files = glob.glob(state_path)
if state_files:
    with open(state_files[-1]) as f:
        state = json.load(f)
    best = state.get('best_metric', 'N/A')
    print(f'  Best eval loss: {best}')
print('[PASS] Stage 2 SFT model verified!')
"
```

---

## 四、Stage 3: RL (GDPO) — 监控与验证

### 4.1 启动确认

Stage 2 完成后自动进入 Stage 3。日志输出:
```
=== Stage 3/3: Reinforcement Learning (GDPO) ===
```

### 4.2 实时监控命令

```bash
# 查看 RL 日志
tail -f /root/autodl-tmp/MiniOneRec/logs/pipeline_20260719_140900/stage3_rl.log

# 查看 RL 进度
grep -E "step|reward|loss" /root/autodl-tmp/MiniOneRec/logs/pipeline_20260719_140900/stage3_rl.log | tail -20

# 查看 eval snapshot 生成
ls -lt /root/autodl-tmp/MiniOneRec/output/rl/Industrial_and_Scientific_rqmoe_gdpo/eval_snapshots/ 2>/dev/null | head -10

# TensorBoard 监控 RL 指标
tensorboard --logdir logs/tensorboard --port 6006 --bind_all
```

### 4.3 训练正常标志

| 指标 | 正常方向 | 说明 |
|------|---------|------|
| reward/rule_reward | **上升** | 正确预测的比例增加 |
| reward/ndcg_rule_reward | **上升** | 排序质量改善 |
| categorical_diversity | **> 0.3** | 生成多样性 |
| KL divergence | **稳定/小幅波动** | 不应爆炸 |
| NDCG@5/10 (如开启test) | **上升** | 核心评测指标 |

### 4.4 Stage 3 完成标志

日志出现:
```
[OK] Stage 3 complete! RL checkpoints in output/rl/Industrial_and_Scientific_rqmoe_gdpo
```

### 4.5 Stage 3 验证方法

```bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate MiniOneRec && python -c "
import os, glob

# 1. 确认 eval_snapshots 存在
snap_dir = 'output/rl/Industrial_and_Scientific_rqmoe_gdpo/eval_snapshots'
snaps = sorted(glob.glob(os.path.join(snap_dir, 'checkpoint-*')))
print(f'Eval snapshots: {len(snaps)}')

# 2. 打印所有 snapshot 步数
for s in snaps:
    step = int(s.split('checkpoint-')[-1])
    has_config = os.path.exists(os.path.join(s, 'config.json'))
    print(f'  step={step} config={\"OK\" if has_config else \"MISSING\"}')

# 3. 检查完整 checkpoint (用于续训)
ckpt_dir = 'output/rl/Industrial_and_Scientific_rqmoe_gdpo'
ckpts = sorted(glob.glob(os.path.join(ckpt_dir, 'checkpoint-*')))
print(f'Full checkpoints: {len(ckpts)}')
for c in ckpts[-3:]:
    step = int(c.split('checkpoint-')[-1])
    print(f'  step={step}')

print(f'[PASS] Stage 3 RL outputs verified! ({len(snaps)} snapshots)')
"
```

---

## 五、Stage 4: 评测 — 监控与验证

### 5.1 实时监控

Stage 3 完成后自动进入评测:
```bash
tail -f /root/autodl-tmp/MiniOneRec/logs/pipeline_20260719_140900/stage4_eval.log
```

### 5.2 最终结果查看

```bash
# 查看汇总指标
cat results/eval/metrics_all.json | python -m json.tool

# 或格式化查看
python -c "
import json
with open('results/eval/metrics_all.json') as f:
    data = json.load(f)
for row in data:
    ndcg = row.get('ndcg', {})
    rec = row.get('recall', {})
    print(f\"[{row['name']:20s}] step={row.get('step','?'):5s}  \")
    print(f\"  NDCG@5={ndcg.get('5','?'):.6f}  NDCG@10={ndcg.get('10','?'):.6f}\")
    print(f\"  HR@5 ={rec.get('5','?'):.6f}  HR@10 ={rec.get('10','?'):.6f}\")
"
```

---

## 六、TensorBoard 可视化

```bash
# 启动 TensorBoard (在另一个 screen 中)
screen -S tensorboard
tensorboard --logdir /root/autodl-tmp/MiniOneRec/logs/tensorboard --port 6006 --bind_all
# Ctrl+A, D 离开

# 浏览器访问 (需要端口转发):
# http://<服务器IP>:6006

# 或用 tmux 监控面板
bash scripts/monitor_panel.sh
# tmux attach -t monitor
```

---

## 七、常见问题处理

### 7.1 训练中断

```bash
# 如果 screen session 还在但训练卡住了
screen -r minionerec_0719_1409
# Ctrl+C 停止当前阶段

# 重新启动 (会从断点继续)
bash scripts/launch_full_pipeline.sh --screen
```

### 7.2 GPU OOM

```bash
# 查看显存使用
nvidia-smi

# 如果 OOM, 调整 batch_size:
# Stage 1: 修改 rq/train_rqmoe.sh 中的 --batch_size (减小)
# Stage 2: 修改 sft.sh 中的 --micro_batch_size (减小)
# Stage 3: 修改 run_rl_gdpo.sh 中的 --train_batch_size (减小)
```

### 7.3 查看训练趋势

```bash
# 提取 Stage 1 loss 曲线数据
grep "epoch.*training" logs/pipeline_20260719_140900/stage1_train.log | \
  sed 's/.*epoch \([0-9]*\) training.*train loss: \([0-9.]*\).*reconstruction loss: \([0-9.]*\).*/\1 \2 \3/' \
  > /tmp/stage1_loss.txt

# 可视化 (需要 matplotlib)
python -c "
import numpy as np
data = np.loadtxt('/tmp/stage1_loss.txt')
epochs, total_loss, recon_loss = data[:,0], data[:,1], data[:,2]
print(f'Epochs: {len(epochs)}')
print(f'Initial loss: {total_loss[0]:.4f}')
print(f'Latest loss: {total_loss[-1]:.4f}')
print(f'Improvement: {(1-total_loss[-1]/total_loss[0])*100:.1f}%')
"
```

---

## 八、Screen 管理速查

```bash
screen -ls                          # 列出所有 session
screen -r minionerec_0719_1409      # 连接到训练 session
screen -r tensorboard               # 连接到 TensorBoard (如果创建了)
# 在 screen 内:
#   Ctrl+A, D  = 离开 (detach)
#   Ctrl+C     = 终止当前程序
#   Ctrl+A, K  = 杀掉当前窗口
screen -S <name> -X quit            # 从外部终止 session
```
