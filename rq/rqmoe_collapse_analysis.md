# RQ-MoE 码本坍塌分析与优化方案

> **背景**: RQ-MoE 替代 RQ-VAE 后, Level 0 码本完全坍塌 — 3686 items 全部映射到同一首码 `<a_115>`。
> RQ-VAE 则三级全利用 (256+256+256 codes), SFT NDCG@10 领先 RQ-MoE 20.3%。

---

## 一、坍塌根因分析

### 1.1 梯度被完全截断 (根本原因)

```python
# rq/models/rqmoe.py:321-322
def forward(self, x):
    with torch.no_grad():          # ← 🔴 编码阶段梯度全部截断
        codes, _ = self.encode(x)

    # 后续只通过 decode 路径计算 loss
    losses[0] = ((self.codebook0(codes[:, 0]) - x) ** 2).sum()
    for i, step in enumerate(self.steps):
        toadd, ins = step.decode(codes[:, i + 1], instruct)
        ...
```

**问题**: `encode()` 调用 `assign_to_codebook()` 做 argmin 分配, 已被 `torch.no_grad()` 包裹。
argmin 本身不可导, 需要 Straight-Through Estimator (STE) 或 Gumbel-Softmax 传递梯度。
当前实现中, 梯度**完全无法到达 codebook0 和 encoder**。

**对比 RQ-VAE**: VectorQuantizer 使用 `x_q = x + (x_q - x).detach()` (STE), 将量化损失
的梯度传回 encoder, 使编码器能够调整输出以匹配码本分布。

### 1.2 投影维度压缩过高

```python
# RQMoEWrapper: in_dim=2560 → input_proj → e_dim=64
self.input_proj = nn.Linear(2560, 64)   # 40x 压缩
```

64 维空间中, 3686 个向量被压缩到极低维度。随机初始化下, 投影后的向量很可能
集中在第一码本向量的 Voronoi 区域内, 导致所有输入被分配到同一码字。

**对比 RQ-VAE**: `e_dim=2560`, 在原始高维空间进行量化, 保留了充分的区分度。

### 1.3 编码器零初始化 + 无残差连接

```python
# rq/models/rqmoe.py:498-511
def _zero_init_last_layer(self):
    for m in reversed(list(encoder_mod.modules())):
        if isinstance(m, nn.Linear):
            last_linear = m; break
    if last_linear is not None:
        last_linear.weight.fill_(0.0)   # 🔴 最后一层权重全零
```

零初始化后 encoder 输出 ≈ 0, 所以 `z = input_proj(x) + encoder(input_proj(x)) ≈ input_proj(x)`。
理论上是合理的残差 init 策略, 但由于梯度截断, encoder **永远无法从零状态学习到
有意义的表示**。实际效果等同于:

```
z ≈ Linear(2560→64)(x)   # 单一的线性投影, 无非线性能力
```

**对比 RQ-VAE**: 虽有 `ResidualEncoderWrapper` (`z = x + MLP(x)`), 但:
- `e_dim=2560` (无压缩), `z ≈ x` 天然保留多样性
- STE 传递梯度, encoder 可被正常训练

### 1.4 码本初始化不当

RQ-MoE 使用 `nn.Embedding` 的默认均匀初始化 (范围由 kaiming/glorot 决定)。
在 64 维空间中, 256 个均匀随机码本向量的 Voronoi 剖分容易出现少数几个码本
占据绝大多数样本的情况。

**对比 RQ-VAE**: 使用 Constrained K-Means 将聚类中心作为码本初始值,
确保了码本向量均匀分布在整个数据流形上。

---

## 二、链式反应: 从 Level 0 坍塌到全局退化

```
encode(x) with no_grad
    ↓
码本向量随机初始化 → 64维空间 → 样本集中在少数 Voronoi cells
    ↓
Level 0 assign: 所有 3686 样本 → 同一 code (如 index=115)
    ↓
codebook0.weight[115] 是唯一的首层重建贡献
    ↓
residual = x - code0[115]  → 残留了几乎全部信息
    ↓
Level 1-2 RQ-MoE: 在残留上动态码本, 但 2 层难以弥补首层损失
    ↓
最终 SID: 仅 2 级有效 + dedup suffix
```

---

## 三、优化方案 (按优先级排序)

### 方案 A: 移除 `no_grad()` + 使用 Straight-Through Estimator (推荐)

**修改点**: `rq/models/rqmoe.py` 的 `forward()` 方法

```python
def forward(self, x):
    # 移除 torch.no_grad(), 改为在 argmin 处使用 STE
    codes, x_hat_enc = self.encode(x)  # enc 中 argmin 不可导, 但...

    # STE: 用编码器的输出 + 量化输出的 detach 拷贝
    x_hat_ste = x_hat_enc + (self.codebook0(codes[:, 0]) - x_hat_enc).detach()
    # 梯度可以通过 codebook0.weight, encoder 参数回流

    losses = torch.zeros(self.M, device=x.device)
    losses[0] = ((x_hat_ste - x) ** 2).sum()
    ...
```

**核心**: 让 `encode()` 的码本查找**知道**其索引来自哪个计算, 然后用 STE
把梯度"复制"到编码器输出上。

**预期效果**: 编码器收到梯度 → 调整输出分布 → 码本利用趋于均衡

### 方案 B: 增大 e_dim 至 256 (或更高)

**修改点**: `rq/train_rqmoe.sh` 和 `rq/generate_indices_rqmoe.sh`

```bash
--e_dim 256   # 从 64 → 256 (4x 提升, 压缩比从 40x → 10x)
```

也可去掉 encoder/decoder 瓶颈, 直接在 2560 维上做 RQ-MoE (匹配 RQ-VAE 的 e_dim=2560):

```bash
--e_dim 2560
--layers        # 设为空, 不使用 bottleneck
```

**预期效果**: 更高维空间保留更多区分度, 降低码本坍塌风险

### 方案 C: K-Means 初始化码本

**修改点**: `rq/models/rqmoe.py` 的 `RQMoE.__init__()`

```python
def init_codebook(self, data):
    """用 K-Means 初始化所有码本"""
    from sklearn.cluster import KMeans
    residual = data
    for level in range(self.M):
        km = KMeans(n_clusters=self.K, n_init=10)
        km.fit(residual.cpu().numpy())
        centroids = torch.tensor(km.cluster_centers_, device=data.device)
        if level == 0:
            self.codebook0.weight.data.copy_(centroids)
        else:
            self.steps[level-1].codebook.weight.data.copy_(centroids)
        # 更新 residual
        codes = km.labels_
        residual = residual - centroids[codes]
```

**预期效果**: 初始化阶段就保证码本向量均匀覆盖数据分布

### 方案 D: 添加 Commitment Loss 或 EMA 更新

**修改点**: `rq/models/rqmoe.py` 的 `forward()` 方法

```python
# 在 forward() 中加入 commitment loss (参考 VQ-VAE)
commitment_loss = F.mse_loss(x_hat, x.detach()) + \
                  self.beta * F.mse_loss(x_hat.detach(), x)

# 或使用 EMA (指数移动平均) 更新码本
if self.training:
    self._ema_update_codebook(x, codes)
```

EMA 更新类似经典的 VQ-VAE 训练技巧, 让码本向量缓慢向编码器输出靠近,
而非依赖梯度直接更新。

### 方案 E: 引入 Sinkhorn 软分配 (参考 RQ-VAE)

**修改点**: `assign_to_codebook()` 替换为 Sinkhorn-based 软分配

RQ-VAE 的 VectorQuantizer 使用 Sinkhorn 算法将 argmin 替换为 soft assignment,
在训练早期提供平滑的码本梯度。训练后期逐渐退化为硬分配。

```python
def assign_soft_to_codebook(x, c, epsilon=0.01, iters=10):
    """Sinkhorn-based soft assignment with temperature"""
    d = pairwise_distances(x, c)
    Q = sinkhorn_algorithm(-d / epsilon, epsilon, iters)  # soft assignment
    return Q  # [N, K] soft assignment matrix
```

**预期效果**: 训练初期所有码本都能接收到梯度, 避免硬分配导致的"赢者通吃"

---

## 四、组合推荐方案

| 优先级 | 方案 | 难度 | 预期效果 | 风险 |
|-------|------|------|---------|------|
| **P0** | A: 移除 no_grad + STE | 中 | 核心修复 | 训练可能不稳定 |
| **P1** | B: e_dim=256 | 低 | 大幅改善 | 参数量增加 |
| **P2** | C: K-Means 初始化 | 低 | 基础改善 | 无 |
| P3 | D: Commitment Loss | 高 | 辅助稳定 | 需调参 |
| P4 | E: Sinkhorn 软分配 | 高 | 最平滑 | 训练早期慢 |

**最小可行方案 (MVP)**: A + B + C 三者组合:
1. **移除 no_grad + STE** 使 encoder 可训练
2. **e_dim=256** 保留足够区分度
3. **K-Means 初始化** 提供良好起点

---

## 五、实施步骤

### Phase 1: 环境准备
- 复用现有 baseline RQ-VAE 的 Constrained K-Means 码本作为 RQ-MoE 初始化
- 新增 `--kmeans_init` 和 `--kmeans_ckpt` 参数到 `train_rqmoe.py`

### Phase 2: 梯度修复
- 重写 `rq/models/rqmoe.py` 的 `forward()` method
- 在 `encode()` 的 argmin 处插入 STE
- 验证梯度能到达 encoder 和 codebook0

### Phase 3: 参数调整
- `e_dim: 64 → 256`
- `moe_H: 256 → 512` (补偿维度增加)
- 训练监控: collision rate 应持续下降, Level 0 码本利用率 > 50%

### Phase 4: 验证
- 训练完成后检查 Level 0/1/2 的唯一 token 数量
- 与 RQ-VAE baseline 对比 SID 分布熵
- 确认无码本坍塌后, 进入 SFT + RL 流程

---

## 六、参考

- RQ-VAE 源码: `rq/models/vq.py` (STE: `x_q = x + (x_q - x).detach()`)
- RQ-VAE 源码: `rq/models/rqvae.py` (ResidualEncoderWrapper)
- Van Den Oord et al., "Neural Discrete Representation Learning" (VQ-VAE, 2017)
- Zeghidour et al., "SoundStream" (RQ-VAE, 2022)
- Zhong et al., "RQ-MoE" (ICML 2026)
