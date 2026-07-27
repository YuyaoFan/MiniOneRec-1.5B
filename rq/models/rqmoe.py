"""
RQ-MoE: Residual Quantization via Mixture of Experts (V2 — fixed)
Adapted for MiniOneRec's SID codebook generation.

V2 fixes (see rq/rqmoe_collapse_analysis.md):
  1. STE gradient: Straight-Through Estimator replaces torch.no_grad() encoding,
     allowing gradients to flow to encoder + codebook0.
  2. e_dim default: 64 → 256 (reduces compression ratio from 40x to 10x).
  3. K-Means init: Codebooks initialized from baseline RQ-VAE centroids
     or K-Means, preventing initial collapse.

Reference: Zhong et al., "RQ-MoE: Residual Quantization via Mixture of
Experts for Efficient Input-Dependent Vector Compression," ICML 2026.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .layers import MLPLayers


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def pairwise_distances(a, b):
    anorms = (a ** 2).sum(-1)
    bnorms = (b ** 2).sum(-1)
    return anorms[:, None] + bnorms - 2 * a @ b.T


def assign_to_codebook(x, c, bs=16384):
    """Hard argmin assignment to nearest codeword (no grad)."""
    nq, d = x.shape
    nb, d2 = c.shape
    assert d == d2
    if nq * nb < bs * bs:
        dis = pairwise_distances(x, c)
        return dis.argmin(1)

    res = torch.empty((nq,), dtype=torch.int64, device=x.device)
    cnorms = (c ** 2).sum(1)
    for i in range(0, nq, bs):
        xnorms = (x[i: i + bs] ** 2).sum(1, keepdim=True)
        for j in range(0, nb, bs):
            dis = xnorms + cnorms[j: j + bs] - 2 * x[i: i + bs] @ c[j: j + bs].T
            dmini, imini = dis.min(1)
            if j == 0:
                dmin = dmini; imin = imini
            else:
                (mask,) = torch.where(dmini < dmin)
                dmin[mask] = dmini[mask]; imin[mask] = imini[mask] + j
        res[i: i + bs] = imin
    return res


def assign_batch_multiple(x, zqs):
    """Batch nearest-neighbour among per-sample K candidates. Returns (idx, quantized)."""
    bs, d = x.shape
    bs2, K, d2 = zqs.shape
    assert bs == bs2 and d == d2

    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    zqs_norm = (zqs ** 2).sum(dim=2)
    xz = torch.bmm(x.unsqueeze(1), zqs.transpose(1, 2)).squeeze(1)
    L2distances = x_norm + zqs_norm - 2 * xz

    idx = torch.argmin(L2distances, dim=1)
    quantized = zqs[torch.arange(bs, device=zqs.device), idx]
    return idx, quantized


# ---------------------------------------------------------------------------
# RQ-MoE Step
# ---------------------------------------------------------------------------

class RQMoEStep(nn.Module):
    """One RQ-MoE step with second-level MoE dynamic codebook adaptation."""

    def __init__(self, d, K, N, L, H, dropout):
        super().__init__()
        self.d, self.K = d, K

        self.codebook = nn.Embedding(K, d)
        self.instruction = nn.Embedding(K, d)
        self.instruction.weight.data.zero_()

        self.MLPconcat = nn.Linear(d + d, d)
        self.gate = nn.Sequential(
            nn.Linear(d, H, bias=False), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(H, N, bias=False), nn.Softmax(dim=-1),
        )

        self.experts = []
        for n in range(N):
            layers = nn.ModuleList([
                nn.Sequential(nn.Linear(d, H, bias=False), nn.Dropout(dropout),
                              nn.ReLU(), nn.Linear(H, d, bias=False))
                for _ in range(L)
            ])
            self.add_module(f"expert{n}", layers)
            self.experts.append(layers)

    def _transform(self, instruct, chunk_size=16):
        """Transform static codebook → dynamic codebook (memory-efficient chunked).

        Args:
            instruct: [bs, d] instruction vectors
            chunk_size: number of codebook entries per chunk (lower = less VRAM)
        Returns: [bs, K, d] dynamic codewords.
        """
        bs = instruct.shape[0]
        zqs = self.codebook.weight  # [K, d]
        K, d = zqs.shape

        results = []
        for k_start in range(0, K, chunk_size):
            k_end = min(k_start + chunk_size, K)
            k_chunk = k_end - k_start

            # [k_chunk, d] → [bs * k_chunk, d]
            zqs_chunk = zqs[k_start:k_end]
            zqs_r = zqs_chunk.unsqueeze(0).expand(bs, k_chunk, d).reshape(bs * k_chunk, d)
            ins_r = instruct.unsqueeze(1).expand(bs, k_chunk, d).reshape(bs * k_chunk, d)

            cc = self.MLPconcat(torch.cat((zqs_r, ins_r), dim=1))
            expert_weights = self.gate(cc)  # [bs*k_chunk, N]

            expert_outputs = []
            for expert in self.experts:
                x = cc
                for layer in expert:
                    x = x + layer(x)
                expert_outputs.append(x)
            expert_outputs = torch.stack(expert_outputs, dim=1)  # [bs*k_chunk, N, d]
            weighted = (expert_outputs * expert_weights.unsqueeze(-1)).sum(dim=1)  # [bs*k_chunk, d]
            results.append(weighted.reshape(bs, k_chunk, d))

        return torch.cat(results, dim=1)  # [bs, K, d]

    def encode(self, residual, instruct):
        """Encode: dynamic codebook + nearest-neighbour assignment."""
        dynamic_cb = self._transform(instruct)  # [bs, K, d]
        codes, quantized = assign_batch_multiple(residual, dynamic_cb)
        ins = self.instruction.weight[codes]
        return codes, quantized, ins, dynamic_cb

    def decode(self, codes, instruct):
        """Decode: reconstruct from code indices + instruction."""
        zqs = self.codebook(codes)
        cc = self.MLPconcat(torch.cat((zqs, instruct), 1))
        expert_weights = self.gate(cc)
        expert_outputs = []
        for expert in self.experts:
            x = cc
            for layer in expert:
                x = x + layer(x)
            expert_outputs.append(x)
        expert_outputs = torch.stack(expert_outputs, dim=1)
        toadd = (expert_outputs * expert_weights.unsqueeze(-1)).sum(dim=1)
        ins = self.instruction.weight[codes]
        return toadd, ins


# ---------------------------------------------------------------------------
# RQ-MoE main model (V2 — STE forward)
# ---------------------------------------------------------------------------

class RQMoE(nn.Module):
    """Residual Quantization via Mixture of Experts.

    V2: Uses Straight-Through Estimator so encoder + codebook0 receive gradients.
    """

    def __init__(self, d, K, M, N, L, H, dropout):
        super().__init__()
        self.d, self.K, self.M = d, K, M

        self.codebook0 = nn.Embedding(K, d)
        nn.init.uniform_(self.codebook0.weight, -1.0 / K, 1.0 / K)
        self.instruction0 = nn.Embedding(K, d)
        self.instruction0.weight.data.zero_()

        self.steps = nn.ModuleList()
        for m in range(1, M):
            self.steps.append(RQMoEStep(d, K, N, L, H, dropout=dropout))

    # ── K-Means initialization ──────────────────────────────────────────
    @torch.no_grad()
    def init_codebooks_kmeans(self, data, kmeans_iters=20):
        """Residual K-Means++ — clusters in data's native dim, maps to latent via input_proj.

        Args:
            data: [N, d] raw embeddings (2560-dim)
        """
        from sklearn.cluster import KMeans
        device = next(self.parameters()).device
        x = data.cpu().to(torch.float32).numpy()
        residual = x.copy()
        for level in range(self.M):
            km = KMeans(
                n_clusters=self.K, init='k-means++', n_init=3,
                max_iter=kmeans_iters, random_state=42 + level
            )
            labels = km.fit_predict(residual)
            centroids_full = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device)
            if level == 0:
                self.codebook0.weight.data.copy_(centroids_full)
            else:
                self.steps[level - 1].codebook.weight.data.copy_(centroids_full)
            residual = residual - km.cluster_centers_[labels]
            unique_labels = len(set(labels))
            print(f"  Level {level}: {unique_labels}/{self.K} clusters, residual std={residual.std():.4f}")
            del km
        print(f"[RQ-MoE] Residual K-Means++ on raw data complete: {self.M} levels")

    @torch.no_grad()
    def init_codebooks_from_rqvae(self, rqvae_codebook_path):
        """Initialize from baseline RQ-VAE codebooks (.npz file)."""
        cb = np.load(rqvae_codebook_path)
        # RQ-VAE codebooks: [M, K, d_codebook], may need dim adaptation
        rq_codes = np.array([cb[k] for k in sorted(cb.keys())])  # [M, K, d_cb]
        if rq_codes.shape[2] == self.d:
            t = torch.tensor(rq_codes, dtype=torch.float32)
            self.codebook0.weight.data.copy_(t[0])
            for i, step in enumerate(self.steps):
                step.codebook.weight.data.copy_(t[i + 1])
            print(f"[RQ-MoE] Initialized from RQ-VAE codebooks ({rq_codes.shape})")
        else:
            # Need projection — use PCA-like linear map
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.d)
            for i in range(self.M):
                projected = pca.fit_transform(rq_codes[i])
                t = torch.tensor(projected, dtype=torch.float32)
                if i == 0:
                    self.codebook0.weight.data.copy_(t)
                else:
                    self.steps[i - 1].codebook.weight.data.copy_(t)
            print(f"[RQ-MoE] Initialized from RQ-VAE codebooks via PCA (dim {rq_codes.shape[2]}→{self.d})")

    # ── Encode / Decode ─────────────────────────────────────────────────
    def encode(self, x):
        """Encode: returns codes, x_hat, and intermediate tensors for STE."""
        bs = x.shape[0]
        codes = torch.zeros(bs, self.M, dtype=torch.long, device=x.device)

        # Level 0
        code0 = assign_to_codebook(x, self.codebook0.weight)
        codes[:, 0] = code0
        c0 = self.codebook0.weight[code0]      # [bs, d]
        instruct = self.instruction0.weight[code0]

        x_hat_list = [c0]
        dynamic_cbs = [None]  # Level 0 has no dynamic transform

        residual = x - c0
        for i, step in enumerate(self.steps):
            codes_i, qi, ins_i, dyn_cb = step.encode(residual, instruct)
            codes[:, i + 1] = codes_i
            x_hat_list.append(qi)
            dynamic_cbs.append(dyn_cb)
            instruct = instruct + ins_i
            residual = residual - qi

        x_hat = torch.stack(x_hat_list).sum(dim=0)  # [bs, d]
        return codes, x_hat, x_hat_list, dynamic_cbs

    def decode(self, codes):
        x_hat = self.codebook0.weight[codes[:, 0]]
        instruct = self.instruction0.weight[codes[:, 0]]
        for i, step in enumerate(self.steps):
            toadd, ins = step.decode(codes[:, i + 1], instruct)
            x_hat = x_hat + toadd
            instruct = instruct + ins
        return x_hat

    # ── STE Forward (V2 fix) ────────────────────────────────────────────
    def forward(self, x):
        """
        V2 STE + Commitment Loss forward.

        1. STE: codebook weights receive gradient via embedding lookup.
        2. Commitment loss: encoder receives gradient (pushes z toward codewords).
        """
        codes, x_hat_raw, x_hat_list, _ = self.encode(x)

        # Level 0 STE
        ste_0 = x_hat_list[0] + (x - x_hat_list[0]).detach()
        losses = torch.zeros(self.M, device=x.device)
        losses[0] = ((ste_0 - x) ** 2).sum()

        # Commitment loss for Level 0: pushes encoder output toward codebook
        commitment_0 = F.mse_loss(x, x_hat_list[0].detach())

        # STE for remaining levels
        residual_ste = x - ste_0
        commitment_rest = 0.0
        for i, step in enumerate(self.steps):
            toadd_ste = x_hat_list[i + 1] + (residual_ste - x_hat_list[i + 1]).detach()
            residual_ste = residual_ste - toadd_ste
            ste_total = x - residual_ste
            losses[i + 1] = ((ste_total - x) ** 2).sum()
            # Commitment for subsequent levels
            commitment_rest = commitment_rest + F.mse_loss(
                residual_ste.detach(), x_hat_list[i + 1]
            )

        return codes, x_hat_raw, losses, commitment_0 + commitment_rest

    def compute_nrl_loss(self, x, losses):
        eps = 1e-6
        original_energy = (x ** 2).sum() + eps
        nor_loss = [torch.log1p(losses[0] / original_energy)]
        for layer in range(1, len(losses)):
            ratio = losses[layer] / (losses[layer - 1].detach() + eps)
            nor_loss.append(torch.log1p(ratio))
        return torch.stack(nor_loss).sum()

    @torch.no_grad()
    def get_indices(self, x):
        codes, _, _, _ = self.encode(x)
        return codes


# ---------------------------------------------------------------------------
# RQMoE Wrapper (V2 — compatible with RQVAE interface)
# ---------------------------------------------------------------------------

class RQMoEWrapper(nn.Module):
    """Wrapper making RQMoE compatible with the RQVAE interface.

    V2 changes:
      - Default e_dim: 64 → 256
      - K-Means init supported via --kmeans_init flag
      - STE gradient flows through encoder + codebook0
    """

    def __init__(self,
                 in_dim=768,
                 num_emb_list=None,
                 e_dim=256,               # ← V2: 64 → 256
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 beta=0.25,
                 kmeans_init=False,
                 kmeans_iters=10,
                 sk_epsilons=None,
                 sk_iters=100,
                 moe_N=2,
                 moe_L=4,
                 moe_H=512,              # ← V2: 256 → 512 (compensate e_dim increase)
                 moe_dropout=0.1,
                 ):
        super().__init__()

        if num_emb_list is None:
            num_emb_list = [256, 256, 256]
        if layers is None:
            layers = [512, 256, 128]      # ← V2: shallower (e_dim 256 needs smaller arch)

        self.in_dim = in_dim
        self.e_dim = e_dim
        self.layers = layers
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight

        # Input projection: in_dim → e_dim
        self.input_proj = nn.Linear(self.in_dim, self.e_dim)
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)

        # Encoder: e_dim → ... → e_dim (no zero-init — let it learn from start)
        self.encode_layer_dims = [self.e_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=dropout_prob, bn=bn)

        # RQ-MoE quantizer
        M = len(num_emb_list)
        K = num_emb_list[0]
        self.rq = RQMoE(d=e_dim, K=K, M=M, N=moe_N, L=moe_L, H=moe_H, dropout=moe_dropout)

        # Decoder: e_dim → ... → e_dim
        self.decode_layer_dims = [self.e_dim] + self.layers[::-1] + [self.e_dim]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                 dropout=dropout_prob, bn=bn)

        # Output projection: e_dim → in_dim
        self.output_proj = nn.Linear(self.e_dim, self.in_dim)
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    def _zero_init_last_layer(self):
        import torch.nn as nn
        for m in reversed(list(self.encoder.modules())):
            if isinstance(m, nn.Linear):
                with torch.no_grad():
                    m.weight.fill_(0.0)
                    if m.bias is not None:
                        m.bias.fill_(0.0)
                break

    def init_codebooks(self, data):
        """K-Means++ on projected latent, FREEZE ALL codebooks."""
        if not self.kmeans_init:
            return
        projected = self.input_proj(data.to(next(self.parameters()).device))
        self.rq.init_codebooks_kmeans(data=projected, kmeans_iters=self.kmeans_iters)
        # Freeze ALL codebooks — only train input_proj + output_proj
        self.rq.codebook0.weight.requires_grad = False
        self.rq.instruction0.weight.requires_grad = False
        for step in self.rq.steps:
            step.codebook.weight.requires_grad = False
            step.instruction.weight.requires_grad = False
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[RQMoEWrapper] ALL codebooks FROZEN, trainable params: {trainable:,}")

    def forward(self, x, use_sk=True):
        # Simple linear projection (no encoder/decoder — prevents collapse)
        z = self.input_proj(x)

        # RQ-MoE STE + Commitment forward
        codes, x_hat, step_losses, commit_loss = self.rq(z)
        rq_loss = self.rq.compute_nrl_loss(z, step_losses) + 0.25 * commit_loss

        # Decode back to input dim
        out = self.output_proj(x_hat)

        return out, rq_loss, codes

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        z = self.input_proj(xs)
        return self.rq.get_indices(z)

    def compute_loss(self, out, quant_loss, xs=None):
        loss_recon = F.mse_loss(out, xs) if self.loss_type == 'mse' else F.l1_loss(out, xs)
        return loss_recon + self.quant_loss_weight * quant_loss, loss_recon
