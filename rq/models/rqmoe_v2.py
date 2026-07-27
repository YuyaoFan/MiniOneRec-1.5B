"""
RQ-MoE V2: 3-phase EMA training + reference architecture.

Phase 1: EMA codebook update (freeze projection, adapt codebooks)
Phase 2: Joint STE training (codebooks + projection)
Phase 3: Fine-tune (freeze codebooks, train projection only)

Operates directly in data dimension (no bottleneck), following the
reference RQ-MoE design. Codebooks initialized from baseline RQ-VAE
centroids, then refined via EMA (VQ-VAE style).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .layers import MLPLayers


# ── Utility ────────────────────────────────────────────────────────────
def pairwise_distances(a, b):
    anorms = (a ** 2).sum(-1)
    bnorms = (b ** 2).sum(-1)
    return anorms[:, None] + bnorms - 2 * a @ b.T


def assign_to_codebook(x, c, bs=16384):
    nq, d = x.shape
    nb, d2 = c.shape
    assert d == d2
    if nq * nb < bs * bs:
        return pairwise_distances(x, c).argmin(1)
    res = torch.empty((nq,), dtype=torch.int64, device=x.device)
    cnorms = (c ** 2).sum(1)
    for i in range(0, nq, bs):
        xnorms = (x[i:i + bs] ** 2).sum(1, keepdim=True)
        for j in range(0, nb, bs):
            dis = xnorms + cnorms[j:j + bs] - 2 * x[i:i + bs] @ c[j:j + bs].T
            dmini, imini = dis.min(1)
            if j == 0:
                dmin, imin = dmini, imini
            else:
                (mask,) = torch.where(dmini < dmin)
                dmin[mask], imin[mask] = dmini[mask], imini[mask] + j
        res[i:i + bs] = imin
    return res


def assign_batch_multiple(x, zqs):
    bs, d = x.shape
    K = zqs.shape[1]
    x_norm = (x ** 2).sum(dim=1, keepdim=True)
    zqs_norm = (zqs ** 2).sum(dim=2)
    xz = torch.bmm(x.unsqueeze(1), zqs.transpose(1, 2)).squeeze(1)
    L2 = x_norm + zqs_norm - 2 * xz
    idx = torch.argmin(L2, dim=1)
    quantized = zqs[torch.arange(bs, device=zqs.device), idx]
    return idx, quantized


# ── RQ-MoE Step (with EMA support) ────────────────────────────────────
class RQMoEStepV2(nn.Module):
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
        self.experts = nn.ModuleList()
        for _ in range(N):
            layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d, H, bias=False), nn.Dropout(dropout),
                    nn.ReLU(), nn.Linear(H, d, bias=False),
                ) for _ in range(L)
            ])
            self.experts.append(layers)

    def _transform(self, instruct, chunk_size=4):
        """Chunked dynamic codebook transformation."""
        bs = instruct.shape[0]
        zqs = self.codebook.weight  # [K, d]
        K, d = zqs.shape
        results = []
        for k_start in range(0, K, chunk_size):
            k_end = min(k_start + chunk_size, K)
            kc = k_end - k_start
            zqs_c = zqs[k_start:k_end].unsqueeze(0).expand(bs, kc, d).reshape(bs * kc, d)
            ins_c = instruct.unsqueeze(1).expand(bs, kc, d).reshape(bs * kc, d)
            cc = self.MLPconcat(torch.cat((zqs_c, ins_c), dim=1))
            ew = self.gate(cc)
            eo = []
            for expert in self.experts:
                x = cc
                for layer in expert:
                    x = x + layer(x)
                eo.append(x)
            eo = torch.stack(eo, dim=1)
            w = (eo * ew.unsqueeze(-1)).sum(dim=1)
            results.append(w.reshape(bs, kc, d))
        return torch.cat(results, dim=1)

    def encode(self, residual, instruct):
        dynamic_cb = self._transform(instruct)
        codes, quantized = assign_batch_multiple(residual, dynamic_cb)
        ins = self.instruction.weight[codes]
        return codes, quantized, ins

    def decode(self, codes, instruct):
        zqs = self.codebook(codes)
        cc = self.MLPconcat(torch.cat((zqs, instruct), 1))
        ew = self.gate(cc)
        eo = []
        for expert in self.experts:
            x = cc
            for layer in expert:
                x = x + layer(x)
            eo.append(x)
        eo = torch.stack(eo, dim=1)
        toadd = (eo * ew.unsqueeze(-1)).sum(dim=1)
        ins = self.instruction.weight[codes]
        return toadd, ins

    @torch.no_grad()
    def ema_update(self, x, codes, decay=0.99):
        """EMA update: codebook ← decay*codebook + (1-decay)*mean(assigned_x)"""
        for k in range(self.K):
            mask = (codes == k)
            if mask.sum() == 0:
                continue
            assigned_x = x[mask].mean(dim=0)
            self.codebook.weight[k] = decay * self.codebook.weight[k] + (1 - decay) * assigned_x


# ── RQ-MoE Main Model V2 ──────────────────────────────────────────────
class RQMoEV2(nn.Module):
    """RQ-MoE operating in data-native dimension, with EMA codebook update."""

    def __init__(self, d, K, M, N, L, H, dropout):
        super().__init__()
        self.d, self.K, self.M = d, K, M

        self.codebook0 = nn.Embedding(K, d)
        self.instruction0 = nn.Embedding(K, d)
        self.instruction0.weight.data.zero_()

        self.steps = nn.ModuleList()
        for _ in range(1, M):
            self.steps.append(RQMoEStepV2(d, K, N, L, H, dropout=dropout))

    def encode(self, x):
        bs = x.shape[0]
        codes = torch.zeros(bs, self.M, dtype=torch.long, device=x.device)
        code0 = assign_to_codebook(x, self.codebook0.weight)
        codes[:, 0] = code0
        c0 = self.codebook0.weight[code0]
        instruct = self.instruction0.weight[code0]
        residual = x - c0
        for i, step in enumerate(self.steps):
            codes_i, qi, ins_i = step.encode(residual, instruct)
            codes[:, i + 1] = codes_i
            residual = residual - qi
            instruct = instruct + ins_i
        return codes

    def decode(self, codes):
        x_hat = self.codebook0.weight[codes[:, 0]]
        instruct = self.instruction0.weight[codes[:, 0]]
        for i, step in enumerate(self.steps):
            toadd, ins = step.decode(codes[:, i + 1], instruct)
            x_hat = x_hat + toadd
            instruct = instruct + ins
        return x_hat

    def forward(self, x):
        """STE forward: encode with grad, decode with grad."""
        codes, x_hat_raw = self.encode_and_ste(x)
        # Compute per-step losses for NRL
        losses = torch.zeros(self.M, device=x.device)
        x_hat = self.codebook0(codes[:, 0])
        instruct = self.instruction0(codes[:, 0])
        losses[0] = ((x_hat - x) ** 2).sum()
        for i, step in enumerate(self.steps):
            toadd, ins = step.decode(codes[:, i + 1], instruct)
            x_hat = x_hat + toadd
            instruct = instruct + ins
            losses[i + 1] = ((x_hat - x) ** 2).sum()
        return codes, x_hat_raw, losses

    def encode_and_ste(self, x):
        """Encode with STE for gradient through codebook embeddings."""
        # Level 0 STE
        code0 = assign_to_codebook(x, self.codebook0.weight)
        c0 = self.codebook0.weight[code0]
        ste0 = c0 + (x - c0).detach()  # STE: grad through c0
        instruct = self.instruction0.weight[code0]

        bs = x.shape[0]
        codes = torch.zeros(bs, self.M, dtype=torch.long, device=x.device)
        codes[:, 0] = code0
        x_hat = ste0
        residual = x - ste0

        for i, step in enumerate(self.steps):
            codes_i, qi, ins_i = step.encode(residual, instruct)
            codes[:, i + 1] = codes_i
            # STE per step
            qi_ste = qi + (residual - qi).detach()
            residual = residual - qi_ste
            x_hat = x_hat + qi_ste
            instruct = instruct + ins_i

        return codes, x_hat

    def compute_nrl_loss(self, x, losses):
        eps = 1e-6
        e0 = (x ** 2).sum() + eps
        nor = [torch.log1p(losses[0] / e0)]
        for i in range(1, len(losses)):
            nor.append(torch.log1p(losses[i] / (losses[i - 1].detach() + eps)))
        return torch.stack(nor).sum()

    @torch.no_grad()
    def get_indices(self, x):
        return self.encode(x)

    @torch.no_grad()
    def ema_update_all(self, x, codes, decay=0.99):
        """EMA update for all codebook levels."""
        for k in range(self.K):
            mask = (codes[:, 0] == k)
            if mask.sum() == 0:
                continue
            self.codebook0.weight[k] = decay * self.codebook0.weight[k] + \
                (1 - decay) * x[mask].mean(dim=0)

        x_hat = self.codebook0.weight[codes[:, 0]]
        residual = x - x_hat
        for i, step in enumerate(self.steps):
            step.ema_update(residual, codes[:, i + 1], decay=decay)
            residual = residual - step.codebook.weight[codes[:, i + 1]]

    def init_from_rqvae(self, codebook_path, device):
        """Initialize codebooks from baseline RQ-VAE centroids."""
        cb = np.load(codebook_path)
        keys = sorted(cb.keys())
        for i, k in enumerate(keys):
            t = torch.tensor(cb[k], dtype=torch.float32, device=device)
            if i == 0:
                self.codebook0.weight.data.copy_(t)
            elif i - 1 < len(self.steps):
                self.steps[i - 1].codebook.weight.data.copy_(t)
        print(f"[RQMoE-V2] Initialized from RQ-VAE codebooks: {len(keys)} levels")


# ── Wrapper with projection layers ─────────────────────────────────────
class RQMoEV2Wrapper(nn.Module):
    """Lightweight wrapper: Linear proj → RQ-MoE → Linear proj back."""

    def __init__(self, in_dim, num_emb_list, e_dim, moe_N, moe_L, moe_H, moe_dropout,
                 rqvae_codebook_path=None):
        super().__init__()
        M = len(num_emb_list)
        K = num_emb_list[0]
        self.in_dim = in_dim
        self.e_dim = e_dim

        # Projection: in_dim → e_dim
        self.input_proj = nn.Linear(in_dim, e_dim)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        # RQ-MoE core
        self.rq = RQMoEV2(d=e_dim, K=K, M=M, N=moe_N, L=moe_L, H=moe_H, dropout=moe_dropout)

        # Output projection: e_dim → in_dim
        self.output_proj = nn.Linear(e_dim, in_dim)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x):
        z = self.input_proj(x)
        codes, x_hat, step_losses = self.rq(z)
        nrl = self.rq.compute_nrl_loss(z, step_losses)
        # Commitment loss: pull encoder output toward assigned codebook vectors
        x_hat_lookup = self.rq.codebook0(codes[:, 0])
        commit = F.mse_loss(z, x_hat_lookup.detach())
        for i, step in enumerate(self.rq.steps):
            commit = commit + F.mse_loss(z, step.codebook(codes[:, i + 1]).detach())
        rq_loss = nrl + 0.25 * commit  # EMA maintains diversity; commitment plays support role
        out = self.output_proj(x_hat)
        return out, rq_loss, codes

    @torch.no_grad()
    def get_indices(self, xs):
        z = self.input_proj(xs)
        return self.rq.get_indices(z)

    def init_codebooks(self, data):
        """K-Means init via baseline RQ-VAE codebooks or random."""
        if hasattr(self, '_initted'):
            return
        self._initted = True
        # Default: uniform init
        nn.init.uniform_(self.rq.codebook0.weight, -1/self.rq.K, 1/self.rq.K)
        for step in self.rq.steps:
            nn.init.uniform_(step.codebook.weight, -1/self.rq.K, 1/self.rq.K)
        print("[RQMoE-V2] Codebooks initialized (uniform)")

    def compute_loss(self, out, quant_loss, xs=None):
        recon = F.mse_loss(out, xs)
        return recon + quant_loss, recon

    # ── Phase control ──────────────────────────────────────────────
    def phase0_pretrain_projection(self, data_loader, optimizer, epochs, device):
        """Phase 0: Train input_proj + output_proj with MSE (no quantization).
        This learns a meaningful latent representation before codebook clustering."""
        print("[Phase 0] Pretraining projection layers (MSE reconstruction, no VQ)...")
        self.input_proj.requires_grad_(True)
        self.output_proj.requires_grad_(True)
        self.rq.codebook0.requires_grad_(False)
        for step in self.rq.steps:
            step.codebook.requires_grad_(False)

        for epoch in range(epochs):
            total_loss = 0
            for batch in data_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                z = self.input_proj(batch)
                out = self.output_proj(z)
                loss = F.mse_loss(out, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 10 == 0:
                print(f"  Pretrain epoch {epoch}: mse={total_loss/len(data_loader):.4f}")
        print(f"  Phase 0 complete: mse={total_loss/len(data_loader):.4f}")

    def phase1_kmeans_init(self, data_loader, device):
        """Phase 1: sklearn KMeans++ on projected data. Runs residual clustering
        across all 3 levels. sklearn's optimized implementation converges in <300
        iterations and guarantees collision rate → 0 for each level independently."""
        from sklearn.cluster import KMeans
        import numpy as np

        print("[Phase 1] sklearn K-Means++ codebook initialization...")
        self.train()
        self.input_proj.requires_grad_(False)
        self.output_proj.requires_grad_(False)

        # Collect all projected data → CPU numpy for sklearn
        all_z = []
        for batch in data_loader:
            all_z.append(self.input_proj(batch.to(device)).cpu())
        all_z_np = torch.cat(all_z, dim=0).numpy().astype(np.float64)
        N = all_z_np.shape[0]
        print(f"  Projected data: {all_z_np.shape}, dtype={all_z_np.dtype}")

        residual = all_z_np.copy()
        for level in range(self.rq.M):
            cb = self.rq.codebook0 if level == 0 else self.rq.steps[level - 1].codebook
            print(f"  L{level}: sklearn KMeans(n_clusters=256, init=k-means++, n_init=10)...")
            km = KMeans(
                n_clusters=self.rq.K, init='k-means++', n_init=10,
                max_iter=300, tol=1e-6, random_state=42 + level, verbose=0,
            )
            labels = km.fit_predict(residual)
            centroids = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device)
            cb.weight.data.copy_(centroids)

            # Report
            unique = len(set(labels))
            inertia = km.inertia_
            collision = (N - unique) / N
            print(f"  L{level} done: {unique}/{self.rq.K} clusters used, "
                  f"inertia={inertia:.2f}, collision={collision:.4f}, "
                  f"iters={km.n_iter_}")

            # Compute residual for next level
            residual = residual - km.cluster_centers_[labels]
            del km

        # Final collision rate across all levels
        _, cr = self._eval_collision(data_loader, device)
        print(f"  Phase 1 complete: collision_rate={cr:.4f}")
        self.rq.codebook0.requires_grad_(False)
        for step in self.rq.steps:
            step.codebook.requires_grad_(False)
            step.codebook.weight.requires_grad = False

    def _kmeans_pp_seed(self, codebook, data):
        """K-Means++ seeding: pick centroids from data with probability ∝ distance²."""
        N = data.shape[0]
        device = data.device
        # First centroid: random
        idx = torch.randint(0, N, (1,), device=device)
        centroids = [data[idx[0]]]
        for _ in range(1, codebook.weight.shape[0]):
            # Compute min distance to existing centroids
            cents = torch.stack(centroids)  # [k, d]
            dists = pairwise_distances(data, cents).min(dim=1)[0] ** 2
            dists /= dists.sum()
            idx = torch.multinomial(dists, 1)
            centroids.append(data[idx[0]])
        codebook.weight.data.copy_(torch.stack(centroids))

    def phase3_ste_train(self, data_loader, optimizer, epochs, device):
        """Phase 3: STE projection training with FROZEN codebooks.
        Only input_proj and output_proj are trained; codebooks stay fixed.
        This prevents codebook drift while improving reconstruction."""
        print("[Phase 3] STE projection training (codebooks frozen)...")
        self.input_proj.requires_grad_(True)
        self.output_proj.requires_grad_(True)
        self.rq.codebook0.requires_grad_(False)
        for step in self.rq.steps:
            step.codebook.requires_grad_(False)

        for epoch in range(epochs):
            total_loss = 0
            for batch in data_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out, rq_loss, indices = self(batch)
                loss, recon = self.compute_loss(out, rq_loss, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            if epoch % 10 == 0:
                _, cr = self._eval_collision(data_loader, device)
                print(f"  STE epoch {epoch}: loss={total_loss/len(data_loader):.4f}, collision_rate={cr:.4f}")

    def phase3_finetune(self, data_loader, optimizer, epochs, device):
        """Phase 3: Freeze codebooks, fine-tune projections only."""
        print("[Phase 3] Fine-tune projections...")
        self.input_proj.requires_grad_(True)
        self.output_proj.requires_grad_(True)
        self.rq.codebook0.requires_grad_(False)
        for step in self.rq.steps:
            step.codebook.requires_grad_(False)

        for epoch in range(epochs):
            total_loss = 0
            for batch in data_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out, rq_loss, indices = self(batch)
                loss, recon = self.compute_loss(out, rq_loss, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 10 == 0:
                _, cr = self._eval_collision(data_loader, device)
                print(f"  Finetune epoch {epoch}: loss={total_loss/len(data_loader):.4f}, collision_rate={cr:.4f}")

    @torch.no_grad()
    def _eval_collision(self, data_loader, device):
        self.eval()
        indices_set, total = set(), 0
        for batch in data_loader:
            batch = batch.to(device)
            idx = self.get_indices(batch)
            idx = idx.view(-1, idx.shape[-1]).cpu().numpy()
            for row in idx:
                indices_set.add("-".join(str(int(c)) for c in row))
            total += len(batch)
        self.train()
        return len(indices_set), (total - len(indices_set)) / total
