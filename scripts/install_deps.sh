#!/bin/bash
# =============================================================================
# MiniOneRec 依赖安装脚本
#
# 在 MiniOneRec conda 环境中安装所有必需依赖。
# 自动检测 CUDA 版本并选择合适的 PyTorch 构建。
# 跳过代码中未实际使用的包（torchrec, fbgemm_gpu, pulsar-client等）。
#
# 用法:
#   conda activate MiniOneRec
#   bash scripts/install_deps.sh
# =============================================================================
set -e

CONDA_ENV="MiniOneRec"
PIP="conda run -n $CONDA_ENV pip"
PIP_INSTALL="$PIP install --no-cache-dir"

# ── 选择镜像 ──
MIRROR=""
# MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple"  # 国内加速

echo "============================================"
echo "  MiniOneRec Dependency Installation"
echo "  Environment: $CONDA_ENV"
echo "============================================"

# =========================================================================
# Phase 1: PyTorch (CUDA 12.x)
# =========================================================================
echo ""
echo "[1/5] Installing PyTorch 2.6.0 with CUDA 12.4 support..."
$PIP_INSTALL $MIRROR \
    torch==2.6.0 \
    --extra-index-url https://download.pytorch.org/whl/cu124

echo "  Verifying PyTorch..."
$PIP run python -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
print(f'  cuDNN: {torch.backends.cudnn.version()}')
"

# =========================================================================
# Phase 2: HuggingFace ecosystem
# =========================================================================
echo ""
echo "[2/5] Installing HuggingFace ecosystem..."
$PIP_INSTALL $MIRROR \
    "transformers==4.57.1" \
    "trl==0.24.0" \
    "accelerate==1.10.1" \
    "datasets==4.2.0" \
    "tokenizers==0.22.1" \
    "peft==0.19.1" \
    "huggingface-hub==0.35.3" \
    "safetensors==0.6.2"

# =========================================================================
# Phase 3: Training infrastructure
# =========================================================================
echo ""
echo "[3/5] Installing training infrastructure..."
$PIP_INSTALL $MIRROR \
    "deepspeed==0.18.0" \
    "bitsandbytes==0.48.1"

# =========================================================================
# Phase 4: Data processing & evaluation
# =========================================================================
echo ""
echo "[4/5] Installing data processing & evaluation libs..."
$PIP_INSTALL $MIRROR \
    "numpy==1.26.3" \
    "scipy==1.14.0" \
    "pandas==2.2.2" \
    "polars==1.41.2" \
    "scikit-learn==1.9.0" \
    "matplotlib" \
    "tqdm==4.67.1" \
    "POT==0.9.6.post1" \
    "k_means_constrained==0.9.0" \
    "faiss-cpu==1.14.3" \
    "fire==0.7.1"

# =========================================================================
# Phase 5: Monitoring & utilities
# =========================================================================
echo ""
echo "[5/5] Installing monitoring & utilities..."
$PIP_INSTALL $MIRROR \
    "tensorboard" \
    "wandb==0.22.2" \
    "nvitop" \
    "einops==0.8.0" \
    "sentencepiece" \
    "protobuf==3.19.6" \
    "ninja==1.11.1.1" \
    "packaging==24.1" \
    "pyarrow==21.0.0" \
    "regex==2025.9.18" \
    "PyYAML==6.0.1" \
    "Jinja2==3.1.3" \
    "jsonargparse==4.24.1" \
    "dill==0.4.0" \
    "multiprocess==0.70.16" \
    "xxhash==3.6.0"

# =========================================================================
# Verification
# =========================================================================
echo ""
echo "============================================"
echo "  Verifying installation..."
echo "============================================"

$PIP run python -c "
import sys
errors = []

# Core
try:
    import torch; print(f'[OK] torch {torch.__version__}')
except Exception as e: errors.append(f'torch: {e}')

try:
    import transformers; print(f'[OK] transformers {transformers.__version__}')
except Exception as e: errors.append(f'transformers: {e}')

try:
    import trl; print(f'[OK] trl {trl.__version__}')
except Exception as e: errors.append(f'trl: {e}')

try:
    import accelerate; print(f'[OK] accelerate {accelerate.__version__}')
except Exception as e: errors.append(f'accelerate: {e}')

try:
    import datasets; print(f'[OK] datasets {datasets.__version__}')
except Exception as e: errors.append(f'datasets: {e}')

# Training
try:
    import deepspeed; print(f'[OK] deepspeed')
except Exception as e: errors.append(f'deepspeed: {e}')

try:
    import bitsandbytes; print(f'[OK] bitsandbytes')
except Exception as e: errors.append(f'bitsandbytes: {e}')

try:
    import peft; print(f'[OK] peft')
except Exception as e: errors.append(f'peft: {e}')

# Data
try:
    import numpy; print(f'[OK] numpy {numpy.__version__}')
except Exception as e: errors.append(f'numpy: {e}')

try:
    import pandas; print(f'[OK] pandas {pandas.__version__}')
except Exception as e: errors.append(f'pandas: {e}')

try:
    import polars; print(f'[OK] polars {polars.__version__}')
except Exception as e: errors.append(f'polars: {e}')

try:
    import sklearn; print(f'[OK] scikit-learn {sklearn.__version__}')
except Exception as e: errors.append(f'sklearn: {e}')

try:
    from k_means_constrained import KMeansConstrained; print(f'[OK] k_means_constrained')
except Exception as e: errors.append(f'k_means_constrained: {e}')

try:
    import faiss; print(f'[OK] faiss')
except Exception as e: errors.append(f'faiss: {e}')

try:
    import POT; print(f'[OK] POT')
except Exception as e: errors.append(f'POT: {e}')

# Monitoring
try:
    from torch.utils.tensorboard import SummaryWriter; print(f'[OK] tensorboard')
except Exception as e: errors.append(f'tensorboard: {e}')

try:
    import wandb; print(f'[OK] wandb')
except Exception as e: errors.append(f'wandb: {e}')

# Utilities
try:
    import tqdm; print(f'[OK] tqdm {tqdm.__version__}')
except Exception as e: errors.append(f'tqdm: {e}')

try:
    import fire; print(f'[OK] fire')
except Exception as e: errors.append(f'fire: {e}')

try:
    import matplotlib; print(f'[OK] matplotlib')
except Exception as e: errors.append(f'matplotlib: {e}')

if errors:
    print(f'\n[ERROR] {len(errors)} package(s) failed:')
    for e in errors: print(f'  - {e}')
    sys.exit(1)
else:
    print(f'\n[SUCCESS] All packages verified!')
"

echo ""
echo "============================================"
echo "  Installation complete!"
echo "============================================"
echo ""
echo "To verify in this environment:"
echo "  conda activate $CONDA_ENV"
echo "  python -c 'import torch, transformers, trl; print(\"OK\")'"
