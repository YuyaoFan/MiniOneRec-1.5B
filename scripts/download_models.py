#!/usr/bin/env python
"""
Download all pretrained models needed by MiniOneRec.

Models:
  1. Qwen2.5-1.5B-Instruct → models/Qwen2.5-1.5B/  (SFT + RL backbone)

Uses HF_ENDPOINT=https://hf-mirror.com for faster download in China.
CPU-only safe - just downloads files, no GPU needed.
"""

import os
import sys

# Use HF mirror for faster download
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


def download_qwen_model(model_id: str, local_dir: str):
    """Download a model from HuggingFace to local directory."""
    print(f"\n{'='*60}")
    print(f"Downloading: {model_id}")
    print(f"To: {local_dir}")
    print(f"{'='*60}")

    # Download only essential files (skip pytorch .bin if .safetensors available)
    ignore_patterns = [
        "*.msgpack", "*.h5", "*.ot", "*.pdf",
        "flax_model.*", "tf_model.*", "model.safetensors.index.json.bak",
    ]

    try:
        path = snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            ignore_patterns=ignore_patterns,
            resume_download=True,
            max_workers=4,
        )
        print(f"[OK] Downloaded to: {path}")
    except Exception as e:
        print(f"[ERROR] Failed to download {model_id}: {e}")
        # Try with regular HF endpoint
        print("Retrying with default HF endpoint...")
        os.environ.pop("HF_ENDPOINT", None)
        try:
            path = snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                ignore_patterns=ignore_patterns,
                resume_download=True,
                max_workers=4,
            )
            print(f"[OK] Downloaded to: {path}")
        except Exception as e2:
            print(f"[FATAL] Failed to download {model_id}: {e2}")
            sys.exit(1)

    # Verify with tokenizer loading
    try:
        tok = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
        print(f"[OK] Tokenizer loaded. Vocab size: {len(tok)}")
    except Exception as e:
        print(f"[WARN] Tokenizer verification failed (may be normal during download): {e}")

    return path


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(root, "models")
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 60)
    print("  MiniOneRec Model Download")
    print("=" * 60)

    # ── Model 1: Qwen2.5-1.5B-Instruct (SFT + RL backbone) ──
    download_qwen_model(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        local_dir=os.path.join(models_dir, "Qwen2.5-1.5B"),
    )

    # ── Summary ──
    print(f"\n{'='*60}")
    print("  All models downloaded!")
    print(f"  Location: {models_dir}/")
    print(f"{'='*60}")

    # List downloaded files
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(models_dir):
        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            if os.path.isfile(fp):
                size = os.path.getsize(fp)
                total_size += size
                if size > 1024 * 1024:  # > 1MB
                    print(f"  {os.path.relpath(fp, models_dir)} ({size/1024/1024:.1f} MB)")

    print(f"\n  Total: {total_size/1024/1024/1024:.1f} GB")


if __name__ == "__main__":
    main()
