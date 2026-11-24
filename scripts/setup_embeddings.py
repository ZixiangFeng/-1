"""Download the pinned embedding snapshot. Runtime uses only local files."""

import os, sys
from pathlib import Path

os.environ["HF_HUB_DISABLE_XET"] = "1"
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from huggingface_hub import snapshot_download
from rag import EMBED_REVISION

snapshot_download(
    "Qdrant/bge-small-zh-v1.5",
    revision=EMBED_REVISION,
    cache_dir=str(root / "models/embeddings"),
    allow_patterns=[
        "config.json",
        "model_optimized.onnx",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ],
)
print("Chinese embedding snapshot installed")
