"""Run both original training stages in a new directory, preserving the frozen release."""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "scripts/train_context_ranker.py", "scripts/review_sticker_ranker.py",
    "scripts/train_pragmatic_intent.py", "sticker_ranker.py", "sticker_pragmatics.py",
    "rag.py", "companion.py", "support_skills.py",
)
INPUTS = (
    "outputs/sticker-broad-v5-20261119/display-dataset.json",
    "outputs/sticker-divergence-v4-20261119/all-caption-features.npz",
    "outputs/sticker-divergence-v4-20261119/retrieval-matrices.npz",
    "outputs/sticker-codex-local-20261118/pair-review.json",
    "outputs/sticker-context-v6/training-phrases.npz",
    "outputs/sticker-pragmatic-v7/holdout/manifest.json",
    "research/sticker_encoder/v6/dialogues.json",
    "research/sticker_encoder/v6/reviewed-training.json",
    "research/sticker_encoder/v6/validation-judgments.json",
    "research/sticker_encoder/v7/training-dialogues.tsv",
    "research/sticker_encoder/v7/development-dialogues.tsv",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="New output directory for an independent fit")
    dest = parser.parse_args().directory.resolve()
    sys.path.insert(0, str(ROOT))
    from rag import EMBED_REVISION
    snapshot = Path("models/embeddings/models--Qdrant--bge-small-zh-v1.5/snapshots") / EMBED_REVISION
    if not (ROOT / snapshot / "model_optimized.onnx").is_file():
        parser.error("Install the pinned BGE first with python scripts/setup_embeddings.py")
    if dest.exists():
        parser.error("Choose a new directory for this training run")
    dest.mkdir(parents=True)
    for name in (*SCRIPTS, *INPUTS):
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    shutil.copytree(ROOT / snapshot, dest / snapshot)
    subprocess.run([sys.executable, "scripts/train_context_ranker.py"], cwd=dest, check=True)
    shutil.copy2(dest / "outputs/sticker-context-v6/semantic-ranker-v6.npz",
                 dest / "research/sticker_encoder/v6/semantic-ranker-v6.npz")
    subprocess.run([sys.executable, "scripts/train_pragmatic_intent.py"], cwd=dest, check=True)
    print(f"Two fitted checkpoints and their training reports are in {dest}")


if __name__ == "__main__":
    main()
