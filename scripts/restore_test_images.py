"""Restore the public 100-image test set using the pinned manifest and verified source bytes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import urllib.request

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
HOLDOUT = ROOT / "outputs/stickerqueries-100-20261119/holdout"
SOURCE = "https://huggingface.co/datasets/metchee/sticker-queries/resolve/c28a358226990ef183703bb3797c2b4c5b59e73c/stickers/"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def restore(row):
    path = HOLDOUT / "images" / (row["id"] + ".png")
    if path.is_file():
        if digest(path.read_bytes()) != row["image_sha256"]:
            raise ValueError(f"Existing image has an unexpected digest: {path.name}")
        return
    if not row["url"].startswith(SOURCE):
        raise ValueError("Unexpected test dataset URL")
    with urllib.request.urlopen(row["url"], timeout=35) as response:
        raw = response.read()
    if digest(raw) != row["original_sha256"]:
        raise ValueError(f"Upstream image differs: {path.name}")
    with Image.open(io.BytesIO(raw)) as image:
        rgba = ImageOps.exif_transpose(image).convert("RGBA")
        base = Image.new("RGBA", rgba.size, "white")
        base.alpha_composite(rgba)
        rgb = base.convert("RGB")
    if digest(str(rgb.size).encode() + rgb.tobytes()) != row["pixel_hash"]:
        raise ValueError(f"Decoded pixels differ: {path.name}")
    rgb.thumbnail((640, 640))
    encoded = io.BytesIO()
    rgb.save(encoded, format="PNG")
    content = encoded.getvalue()
    if digest(content) != row["image_sha256"]:
        raise ValueError(f"PNG encoding differs from the original study environment: {path.name}")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as out:
        out.write(content)
    os.replace(out.name, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ids", nargs="*", help="Optional IDs such as Z001; defaults to the whole set")
    args = parser.parse_args()
    rows = json.loads((HOLDOUT / "manifest.json").read_text())["rows"]
    if set(args.ids) - {r["id"] for r in rows}:
        parser.error("Unknown test ID")
    selected = [r for r in rows if not args.ids or r["id"] in args.ids]
    (HOLDOUT / "images").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(restore, selected))
    print(f"Verified {len(selected)} public test images")
