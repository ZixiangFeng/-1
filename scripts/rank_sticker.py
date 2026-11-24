"""Run the frozen two-stage selector on a public gallery or external test image."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sticker_pragmatics as intent
import sticker_ranker
from research.sticker_encoder.disentangled import identity_reason
from research.sticker_encoder.reviewed_gallery import meaning_vectors, reviewed_rows
from scripts.prepare_sticker_v4 import extract_families
from scripts.train_context_ranker import CORPUS, MATRICES, features, read

TEST = ROOT / "outputs/stickerqueries-100-20261119"
MANIFEST = TEST / "holdout/manifest.json"
GALLERY_STYLE = ROOT / "outputs/sticker-semantic-v3-local-20261118/all-style-features.npz"
CACHE = ROOT / "reproduction/public-phrase-vectors.npz"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def public_phrases(gallery, queries):
    phrases = [r["meaning"] for r in gallery if r.get("meaning_review")]
    return list(dict.fromkeys(phrases + [r["source_caption"] for r in queries]))


class FrozenPhrases:
    """BGE input features only. Labels, candidate IDs and rankings are not cached here."""
    def __init__(self, texts):
        with np.load(CACHE, allow_pickle=False) as data:
            if list(data["texts"]) != texts or str(data["manifest"]) != sha(MANIFEST):
                raise ValueError("Public phrases do not match the frozen catalog and test set")
            self.vectors = dict(zip(texts, data["vectors"]))

    def encode(self, texts):
        return [self.vectors[text] for text in texts]


def ordered_features(path, key, rows):
    with np.load(path, allow_pickle=False) as f:
        if list(f["ids"]) != [r["id"] for r in rows]:
            raise ValueError(f"Image IDs do not match {path.name}")
        return f[key]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", nargs="?", help="Public image ID, for example Z001 or B0161")
    parser.add_argument("--text", help="Replace the default use with a new Chinese utterance (requires local BGE)")
    parser.add_argument("--limit", type=int, default=5, help="Number displayed per stage; zero displays all")
    parser.add_argument("--verify-holdout", action="store_true", help="Recompute both stages for all 100 external queries")
    parser.add_argument("--build-cache", action="store_true", help="Re-encode the public phrases with the pinned local BGE")
    args = parser.parse_args()
    if args.limit < 0 or (args.text and (args.verify_holdout or not args.reference)):
        parser.error("--text needs one reference; --limit must be nonnegative")

    gallery = reviewed_rows(read(CORPUS)["rows"])
    queries = read(MANIFEST)["rows"]
    texts = public_phrases(gallery, queries)
    if args.build_cache:
        vectors = np.asarray(sticker_ranker.local_encoder().encode(texts), dtype=np.float32)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, texts=texts, vectors=vectors, manifest=sha(MANIFEST))
        return
    cached = FrozenPhrases(texts)
    metric = sticker_ranker.metric()
    language = intent.model()
    projected = metric.project(meaning_vectors(gallery, features(gallery), cached))
    inferred = language.groups(projected, [r["meaning"] for r in gallery])
    groups = [intent.family(r.get("reviewed_intent")) or g for r, g in zip(gallery, inferred)]
    gallery_index = {row["id"]: i for i, row in enumerate(gallery)}
    query_index = {row["id"]: i for i, row in enumerate(queries)}
    query_vectors = metric.project(cached.encode([r["source_caption"] for r in queries]))
    gallery_style = ordered_features(GALLERY_STYLE, "style", gallery).astype("float64")
    external_style = ordered_features(TEST / "holdout/style.npz", "style", queries).astype("float64")
    style_matrix = None

    def rank(identifier):
        nonlocal style_matrix
        if identifier in gallery_index:
            i = gallery_index[identifier]
            ref = gallery[i]
            text = ref["meaning"]
            vector, group = projected[i], groups[i] if ref["quality"] != "uncertain" else ""
            if style_matrix is None:
                style_matrix = ordered_features(MATRICES, "style", gallery)
            distance = style_matrix[i]
        elif identifier in query_index:
            i = query_index[identifier]
            row = queries[i]
            text = row["source_caption"]
            ref = dict(row, pack=row["source_pack"], families=extract_families(text))
            vector = query_vectors[i]
            group = language.groups(vector[None, :], [text])[0]
            distance = 1 - external_style[i] @ gallery_style.T
        else:
            parser.error("Unknown public image ID: " + identifier)
        if args.text:
            text = args.text
            vector = metric.project(sticker_ranker.local_encoder().encode([text]))[0]
            group = language.groups(vector[None, :], [text])[0]
        return intent.gallery_result(ref, gallery, groups, group, text, projected @ vector,
                                     distance, identity_reason, language.pair_floor)

    if args.verify_holdout:
        locked = read(TEST / "rankings.json")
        returned = 0
        for row in queries:
            identifier = row["id"]
            actual = rank(identifier)
            expected = locked[identifier]["8"]
            for stage in ("semantic", "divergent"):
                winner = next(iter(actual[stage]), None)
                if (winner and winner["id"]) != (expected[stage] and expected[stage]["id"]):
                    raise ValueError(f"{identifier}: {stage} differs from frozen prediction")
            returned += int(actual["maximum"] is not None)
        print(f"{len(queries)} queries matched the recorded first candidate in both stages; {returned} returned a divergent sticker.")
        return
    identifier = args.reference or "Z001"
    result = rank(identifier)
    end = args.limit or None
    def display(entry):
        return dict(entry, meaning=gallery[gallery_index[entry["id"]]]["meaning"])
    print(json.dumps({
        "reference": identifier, "intent": result["group_name"],
        "semantic_candidate_count": len(result["semantic"]), "style_candidate_count": len(result["divergent"]),
        "semantic_order": [display(v) for v in result["semantic"][:end]],
        "style_order": [display(v) for v in result["divergent"][:end]],
        "selected": display(result["maximum"]) if result["maximum"] else None,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
