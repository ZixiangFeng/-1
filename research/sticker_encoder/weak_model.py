"""Our Chinese sticker encoder experiment. Automatic captions are weak labels.

This module intentionally does not relax semantic_training.validate_dataset's
human-review requirements or mark machine labels as reviewed.
"""
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

IMPLEMENTATION = 'xiaoyu-chinese-clip-weak-v1'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def rgb(path):
    # Preserve the full image, including corner text. Never flip or crop text.
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert('RGBA')
        bg = Image.new('RGBA', im.size, 'white')
        bg.alpha_composite(im)
        return ImageOps.pad(bg.convert('RGB'), (224, 224), method=Image.Resampling.BICUBIC, color='white')


def split_groups(rows, linked_pairs=(), seed=20261111):
    """Union source packs and near-duplicates BEFORE assigning any split."""
    parent = {r['id']: r['id'] for r in rows}
    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        parent[root(a)] = root(b)
    packs, hashes = {}, {}
    for r in rows:
        for seen, key in [(packs, r['pack']), (hashes, r['pixel_hash'])]:
            if key in seen:
                union(r['id'], seen[key])
            seen[key] = r['id']
    for a, b in linked_pairs:
        union(a, b)
    groups = {}
    for r in rows:
        groups.setdefault(root(r['id']), []).append(r['id'])
    groups = list(groups.values())
    rng = random.Random(seed)
    rng.shuffle(groups)
    # Largest groups first; seeded order breaks equal-size ties.
    groups.sort(key=len, reverse=True)
    target = {'train': .70 * len(rows), 'validation': .15 * len(rows), 'test': .15 * len(rows)}
    result = {k: [] for k in target}
    for group in groups:
        split = max(target, key=lambda k: target[k] - len(result[k]))
        result[split].extend(group)
    if any(not values for values in result.values()):
        raise ValueError('Too few independent groups for three splits')
    return {k: sorted(v) for k, v in result.items()}


def contrastive_loss(images, texts, caption_ids, teacher_similarity, scale):
    """Symmetric CLIP loss; identical captions are positive, likely synonyms masked.

    BGE > .80 only masks possible false negatives. It never creates positive
    labels. Model/description mistakes are still possible and remain weak data.
    """
    logits = scale * images @ texts.T
    ids = torch.as_tensor(caption_ids, device=images.device)
    positive = ids[:, None] == ids[None, :]
    valid = (teacher_similarity <= .80) | positive
    def one(x, p, v):
        masked = x.masked_fill(~v, -1e4)
        log_prob = F.log_softmax(masked, dim=1)
        return -(log_prob * p).sum(1).div(p.sum(1)).mean()
    return (one(logits, positive, valid) + one(logits.T, positive.T, valid.T)) / 2


def retrieval(images, texts):
    """Paired-caption retrieval, NOT a human-judged semantic accuracy metric."""
    sim = np.asarray(images) @ np.asarray(texts).T
    n = len(sim)
    def direction(matrix):
        ranks = np.argsort(np.argsort(-matrix, axis=1), axis=1)[np.arange(n), np.arange(n)] + 1
        return {'n': n, 'recall_at_1': float(np.mean(ranks <= 1)),
                'recall_at_5': float(np.mean(ranks <= min(5, n))),
                'recall_at_10': float(np.mean(ranks <= min(10, n))),
                'mrr': float(np.mean(1 / ranks)), 'median_rank': float(np.median(ranks))}
    return {'image_to_caption': direction(sim), 'caption_to_image': direction(sim.T)}
