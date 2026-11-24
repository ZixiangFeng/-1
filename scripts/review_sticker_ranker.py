"""Freeze within-split comparison pools before fitting a new retrieval model."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'research/sticker_encoder/v6'
CORPUS = ROOT / 'outputs/sticker-broad-v5-20261119/display-dataset.json'
MATRICES = ROOT / 'outputs/sticker-divergence-v4-20261119/retrieval-matrices.npz'


def review_set(split):
    rows = json.loads(CORPUS.read_text())['rows']
    matrices = np.load(MATRICES)
    assert list(matrices['ids']) == [r['id'] for r in rows]
    meaning = matrices['meaning']
    style = matrices['style']
    legacy = .6 * meaning + .4 * matrices['v4']
    allowed = [i for i, r in enumerate(rows) if r['split'] == split and r['quality'] != 'uncertain']
    rng = np.random.default_rng(1849 if split == 'validation' else 1850)
    pools = []
    shuffled = rng.permutation(allowed)
    count = 20 if split == 'validation' else 32
    for q in shuffled:
        r = rows[q]
        candidates = [i for i in allowed if rows[i]['pack'] != r['pack'] and
                      not set(rows[i]['families']).intersection(r['families']) and
                      rows[i]['medium'] != r['medium'] and 'unknown' not in (rows[i]['medium'],r['medium'])
                      and style[q,i] >= .65]
        if len(candidates) < 12:
            continue
        # Include retrieval distractors, without consulting the new model. A few
        # low-similarity queries are deliberately included to evaluate abstention.
        if max(meaning[q,i] for i in candidates) < .52 and len(pools) % 5:
            continue
        text = sorted(candidates, key=lambda i: -meaning[q,i])
        old = sorted(candidates, key=lambda i: -legacy[q,i])
        far = sorted(candidates, key=lambda i: -style[q,i])
        ids = list(dict.fromkeys(text[:5] + old[:4] + far[:2] + text[5:]))[:10]
        rng.shuffle(ids)
        pools.append({'query':r['id'],'choices':[rows[i]['id'] for i in ids]})
        if len(pools) == count:
            break
    return pools


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('split', choices=['validation','test'])
    args = p.parse_args()
    items = {r['id']:r['meaning'] for r in json.loads(CORPUS.read_text())['rows']}
    for group in review_set(args.split):
        q = group['query']
        print(q,items[q])
        print(' | '.join(i+' '+items[i] for i in group['choices']))
