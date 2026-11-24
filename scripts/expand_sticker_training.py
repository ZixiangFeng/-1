"""Expand the public weak-caption corpus without reassigning v1 holdout packs.

No participant files are read. Downloads are checked against pinned Git blobs.
Only the new public images are sent to the existing vision annotation API.
"""
import argparse
import base64
from collections import Counter, defaultdict
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
import sys
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
from PIL import Image, ImageOps
import sticker_identity
from research.sticker_encoder.weak_model import digest, save


def collect(old_run, source, out, targets, seed):
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        return
    old = json.loads((old_run / 'dataset.json').read_text())
    tree = json.loads((source / 'source-tree.json').read_text())
    rows = [dict(r) for r in old['rows']]
    packs = {r['pack']: r['split'] for r in rows}
    used = {r['path'] for r in rows}
    for folder in ('images', 'originals', 'annotations'):
        (out / folder).mkdir(exist_ok=True)
    for r in rows:
        shutil.copy2(source / r['image'], out / r['image'])
        shutil.copy2(source / 'annotations' / (r['id'] + '.json'), out / 'annotations' / (r['id'] + '.json'))
    fingerprints = [sticker_identity.fingerprint(base64.b64encode((out / r['image']).read_bytes()).decode()) for r in rows]
    candidates = defaultdict(list)
    for r in tree['images']:
        pack = r['path'].split('/')[0]
        if r['path'] not in used and pack in packs:
            candidates[pack].append(r)
    rng = random.Random(seed)
    for pack in sorted(candidates):
        candidates[pack].sort(key=lambda r: r['path'])
        rng.shuffle(candidates[pack])
    ordered = []
    while any(candidates.values()):
        for pack in sorted(candidates):
            if candidates[pack]:
                ordered.append(candidates[pack].pop())
    counts = Counter()
    rejected = []

    def get(r):
        url = 'https://raw.githubusercontent.com/zhaoolee/ChineseBQB/' + tree['commit'] + '/' + urllib.parse.quote(r['path'])
        path = out / 'originals' / (r['sha'] + Path(r['path']).suffix.lower())
        for attempt in range(3):
            try:
                if path.exists():
                    raw = path.read_bytes()
                else:
                    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Xiaoyu-public-training/2.0'}), timeout=25) as resp:
                        raw = resp.read(2_000_001)
                if len(raw) > 2_000_000 or hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() != r['sha']:
                    raise ValueError('Pinned Git blob verification failed')
                path.write_bytes(raw)
                with Image.open(io.BytesIO(raw)) as im:
                    if im.width * im.height > 16_000_000 or min(im.size) < 32:
                        raise ValueError('Image dimensions outside bounds')
                    im = ImageOps.exif_transpose(im).convert('RGBA')
                    bg = Image.new('RGBA', im.size, 'white'); bg.alpha_composite(im)
                    image = bg.convert('RGB')
                    ph = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
                    dimensions = image.size
                    image.thumbnail((768, 768))
                    buffer = io.BytesIO(); image.save(buffer, 'PNG', optimize=True)
                return dict(r, raw=buffer.getvalue(), pixel_hash=ph, original_dimensions=dimensions,
                            file_sha256=hashlib.sha256(raw).hexdigest(), url=url,
                            original=str(path.relative_to(out)), pack=r['path'].split('/')[0])
            except Exception as exc:
                if attempt == 2:
                    return {'path': r['path'], 'error': type(exc).__name__ + ': ' + str(exc)[:160]}
                time.sleep(1 + attempt)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0, len(ordered), 40):
            batch = [r for r in ordered[start:start+40] if counts[packs[r['path'].split('/')[0]]] < targets[packs[r['path'].split('/')[0]]]]
            for r in pool.map(get, batch):
                if 'error' in r:
                    rejected.append(r); continue
                split = packs[r['pack']]
                if counts[split] >= targets[split]:
                    continue
                fp = sticker_identity.fingerprint(base64.b64encode(r['raw']).decode())
                coarse = np.array([list(f['coarse']) for f in fingerprints], dtype=np.float32)
                nearby = np.flatnonzero(np.mean(np.abs(coarse - np.array(list(fp['coarse']))), axis=1) <= 8)
                duplicate = next((rows[i]['id'] for i in nearby if sticker_identity.same_image(fp, fingerprints[i])), None)
                if duplicate:
                    rejected.append({'path': r['path'], 'reason': 'duplicate_or_near_duplicate', 'of': duplicate}); continue
                aid = 'E%04d' % (sum(counts.values()) + 1)
                image_path = 'images/' + aid + '.png'
                (out / image_path).write_bytes(r.pop('raw'))
                r.update(id=aid, image=image_path, split=split, label_source='doubao_visual_auto', human_reviewed=False)
                rows.append(r); fingerprints.append(fp); counts[split] += 1
            save(out / 'collection-progress.json', {'added': dict(counts), 'targets': targets, 'rejected': len(rejected)})
            print(json.dumps({'event': 'collect', 'added': dict(counts), 'rejected': len(rejected)}), flush=True)
            if all(counts[s] == targets[s] for s in targets):
                break
    if any(counts[s] != targets[s] for s in targets):
        raise ValueError('Not enough unique images within locked source-pack splits')
    save(out / 'manifest.json', {'repository': tree['repository'], 'commit': tree['commit'], 'seed': seed,
         'count': len(rows), 'items': rows, 'added': dict(counts), 'rejected': rejected,
         'parent_dataset_sha256': digest(old_run / 'dataset.json'),
         'split_policy': 'Keep v1 source packs/splits. Reject new near-duplicates against ALL selected images before annotation.',
         'rights': 'Public convenience corpus; image rights remain with original authors; not a participant stimulus library.'})


def prepare(out, old_run, source):
    from fastembed import TextEmbedding
    from rag import EMBED_MODEL
    old = json.loads((old_run / 'dataset.json').read_text())
    manifest = json.loads((out / 'manifest.json').read_text())
    rows = manifest['items']
    for r in rows:
        ap = out / 'annotations' / (r['id'] + '.json')
        a = json.loads(ap.read_text())
        r.update(text=a['meaning'].strip(), annotation_sha256=digest(ap), image_sha256=digest(out / r['image']))
        if not r['text']:
            raise ValueError('Empty description')
    encoder = TextEmbedding(EMBED_MODEL, threads=2, local_files_only=True,
                            specific_model_path=str(source / 'models/bge'), cache_dir=str(source / 'models'))
    meaning = np.asarray(list(encoder.embed([r['text'] for r in rows], batch_size=32)), dtype=np.float32)
    meaning /= np.linalg.norm(meaning, axis=1, keepdims=True)
    np.savez_compressed(out / 'teacher-features.npz', meaning=meaning, ids=np.array([r['id'] for r in rows]))
    splits = {s: [r['id'] for r in rows if r['split'] == s] for s in ('train', 'validation', 'test')}
    save(out / 'dataset.json', {**old, 'source': str(out.resolve()), 'rows': rows, 'splits': splits,
         'counts': {s: len(ids) for s, ids in splits.items()}, 'source_manifest_sha256': digest(out / 'manifest.json'),
         'parent_dataset_sha256': manifest['parent_dataset_sha256'], 'added': manifest['added'],
         'split_note': manifest['split_policy'],
         'test_note': 'Old test was previously explored. Newly collected test images have no training/selection use; source packs match the old held-out packs.'})
    usage = {}
    for r in rows:
        if not r['id'].startswith('E'):
            continue
        a = json.loads((out / 'annotations' / (r['id'] + '.json')).read_text())
        usage[tuple(a.get('batch_ids', [r['id']]))] = a.get('stats', {}).get('usage', {})
    save(out / 'annotation-usage.json', {'api_requests_with_results': len(usage),
         'total_tokens': sum(u.get('total_tokens', 0) for u in usage.values()),
         'note': 'Counts one shared batch response once; unsuccessful requests may add unreported usage.'})
    print(json.dumps({'event': 'prepared', 'counts': {s: len(ids) for s, ids in splits.items()}}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['collect', 'annotate', 'prepare'])
    p.add_argument('--output', default='outputs/sticker-corpus-v2-20261112')
    p.add_argument('--parent', default='outputs/sticker-semantic-v1-20261111')
    p.add_argument('--source', default='outputs/sticker-benchmark-500')
    args = p.parse_args()
    out, parent, source = Path(args.output).resolve(), Path(args.parent).resolve(), Path(args.source).resolve()
    if args.stage == 'collect':
        collect(parent, source, out, {'train': 600, 'validation': 75, 'test': 75}, 20261112)
    elif args.stage == 'annotate':
        from benchmark_stickers import annotate
        annotate(out)
    else:
        prepare(out, parent, source)
