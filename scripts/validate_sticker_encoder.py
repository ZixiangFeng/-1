# -*- coding: utf-8 -*-
"""Freeze existing weights, audit/export splits, and independently rerun inference.

No training, threshold fitting or participant database writes. Automatic image
descriptions are a retrieval proxy and are never represented as human gold.
"""
import argparse
import base64
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from research.sticker_encoder.weak_model import digest, save

V1 = ROOT / 'outputs/sticker-semantic-v1-20261111'
V2 = ROOT / 'outputs/sticker-semantic-v2-500-20261112'
BASE = ROOT / 'models/research/chinese-clip-vit-base-patch16'
EXTRA = ROOT / 'outputs/sticker-corpus-v2-20261112'


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fields, extrasaction='ignore'); w.writeheader(); w.writerows(rows)


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    data = json.loads((V2 / 'dataset.json').read_text())
    old = json.loads((V1 / 'dataset.json').read_text())
    from research.sticker_encoder.continuation import validate_continuation
    validate_continuation(data, old)
    source = Path(data['source'])
    protocol = {'purpose': 'Frozen-model validation; no training or model selection',
                'old_split_policy': 'Preserve original source-pack/near-duplicate split; never reshuffle already-trained images into a test set.',
                'original_counts': data['counts'],
                'fresh_policy': 'All 75 E-series images assigned to test in the earlier expansion manifest; no score-based selection.',
                'weights': {}, 'evaluation': ['Original within-split paired-caption retrieval',
                    'Fixed common 500-caption gallery for all original train/validation/test queries',
                    'Fresh 75-image within-split retrieval if all fresh annotations are available'],
                'primary_direction': 'image_to_caption', 'secondary_direction': 'caption_to_image',
                'uncertainty': 'Paired source-pack cluster bootstrap for v2-v1 differences; descriptive with only 18 old test packs.',
                'limitations': ['Machine captions are not gold semantic labels.',
                    'Original test images were previously evaluated and browsed, not a new blind test.',
                    'Fresh images share held-out packs with the old test; not an external new-source test.',
                    'No human labels available to estimate semantic equivalence precision or perceived style divergence.'],
                'dataset_sha256': digest(V2 / 'dataset.json')}
    for name, folder in [('v1', V1), ('v2', V2)]:
        report = json.loads((folder / 'training-report.json').read_text())
        for filename, sha in report['checkpoint_files'].items():
            if digest(folder / 'checkpoint' / filename) != sha:
                raise ValueError('Changed checkpoint ' + name)
        protocol['weights'][name] = report['checkpoint_files']
    base = json.loads((BASE / 'download.json').read_text())
    for filename, info in base['files'].items():
        if digest(BASE / filename) != info['sha256']:
            raise ValueError('Changed base model')
    protocol['weights']['base'] = {k: v['sha256'] for k, v in base['files'].items()}
    if (out / 'protocol.json').exists():
        if json.loads((out / 'protocol.json').read_text()) != protocol:
            raise ValueError('Existing evaluation protocol differs; use a new output folder')
    else:
        save(out / 'protocol.json', protocol)
    exported = []
    for r in data['rows']:
        split = r['split']; folder = out / 'splits' / split
        (folder / 'images').mkdir(parents=True, exist_ok=True)
        (folder / 'annotations').mkdir(exist_ok=True)
        ip, ap = source / r['image'], source / 'annotations' / (r['id'] + '.json')
        if digest(ip) != r['image_sha256'] or digest(ap) != r['annotation_sha256']:
            raise ValueError('Dataset content changed ' + r['id'])
        image = folder / 'images' / (r['id'] + '.png')
        shutil.copy2(ip, image); shutil.copy2(ap, folder / 'annotations' / ap.name)
        exported.append({**r, 'export_image': str(image.relative_to(out)), 'eval_split': split})
    fresh_manifest = json.loads((EXTRA / 'manifest.json').read_text())
    fresh = [dict(r) for r in fresh_manifest['items'] if r['id'].startswith('E') and r['split'] == 'test']
    if len(fresh) != 75:
        raise ValueError('Unexpected fresh holdout count')
    corpus = out / 'fresh-test-corpus'; (corpus / 'images').mkdir(parents=True, exist_ok=True)
    for r in fresh:
        path = EXTRA / r['image']
        shutil.copy2(path, corpus / 'images' / (r['id'] + '.png'))
        r.update(image='images/' + r['id'] + '.png', image_sha256=digest(path), eval_split='fresh_test',
                 export_image=str((corpus / 'images' / (r['id'] + '.png')).relative_to(out)))
    save(corpus / 'manifest.json', {**{k: fresh_manifest[k] for k in ('repository', 'commit')}, 'count': 75, 'items': fresh})
    save(out / 'evaluation-items.json', {'old': exported, 'fresh': fresh})
    for s in ('train', 'validation', 'test'):
        rows = [r for r in exported if r['eval_split'] == s]
        save(out / 'splits' / s / 'manifest.json', {'split': s, 'count': len(rows), 'items': rows})
        write_csv(out / 'splits' / s / 'manifest.csv', rows, ['id', 'pack', 'split', 'text', 'image_sha256', 'annotation_sha256', 'export_image'])
    write_csv(corpus / 'manifest.csv', fresh, ['id', 'pack', 'split', 'image_sha256', 'export_image'])
    print(json.dumps({'event': 'prepared', 'counts': data['counts'], 'fresh': len(fresh)}), flush=True)


def audit(out):
    import sticker_identity
    items = json.loads((out / 'evaluation-items.json').read_text())
    rows = items['old'] + items['fresh']
    fps = [sticker_identity.fingerprint(base64.b64encode((out / r['export_image']).read_bytes()).decode()) for r in rows]
    coarse = np.array([list(f['coarse']) for f in fps], dtype=np.float32)
    near = []
    for i, f in enumerate(fps):
        for j in np.flatnonzero(np.mean(np.abs(coarse - coarse[i]), axis=1) <= 8):
            if j > i:
                kind = sticker_identity.same_image(f, fps[j])
                if kind:
                    near.append({'a': rows[i]['id'], 'b': rows[j]['id'], 'split_a': rows[i]['eval_split'], 'split_b': rows[j]['eval_split'], 'type': kind})
    sets = {s: [r for r in rows if r['eval_split'] == s] for s in ('train', 'validation', 'test', 'fresh_test')}
    comparisons = []
    for i, a in enumerate(sets):
        for b in list(sets)[i+1:]:
            values = {'a': a, 'b': b}
            for field in ('id', 'image_sha256', 'pack'):
                values[field+'_overlap'] = sorted({r[field] for r in sets[a]} & {r[field] for r in sets[b]})
            comparisons.append(values)
    cross = [p for p in near if p['split_a'] != p['split_b']]
    illegal_packs = [r for r in comparisons if r['pack_overlap'] and {r['a'], r['b']} != {'test', 'fresh_test'}]
    result = {'counts': {s: len(r) for s, r in sets.items()}, 'packs': {s: len({r['pack'] for r in rs}) for s, rs in sets.items()},
              'comparisons': comparisons, 'all_near_duplicate_pairs': near, 'cross_split_near_duplicates': cross,
              'id_unique': len({r['id'] for r in rows}) == len(rows),
              'passed': not cross and not illegal_packs and not any(r['id_overlap'] or r['image_sha256_overlap'] for r in comparisons),
              'note': 'Source packs and conservative pixel/thumbnail matching; not a guarantee against all cropped/edited near-duplicates or shared characters. Fresh and old test share packs by design.'}
    save(out / 'split-audit.json', result)
    if not result['passed']:
        raise ValueError('Split audit failed; no evaluation until resolved')
    print(json.dumps({'event': 'audit', 'passed': True, 'counts': result['counts'], 'cross_split_near_duplicates': len(cross)}), flush=True)


def ranked_metrics(images, texts, query_indices, gallery_indices):
    """Stable descending sort; a query's exact paired description is its target."""
    query_indices, gallery_indices = list(query_indices), list(gallery_indices)
    positions = {x: j for j, x in enumerate(gallery_indices)}
    targets = np.array([positions[i] for i in query_indices])
    result, ranks_by_direction = {}, {}
    for direction, q, g in [('image_to_caption', images, texts), ('caption_to_image', texts, images)]:
        similarities = q[query_indices] @ g[gallery_indices].T
        order = np.argsort(-similarities, axis=1, kind='stable')
        ranks = np.argmax(order == targets[:, None], axis=1) + 1
        ranks_by_direction[direction] = ranks
        result[direction] = {'n_queries': len(query_indices), 'n_candidates': len(gallery_indices),
            'hits_at_1': int((ranks <= 1).sum()), 'hits_at_5': int((ranks <= 5).sum()), 'hits_at_10': int((ranks <= 10).sum()),
            'recall_at_1': float((ranks <= 1).mean()), 'recall_at_5': float((ranks <= 5).mean()),
            'recall_at_10': float((ranks <= 10).mean()), 'mrr': float((1/ranks).mean()), 'median_rank': float(np.median(ranks))}
    return result, ranks_by_direction


def cluster_bootstrap(before, after, packs, seed=20261118):
    groups = sorted(set(packs)); groups_ix = [np.flatnonzero(np.array(packs) == p) for p in groups]
    rng = np.random.default_rng(seed)
    draws = [np.concatenate([groups_ix[i] for i in row]) for row in rng.integers(0, len(groups), size=(5000, len(groups)))]
    results = {}
    for direction in before:
        a, b = before[direction], after[direction]
        values = {'recall_at_1': ((a <= 1).astype(float), (b <= 1).astype(float)),
                  'recall_at_5': ((a <= 5).astype(float), (b <= 5).astype(float)), 'mrr': (1/a, 1/b)}
        results[direction] = {}
        for key, (x, y) in values.items():
            delta = y-x
            results[direction][key] = {'difference_v2_minus_v1': float(delta.mean()),
                'paired_pack_bootstrap_95ci': np.quantile([delta[ix].mean() for ix in draws], [.025, .975]).tolist()}
    return {'n_packs': len(groups), 'resamples': 5000, 'results': results,
            'note': 'Frozen candidate pool; resample source-pack query clusters. Descriptive, not an external validation or a correction for prior test inspection.'}


def evaluate(out, device_name):
    import gc
    import os
    import torch
    import torch.nn.functional as F
    from transformers import ChineseCLIPModel, ChineseCLIPProcessor
    from research.sticker_encoder.weak_model import rgb
    if not json.loads((out / 'split-audit.json').read_text())['passed']:
        raise ValueError('Audit has not passed')
    protocol = json.loads((out / 'protocol.json').read_text())
    items = json.loads((out / 'evaluation-items.json').read_text()); rows = items['old'].copy()
    corpus = out / 'fresh-test-corpus'
    available = all((corpus / 'annotations' / (r['id']+'.json')).exists() for r in items['fresh'])
    if available:
        for r in items['fresh']:
            ap = corpus / 'annotations' / (r['id']+'.json')
            a = json.loads(ap.read_text())
            rows.append({**r, 'text': a['meaning'], 'annotation_sha256': digest(ap), 'human_reviewed': False})
    torch.set_num_threads(4); torch.manual_seed(20261118); os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    device = torch.device(device_name)
    processor = ChineseCLIPProcessor.from_pretrained(BASE, local_files_only=True)
    pixels = processor(images=[rgb(out/r['export_image']) for r in rows], return_tensors='pt')['pixel_values']
    indices = {s: [i for i, r in enumerate(rows) if r['eval_split'] == s] for s in ('train', 'validation', 'test', 'fresh_test')}
    result = {'protocol': protocol, 'actual_counts': {s: len(ix) for s, ix in indices.items()},
              'fresh_test_evaluated': available, 'metrics': {}, 'cached_feature_comparison': {}, 'weights_changed': False,
              'evaluation_note': 'Independent image/text encoding from saved checkpoints under inference_mode; no optimizer, gradients, fine-tuning or model selection.',
              'label_status': 'Automatic descriptions only; no human gold semantics/style evaluation.'}
    all_ranks = {}; features = {}; prediction_rows = []
    for label, checkpoint in [('base', BASE), ('v1', V1/'checkpoint'), ('v2', V2/'checkpoint')]:
        for filename, sha in protocol['weights'][label].items():
            if digest(checkpoint / filename) != sha:
                raise ValueError('Checkpoint changed after protocol freeze')
        print(json.dumps({'event': 'loading', 'model': label}), flush=True)
        model = ChineseCLIPModel.from_pretrained(checkpoint, local_files_only=True, use_safetensors=label!='base', weights_only=True).eval().to(device)
        images, texts = [], []
        with torch.inference_mode():
            for start in range(0, len(rows), 32):
                images.append(F.normalize(model.get_image_features(pixel_values=pixels[start:start+32].to(device)), dim=-1).cpu())
                tokens = processor(text=[r['text'] for r in rows[start:start+32]], truncation=True, max_length=64, padding=True, return_tensors='pt').to(device)
                texts.append(F.normalize(model.get_text_features(**tokens), dim=-1).cpu())
        images, texts = torch.cat(images).numpy(), torch.cat(texts).numpy()
        if not np.isfinite(images).all() or not np.isfinite(texts).all():
            raise ValueError('Nonfinite features')
        features[label] = (images, texts)
        result['metrics'][label] = {}; all_ranks[label] = {}
        for split, ix in indices.items():
            if not ix: continue
            m, ranks = ranked_metrics(images, texts, ix, ix)
            result['metrics'][label][split] = {'within_split': m}
            all_ranks[label][split] = ranks
            if split != 'fresh_test':
                common, _ = ranked_metrics(images, texts, ix, range(500))
                result['metrics'][label][split]['common_500_gallery'] = common
            for j, i in enumerate(ix):
                prediction_rows.append({'model': label, 'id': rows[i]['id'], 'split': split,
                    'pack': rows[i]['pack'], 'image_to_caption_rank': int(ranks['image_to_caption'][j]),
                    'caption_to_image_rank': int(ranks['caption_to_image'][j]), 'caption': rows[i]['text']})
        cache = np.load(V2/'features.npz' if label=='v2' else V1/'features.npz', allow_pickle=False)
        field = 'baseline' if label=='base' else 'trained'
        err = float(np.max(np.abs(images[:500]-cache[field])))
        result['cached_feature_comparison'][label] = {'max_element_difference': err, 'passed': err < 1e-5}
        if err >= 1e-5:
            raise ValueError('Independent inference differs from archived features')
        del model; gc.collect()
        if device_name == 'mps': torch.mps.empty_cache()
        print(json.dumps({'event': 'evaluated', 'model': label, 'old_test': result['metrics'][label]['test']['within_split']}), flush=True)
    result['paired_model_differences'] = {}
    for split in ('test', 'fresh_test'):
        if indices[split]:
            result['paired_model_differences'][split] = cluster_bootstrap(all_ranks['v1'][split], all_ranks['v2'][split], [rows[i]['pack'] for i in indices[split]])
    for name in ('v1', 'v2'):
        before = json.loads(((V1 if name=='v1' else V2)/'training-report.json').read_text())['trained_test']
        for direction in before:
            for metric in ('recall_at_1', 'recall_at_5', 'recall_at_10', 'mrr', 'median_rank'):
                if abs(result['metrics'][name]['test']['within_split'][direction][metric]-before[direction][metric]) > 1e-7:
                    raise ValueError('Archived metric could not be reproduced')
    result['previous_test_metrics_reproduced'] = True
    result['software'] = {'torch': torch.__version__, 'numpy': np.__version__, 'device': str(device), 'python': sys.version}
    result['code_sha256'] = digest(Path(__file__))
    save(out/'validation-report.json', result)
    save(out/'evaluated-items.json', rows)
    write_csv(out/'per-image-predictions.csv', prediction_rows, ['model','id','split','pack','image_to_caption_rank','caption_to_image_rank','caption'])
    np.savez_compressed(out/'recomputed-features.npz', ids=np.array([r['id'] for r in rows]),
        **{name+'_'+kind: values[j] for name,values in features.items() for j,kind in enumerate(('image','text'))})
    print(json.dumps({'event':'complete','fresh_evaluated':available,'old_test_reproduced':True}),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare','audit','annotate-fresh','evaluate'])
    p.add_argument('--output', default='outputs/sticker-validation-20261118')
    p.add_argument('--device', default='mps')
    args=p.parse_args(); out=Path(args.output).resolve()
    if args.stage=='prepare': prepare(out)
    elif args.stage=='audit': audit(out)
    elif args.stage=='annotate-fresh':
        from benchmark_stickers import annotate
        annotate(out/'fresh-test-corpus')
    else: evaluate(out,args.device)
