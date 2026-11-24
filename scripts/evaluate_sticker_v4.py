"""One fixed holdout evaluation and full-corpus, auditable divergence results."""
import json, sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from safetensors.torch import load_file
from research.sticker_encoder.weak_model import save, digest
from research.sticker_encoder.disentangled import SemanticAdapter, retrieval_metrics, rank_divergence, identity_reason
from scripts.train_sticker_v4 import align

OUT = ROOT / 'outputs/sticker-divergence-v4-20261119'
SOURCE = ROOT / 'outputs/sticker-semantic-v3-local-20261118'


def main():
    if (OUT / 'evaluation-report.json').exists():
        raise ValueError('Frozen evaluation already exists')
    lock = json.loads((OUT / 'checkpoint-lock.json').read_text())
    for key, file in [('checkpoint_sha256', 'semantic-adapter.safetensors'), ('report_sha256', 'training-report.json'), ('data_sha256', 'dataset.json')]:
        if digest(OUT / file) != lock[key]:
            raise ValueError('Locked artifact changed')
    rows = json.loads((OUT / 'dataset.json').read_text())['rows']
    report = json.loads((OUT / 'training-report.json').read_text())
    ids = [r['id'] for r in rows]
    f = np.load(SOURCE / 'evaluation-features.npz')
    x, y, s = align(f, 'v3_image', ids), align(np.load(OUT / 'all-caption-features.npz'), 'meaning', ids), align(np.load(SOURCE / 'all-style-features.npz'), 'style', ids)
    v2 = align(np.load(SOURCE / 'v2-evaluation-features.npz'), 'v2_image', ids)
    model = SemanticAdapter(); model.load_state_dict(load_file(str(OUT / 'semantic-adapter.safetensors'))); model.eval()
    with torch.no_grad():
        z = model(torch.from_numpy(x)).numpy()
    models = {'v2': v2, 'v3': x, 'v4': z}
    metrics = {}
    for split in ('validation', 'test'):
        selected = [i for i, r in enumerate(rows) if r['split'] == split and r['quality'] != 'uncertain']
        sub = [rows[i] for i in selected]
        metrics[split] = {name: retrieval_metrics(v[selected], y[selected], sub, s[selected]) for name, v in models.items()}
    all_test = [i for i, r in enumerate(rows) if r['split'] == 'test']
    metrics['test_including_uncertain'] = {name: retrieval_metrics(v[all_test], y[all_test], [rows[i] for i in all_test], s[all_test]) for name, v in models.items()}
    similarities = {name: np.clip(v @ v.T, -1, 1) for name, v in models.items()}
    style = np.clip(1 - s @ s.T, 0, 2)
    meaning = np.clip(y @ y.T, -1, 1)
    np.savez_compressed(OUT / 'retrieval-matrices.npz', ids=np.array(ids), **similarities, style=style, meaning=meaning)
    threshold = report['calibration']['threshold']; style_min = report['calibration']['style_minimum']
    results, legacy, rejected_total = {}, {}, Counter()
    for i, q in enumerate(rows):
        candidates = [{**r, 'semantic_score': float(similarities['v4'][i, j]), 'context_score': float(meaning[i, j]),
                       'style_distance': float(style[i, j])} for j, r in enumerate(rows) if i != j]
        accepted, rejected = rank_divergence(q, candidates, threshold, style_min)
        results[q['id']] = {'accepted': [{k: r[k] for k in ('id', 'semantic_score', 'context_score', 'style_distance')} for r in accepted],
                            'rejected_counts': dict(Counter(r['reason'] for r in rejected))}
        rejected_total.update(r['reason'] for r in rejected)
        old = sorted([j for j in range(len(rows)) if j != i and similarities['v2'][i, j] >= .7], key=lambda j: -style[i, j])[:5]
        legacy[q['id']] = {'top5': [rows[j]['id'] for j in old],
                           'same_family_or_series': sum(identity_reason(q, rows[j]) is not None for j in old),
                           'same_medium': sum(q['medium'] == rows[j]['medium'] and q['medium'] != 'unknown' for j in old)}
    save(OUT / 'two-stage-results.json', {'threshold': threshold, 'style_minimum': style_min,
         'context_threshold': .72, 'context_source': 'archived_AI_generic_captions_not_personal_chat', 'queries': results})
    test_results = [results[rows[i]['id']]['accepted'] for i in all_test]
    report_out = {'metrics': metrics, 'split_counts_total': dict(Counter(r['split'] for r in rows)),
                  'labels': 'AI captions and machine style labels; not independently human-judged semantic accuracy',
                  'test_used_in_fit_or_selection': 0, 'test_fresh_blind': False,
                  'final_policy_audit': {'query_n': len(rows), 'queries_with_candidates': sum(bool(v['accepted']) for v in results.values()),
                     'heldout_query_n': len(all_test), 'heldout_with_candidates': sum(bool(v) for v in test_results),
                     'returned_pairs': sum(len(v['accepted']) for v in results.values()),
                     'rejections': dict(rejected_total), 'same_family_violation': 0, 'same_medium_violation': 0,
                     'interpretation': 'Zero forbidden outputs is a gate property, not learned model accuracy; weak semantic gate uses the same caption teacher and cannot validate itself.'},
                  'screenshot_regression': {'query': 'B0032', 'old_top5': legacy['B0032'], 'new': results['B0032'],
                                           'must_exclude': ['B0128', 'B0314', 'B0403']},
                  'checkpoint_lock_sha256': digest(OUT / 'checkpoint-lock.json')}
    # Validate policy invariants, not just assign zero counts.
    by = {r['id']: r for r in rows}
    for qid, result in results.items():
        for a in result['accepted']:
            assert identity_reason(by[qid], by[a['id']]) is None
            assert by[qid]['medium'] != by[a['id']]['medium']
            assert 'unknown' not in (by[qid]['medium'], by[a['id']]['medium'])
            assert a['style_distance'] >= style_min and a['context_score'] >= .72
    assert not set(report_out['screenshot_regression']['must_exclude']) & {x['id'] for x in results['B0032']['accepted']}
    save(OUT / 'evaluation-report.json', report_out)
    print(json.dumps(report_out, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
