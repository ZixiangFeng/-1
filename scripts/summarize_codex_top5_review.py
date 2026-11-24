# -*- coding: utf-8 -*-
"""Convert explicit Codex image judgments; never present them as human gold."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.sticker_encoder.top5_review import summarize


def main():
    run = ROOT / 'outputs/sticker-semantic-v3-local-20261118'
    folder = run / 'ai-review'
    plan = json.loads((run / 'ai-top5-plan.json').read_text())
    mapping = json.loads((folder / 'mapping.json').read_text())
    key = json.loads((run / 'top5-researcher-key.json').read_text())
    ratings, seen = {}, set()
    for line in (folder / 'ratings.tsv').read_text().splitlines():
        if not line.strip():
            continue
        query, values, reason = line.split('|', 2)
        labels = values.split(',')
        if query in seen or query not in plan['query_ids']:
            raise ValueError('Duplicate or unexpected query: ' + query)
        if len(labels) != len(mapping[query]):
            raise ValueError('Candidate count: ' + query)
        for pair, label in zip(mapping[query], labels):
            if pair not in key['pairs'] or label not in ('yes', 'no', 'uncertain'):
                raise ValueError('Invalid judgment: ' + pair)
            ratings[pair] = {'meaning': label, 'notes': reason}
        seen.add(query)
    if seen != set(plan['query_ids']):
        raise ValueError('Sample incomplete')
    export = {
        'schema': 'xiaoyu-top5-review-v1',
        'revision': hashlib.sha256((run / 'checkpoint-lock.json').read_bytes()).hexdigest(),
        'rater_id': 'codex-ai-20261118', 'rater_type': 'ai',
        'ratings': ratings,
    }
    summary = {
        'rater_type': 'ai', 'human_gold': False,
        'sampling_plan': plan, 'queries_reviewed': len(seen),
        'unique_pairs_reviewed': len(ratings),
        'label_counts': dict(Counter(r['meaning'] for r in ratings.values())),
        'criteria': 'Same pragmatic act and core meaning in a plausible common chat context. '
                    'Shared character, template, broad emotion or topic alone is insufficient. '
                    'Unclear intent or partial overlap remains uncertain.',
        'limitations': 'Single Codex AI reviewer; same AI system generated training weak labels. '
                       'No independent human agreement, no private user meanings. '
                       'Only a random20 of75 fresh test queries; exploratory diagnosis, '
                       'not an estimate of production accuracy. Uncertain labels form bounds, '
                       'which are not confidence intervals. No subsequent fitting on these judgments.',
        'metrics': summarize(key['rankings'], ratings, plan['query_ids']),
    }
    for name, value in [('ratings.json', export), ('summary.json', summary)]:
        (folder / name).write_text(json.dumps(value, ensure_ascii=False, indent=2))
    print(json.dumps({
        'queries': len(seen), 'pairs': len(ratings), 'labels': summary['label_counts'],
        'metrics': {m: {k: v for k, v in data.items() if k != 'per_query'}
                    for m, data in summary['metrics'].items()},
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
