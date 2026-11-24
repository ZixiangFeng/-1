"""Broad communicative pools for recall; never equate membership with synonymy."""
import math
from .disentangled import identity_reason, style_eligibility


def broad_rank(query, candidates, style_minimum=.65, semantic_window=20,
               query_groups=None, excluded_ids=()):
    groups = set(query_groups if query_groups is not None else query.get('intent_groups', [])[:1])
    excluded = set(excluded_ids)
    stage1, eligible, rejected = [], [], []
    for c in candidates:
        if c['id'] == query['id']:
            continue
        if not groups.intersection(c.get('intent_groups', [])):
            rejected.append({'id': c['id'], 'reason': 'different_intent_pool'})
            continue
        if not all(math.isfinite(c.get(k, float('nan'))) for k in ('semantic_score', 'context_score', 'style_distance')):
            rejected.append({'id': c['id'], 'reason': 'invalid_score'})
            continue
        # Both terms describe meaning; visual distance never compensates meaning.
        r = dict(c, recall_score=.6*c['context_score']+.4*c['semantic_score'])
        stage1.append(r)
        reason = 'participant_owned_or_used' if c['id'] in excluded else style_eligibility(query, c, c['style_distance'], style_minimum)
        if reason:
            rejected.append({'id': c['id'], 'reason': reason})
        else:
            eligible.append(r)
    stage1.sort(key=lambda r: (-r['recall_score'], r['id']))
    eligible.sort(key=lambda r: (-r['recall_score'], r['id']))
    # A fixed semantic rank window bounds broad-pool drift before style sorting.
    window = [dict(r, semantic_rank_before_style=i+1) for i, r in enumerate(eligible[:semantic_window])]
    for r in eligible[semantic_window:]:
        rejected.append({'id': r['id'], 'reason': 'outside_semantic_window'})
    window.sort(key=lambda r: (-r['style_distance'], -r['recall_score'], r['id']))
    return {'stage1': stage1, 'stage2': window, 'rejected': rejected,
            'policy': 'Broad recall candidates, NOT confirmed synonymous; actual chat context still required',
            'query_groups': sorted(groups), 'semantic_window': semantic_window}
