"""V4 semantic adapter and explicit divergence eligibility, independent axes.

Semantic labels never become negative solely because two images share a style.
Source-series exclusion is a conservative proxy, not identity recognition.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

VERSION = 'xiaoyu-semantic-adapter-v4'


class SemanticAdapter(nn.Module):
    def __init__(self, dimensions=512):
        super().__init__()
        self.projection = nn.Linear(dimensions, dimensions)

    def forward(self, x):
        return F.normalize(self.projection(x), dim=-1)


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Invalid feature matrix')
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if (norms < 1e-8).any():
        raise ValueError('Zero feature')
    return x / norms


def identity_reason(a, b):
    if a['id'] == b['id']:
        return 'self'
    for key in ('pixel_hash', 'image_sha256'):
        if a.get(key) and a[key] == b.get(key):
            return 'duplicate'
    if set(a.get('families', [])) & set(b.get('families', [])):
        return 'same_family'
    if a.get('pack') and a['pack'] == b.get('pack'):
        return 'same_source_series'
    return None


def style_eligibility(a, b, distance, minimum):
    reason = identity_reason(a, b)
    if reason:
        return reason
    if not np.isfinite(distance):
        return 'missing_style_distance'
    if a.get('medium', 'unknown') == 'unknown' or b.get('medium', 'unknown') == 'unknown':
        return 'unresolved_medium'
    if a['medium'] == b['medium']:
        return 'same_medium'
    if distance < minimum:
        return 'insufficient_style_distance'
    return None


def rank_divergence(query, candidates, semantic_threshold, style_minimum,
                    context_threshold=.72, excluded_ids=()):
    """No weighted sum can trade a semantic error for greater visual distance.

context_score is an explicitly supplied contextual/description similarity. This
offline preview uses weak captions; production must supply actual usage context.
"""
    accepted, rejected = [], []
    excluded = set(excluded_ids)
    for c in candidates:
        reason = identity_reason(query, c)
        if c['id'] in excluded:
            reason = 'participant_owned_or_used'
        s, context = c.get('semantic_score'), c.get('context_score')
        if reason is None and (s is None or not np.isfinite(s) or s < semantic_threshold):
            reason = 'semantic_gate'
        if reason is None and (context is None or not np.isfinite(context) or context < context_threshold):
            reason = 'context_not_supported'
        if reason is None:
            reason = style_eligibility(query, c, c.get('style_distance', np.nan), style_minimum)
        if reason:
            rejected.append({'id': c['id'], 'reason': reason})
        else:
            accepted.append(c)
    accepted.sort(key=lambda c: (-c['style_distance'], -c['semantic_score'], c['id']))
    return accepted, rejected


def training_objective(z, targets, styles, packs, pair_weight):
    """Caption distillation + cross-style positives + same-style hard negatives."""
    meaning = targets @ targets.T
    style_distance = 1 - styles @ styles.T
    self_mask = torch.eye(len(z), device=z.device, dtype=torch.bool)
    cross_pack = packs[:, None] != packs[None, :]
    positives = (meaning >= .78) & (style_distance >= .65) & cross_pack & ~self_mask
    negatives = (meaning <= .40) & (style_distance <= .45) & ~self_mask
    cosine = z @ z.T
    regression = (1 - (z * targets).sum(-1)).mean()
    positive = (1 - cosine[positives]).mean() if positives.any() else z.sum() * 0
    negative = F.relu(cosine[negatives] - .25).mean() if negatives.any() else z.sum() * 0
    return regression + pair_weight * (positive + negative)


def retrieval_metrics(vectors, teacher, rows, styles, k=5):
    """Multi-positive weak-caption metrics. Never presented as human accuracy."""
    vectors, teacher, styles = map(normalize, (vectors, teacher, styles))
    sim, truth, distance = vectors @ vectors.T, teacher @ teacher.T, 1 - styles @ styles.T
    totals = {'semantic_p_at_5': [], 'cross_style_hit_at_5': [],
              'cross_style_recall_at_5': [], 'cross_style_ndcg_at_5': [],
              'same_series_fraction_at_5': []}
    for i, row in enumerate(rows):
        available = [j for j, r in enumerate(rows) if j != i and r.get('image_sha256') != row.get('image_sha256')]
        if not available:
            continue
        order = sorted(available, key=lambda j: (-float(sim[i, j]), rows[j]['id']))[:k]
        totals['semantic_p_at_5'].append(sum(truth[i, j] >= .72 for j in order) / k)
        totals['same_series_fraction_at_5'].append(sum(identity_reason(row, rows[j]) is not None for j in order) / k)
        eligible = [j for j in available if style_eligibility(row, rows[j], distance[i, j], .65) is None]
        relevant = {j for j in eligible if truth[i, j] >= .72}
        if not relevant:
            continue
        ordered = sorted(eligible, key=lambda j: (-float(sim[i, j]), rows[j]['id']))[:k]
        hits = [j in relevant for j in ordered]
        dcg = sum(float(h) / np.log2(n + 2) for n, h in enumerate(hits))
        ideal = sum(1 / np.log2(n + 2) for n in range(min(k, len(relevant))))
        totals['cross_style_hit_at_5'].append(float(any(hits)))
        totals['cross_style_recall_at_5'].append(sum(hits) / len(relevant))
        totals['cross_style_ndcg_at_5'].append(float(dcg / ideal))
    return {**{key: float(np.mean(values)) if values else None for key, values in totals.items()},
            'queries': len(rows), 'cross_style_answerable_queries': len(totals['cross_style_hit_at_5']),
            'labels': 'BGE caption similarity >=0.72; machine proxy, not human gold',
            'denominator': 'P@5 uses fixed 5; cross-style metrics exclude queries without a weak positive and report their count'}
