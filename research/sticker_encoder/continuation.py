"""Weak semantic-neighborhood distillation and auditable continuation helpers."""
import numpy as np
import torch
import torch.nn.functional as F


def relational_loss(images, teacher_similarity, temperature=.1):
    """Match soft meaning neighborhoods; no teacher pair is declared gold."""
    if len(images) < 2 or temperature <= 0:
        raise ValueError('At least two images and a positive temperature required')
    mask = torch.eye(len(images), dtype=torch.bool, device=images.device)
    student = (images @ images.T / temperature).masked_fill(mask, -1e4)
    teacher = (teacher_similarity.detach() / temperature).masked_fill(mask, -1e4)
    probabilities = F.softmax(teacher, dim=1)
    return F.kl_div(F.log_softmax(student, dim=1), probabilities, reduction='batchmean')


def validate_continuation(data, parent):
    """Parent holdouts must never be silently reassigned or edited."""
    old = {r['id']: r for r in parent['rows']}
    current = {r['id']: r for r in data['rows']}
    if len(current) != len(data['rows']) or not set(old) <= set(current):
        raise ValueError('Missing or repeated IDs')
    packs = {}
    for r in data['rows']:
        if r['split'] not in ('train', 'validation', 'test'):
            raise ValueError('Unknown split')
        if r['pack'] in packs and packs[r['pack']] != r['split']:
            raise ValueError('Source pack crosses splits')
        packs[r['pack']] = r['split']
        if r.get('human_reviewed') is not False:
            raise ValueError('Automatic labels must remain marked unreviewed')
        if r['id'] in old:
            for field in ('split', 'pack', 'image_sha256', 'annotation_sha256', 'text'):
                if r[field] != old[r['id']][field]:
                    raise ValueError('Parent sample changed: ' + field)
    for split in ('train', 'validation', 'test'):
        if {r['id'] for r in data['rows'] if r['split'] == split} != set(data['splits'][split]):
            raise ValueError('Split manifest mismatch')


def neighborhood_overlap(images, teacher, k=5):
    """Fraction of image neighbors agreeing with weak text-teacher top-k."""
    a, b = np.asarray(images) @ np.asarray(images).T, np.asarray(teacher) @ np.asarray(teacher).T
    np.fill_diagonal(a, -np.inf); np.fill_diagonal(b, -np.inf)
    k = min(k, len(a) - 1)
    if k < 1:
        raise ValueError('Insufficient evaluation items')
    ai, bi = np.argsort(-a, axis=1)[:, :k], np.argsort(-b, axis=1)[:, :k]
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(ai, bi)]))


def paired_bootstrap(before, after, texts, seed=20261112, samples=2000):
    """Query-paired descriptive CI; not adjusted for pack clustering."""
    n = len(texts)
    def scores(images):
        sim = images @ texts.T
        ranks = np.argsort(np.argsort(-sim, axis=1), axis=1)[np.arange(n), np.arange(n)] + 1
        return {'recall_at_1': (ranks == 1).astype(float), 'recall_at_5': (ranks <= 5).astype(float), 'mrr': 1 / ranks}
    a, b = scores(before), scores(after)
    draws = np.random.default_rng(seed).integers(0, n, size=(samples, n))
    return {name: {'delta': float((b[name] - a[name]).mean()),
                   'query_bootstrap_95ci': np.quantile((b[name] - a[name])[draws].mean(1), [.025, .975]).tolist()}
            for name in a}
