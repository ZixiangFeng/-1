import unittest
import numpy as np
import torch
from research.sticker_encoder.disentangled import (
    rank_divergence, style_eligibility, training_objective, retrieval_metrics,
)


class DisentangledTests(unittest.TestCase):
    def setUp(self):
        self.q = {'id': 'q', 'pack': 'series-a', 'families': ['face-a'], 'medium': 'photo', 'image_sha256': 'q'}
        self.c = {'id': 'c', 'pack': 'series-b', 'families': [], 'medium': 'line', 'image_sha256': 'c',
                  'semantic_score': .9, 'context_score': .9, 'style_distance': .9}

    def rank(self, c, **kwargs):
        return rank_divergence(self.q, [c], .7, .65, **kwargs)

    def test_same_face_redraw_excluded_even_with_high_style_distance(self):
        a, r = self.rank(dict(self.c, families=['face-a']))
        self.assertEqual(a, []); self.assertEqual(r[0]['reason'], 'same_family')

    def test_same_source_series_is_conservative_exclusion(self):
        a, r = self.rank(dict(self.c, pack='series-a'))
        self.assertFalse(a); self.assertEqual(r[0]['reason'], 'same_source_series')

    def test_style_cannot_compensate_for_wrong_semantics(self):
        a, r = self.rank(dict(self.c, semantic_score=.2, style_distance=1.9))
        self.assertFalse(a); self.assertEqual(r[0]['reason'], 'semantic_gate')

    def test_context_and_visual_gates_are_independent(self):
        for field, value, reason in [('context_score', .4, 'context_not_supported'),
                                     ('medium', 'photo', 'same_medium'),
                                     ('medium', 'unknown', 'unresolved_medium'),
                                     ('style_distance', .2, 'insufficient_style_distance')]:
            a, r = self.rank(dict(self.c, **{field: value}))
            self.assertFalse(a); self.assertEqual(r[0]['reason'], reason)

    def test_participant_own_or_used_images_cannot_enter(self):
        a, r = self.rank(self.c, excluded_ids=['c'])
        self.assertFalse(a); self.assertEqual(r[0]['reason'], 'participant_owned_or_used')

    def test_nan_fails_closed(self):
        for key in ('semantic_score', 'context_score', 'style_distance'):
            self.assertFalse(self.rank(dict(self.c, **{key: float('nan')}))[0])

    def test_accepted_rank_keeps_meaning_gate_and_orders_style(self):
        a, r = rank_divergence(self.q, [self.c, dict(self.c, id='d', style_distance=1.1)], .7, .65)
        self.assertEqual([x['id'] for x in a], ['d', 'c']); self.assertEqual(r, [])

    def test_same_style_same_meaning_is_not_a_semantic_negative(self):
        z = torch.tensor([[1., 0.], [1., 0.]], requires_grad=True)
        loss = training_objective(z, z.detach(), z.detach(), torch.tensor([0, 1]), .35)
        self.assertAlmostEqual(loss.item(), 0.)

    def test_same_style_different_meaning_is_penalized(self):
        z = torch.tensor([[1., 0.], [1., 0.]], requires_grad=True)
        targets = torch.eye(2); styles = torch.tensor([[1., 0.], [1., 0.]])
        a = training_objective(z, targets, styles, torch.tensor([0, 1]), 0.)
        b = training_objective(z, targets, styles, torch.tensor([0, 1]), .35)
        self.assertGreater(b.item(), a.item()); b.backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_multi_positive_retrieval_reports_missing_coverage(self):
        vectors = np.array([[1., 0.], [1., 0.], [0., 1.]])
        rows = [self.q, self.c, dict(self.c, id='e', image_sha256='e', pack='c', medium='line')]
        m = retrieval_metrics(vectors, vectors, rows, np.array([[1., 0.], [0., 1.], [0., 1.]]))
        self.assertEqual(m['queries'], 3)
        self.assertEqual(m['cross_style_answerable_queries'], 2)
        self.assertAlmostEqual(m['cross_style_hit_at_5'], 1.)
        self.assertLess(m['semantic_p_at_5'], 1.)


if __name__ == '__main__':
    unittest.main()
