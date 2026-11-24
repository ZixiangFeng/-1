import unittest
import numpy as np
import torch
from research.sticker_encoder.weak_model import split_groups, contrastive_loss, retrieval


class WeakStickerTests(unittest.TestCase):
    def test_pack_and_cross_pack_duplicates_stay_together(self):
        rows = [{'id': str(i), 'pack': str(i // 2), 'pixel_hash': str(i)} for i in range(40)]
        result = split_groups(rows, [('0', '3')])
        assignment = {v: k for k, values in result.items() for v in values}
        self.assertEqual(len(assignment), 40)
        self.assertEqual(assignment['0'], assignment['3'])
        for i in range(0, 40, 2):
            self.assertEqual(assignment[str(i)], assignment[str(i + 1)])
        self.assertEqual(result, split_groups(rows, [('0', '3')]))

    def test_likely_synonyms_are_not_forced_apart(self):
        x = torch.tensor([[1., 0.], [1., 0.]], requires_grad=True)
        loss = contrastive_loss(x, x.detach(), [0, 1], torch.ones(2, 2), torch.tensor(10.))
        self.assertAlmostEqual(loss.item(), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_same_caption_multi_positive_is_finite(self):
        x = torch.eye(3, requires_grad=True)
        loss = contrastive_loss(x, x.detach(), [0, 0, 1], torch.eye(3), torch.tensor(10.))
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(loss.item(), 0)

    def test_retrieval_has_correct_direction(self):
        x = np.eye(5)
        self.assertEqual(retrieval(x, x)['image_to_caption']['recall_at_1'], 1.)
        self.assertEqual(retrieval(x, np.roll(x, 1, axis=0))['image_to_caption']['recall_at_1'], 0.)


if __name__ == '__main__':
    unittest.main()
