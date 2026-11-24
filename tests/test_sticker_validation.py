import unittest
import numpy as np
from scripts.validate_sticker_encoder import ranked_metrics, cluster_bootstrap


class FrozenEvaluationTests(unittest.TestCase):
    def test_global_targets_with_different_gallery_order(self):
        vectors = np.eye(6, dtype=np.float32)
        metrics, ranks = ranked_metrics(vectors, vectors, [5, 1], [3, 1, 4, 5, 0, 2])
        for direction in metrics:
            self.assertEqual(metrics[direction]['n_queries'], 2)
            self.assertEqual(metrics[direction]['n_candidates'], 6)
            self.assertEqual(metrics[direction]['recall_at_1'], 1.)
            self.assertEqual(ranks[direction].tolist(), [1, 1])

    def test_top_five_is_distinct_from_top_one(self):
        images = np.eye(8, dtype=np.float32)
        texts = np.roll(images, 1, axis=0)
        metrics, ranks = ranked_metrics(images, texts, list(range(8)), list(range(8)))
        for direction in metrics:
            self.assertEqual(metrics[direction]['hits_at_1'], 0)
            self.assertGreater(metrics[direction]['hits_at_5'], 0)
            self.assertTrue((ranks[direction] > 1).all())

    def test_paired_pack_bootstrap_identical_models(self):
        ranks = {'image_to_caption': np.array([1, 5, 10, 3]), 'caption_to_image': np.array([2, 6, 1, 8])}
        result = cluster_bootstrap(ranks, ranks, ['a', 'a', 'b', 'b'])
        self.assertEqual(result['n_packs'], 2)
        for direction in result['results'].values():
            for metric in direction.values():
                self.assertEqual(metric['difference_v2_minus_v1'], 0.)
                self.assertEqual(metric['paired_pack_bootstrap_95ci'], [0., 0.])


if __name__ == '__main__':
    unittest.main()
