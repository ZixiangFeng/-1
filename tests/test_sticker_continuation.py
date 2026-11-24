import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from research.sticker_encoder.continuation import relational_loss, validate_continuation, neighborhood_overlap, paired_bootstrap


class ContinuationTests(unittest.TestCase):
    def example(self):
        rows = [{'id': str(i), 'pack': str(i), 'split': split, 'image_sha256': 'img'+str(i),
                 'annotation_sha256': 'a'+str(i), 'text': 'caption'+str(i), 'human_reviewed': False}
                for i, split in enumerate(('train', 'validation', 'test'))]
        return {'rows': rows, 'splits': {r['split']: [r['id']] for r in rows}}

    def test_original_holdout_cannot_enter_training(self):
        parent = self.example(); current = copy.deepcopy(parent)
        current['rows'][2]['split'] = 'train'
        with self.assertRaisesRegex(ValueError, 'Parent sample changed'):
            validate_continuation(current, parent)

    def test_pack_leakage_and_caption_edits_rejected(self):
        parent = self.example(); current = copy.deepcopy(parent)
        current['rows'].append({**current['rows'][0], 'id': 'new', 'split': 'test'})
        with self.assertRaisesRegex(ValueError, 'crosses splits'):
            validate_continuation(current, parent)
        current = copy.deepcopy(parent); current['rows'][0]['text'] = 'edited'
        with self.assertRaisesRegex(ValueError, 'Parent sample changed'):
            validate_continuation(current, parent)
        validate_continuation(parent, parent)

    def test_relation_excludes_self_and_has_finite_gradients(self):
        torch.manual_seed(7)
        raw = torch.randn(6, 5, requires_grad=True)
        images = torch.nn.functional.normalize(raw, dim=-1)
        teacher = images.detach() @ images.detach().T
        original = relational_loss(images, teacher)
        teacher.fill_diagonal_(-999)
        changed = relational_loss(images, teacher)
        self.assertLess(abs(original.item()), 1e-5)
        torch.testing.assert_close(original, changed)
        target = torch.eye(6); loss = relational_loss(images, target)
        loss.backward()
        self.assertTrue(torch.isfinite(raw.grad).all())
        self.assertGreater(raw.grad.norm().item(), 0)

    def test_neighbor_overlap_and_paired_estimate(self):
        x = np.eye(6, dtype=np.float32)
        self.assertEqual(neighborhood_overlap(x, x), 1.)
        result = paired_bootstrap(x, x, x, samples=20)
        for metric in result.values():
            self.assertEqual(metric['delta'], 0.)
            self.assertEqual(metric['query_bootstrap_95ci'], [0., 0.])

    def test_annotation_stops_before_batch_submission_if_account_overdue(self):
        from PIL import Image
        from scripts.benchmark_stickers import annotate
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            Image.new('RGB', (32, 32), 'white').save(root / 'sample.png')
            (root / 'manifest.json').write_text(json.dumps({'items': [
                {'id': f'E{i:04d}', 'image': 'sample.png'} for i in range(6)]}))
            config = {'api_key': 'test', 'model': 'test', 'vision_model': 'test'}
            with patch('cloud_api.load_config', return_value=config), patch('cloud_api.generate', side_effect=ValueError('方舟 API 请求失败：账户欠费')) as call:
                with self.assertRaisesRegex(ValueError, 'preflight failed'):
                    annotate(root)
                self.assertEqual(call.call_count, 1)


if __name__ == '__main__':
    unittest.main()
