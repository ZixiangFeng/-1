import json
import unittest

import numpy as np

import sticker_ranker
import sticker_template
import stickers
import test_divergence as strength
from test_sticker_identity import image


class MetricTests(unittest.TestCase):
    def test_metric_uses_checkpoint_and_never_reads_visual_fields_for_meaning(self):
        m = sticker_ranker.metric()
        gen = np.random.default_rng(8)
        vectors = sticker_ranker.unit(gen.normal(size=(6,512)))
        z = m.project(vectors)
        self.assertFalse(np.allclose(vectors @ vectors.T,z @ z.T))
        self.assertTrue(np.allclose(np.linalg.norm(z,axis=1),1,atol=1e-5))
        class Encoder:
            def encode(self, texts):
                return [vectors[0] for _ in texts]
        def option(aid, desc):
            return {'asset_id':aid,'description':desc,'style':desc,'usages':[{'meaning':'相同的测试用法'}]}
        result = sticker_ranker.ranked_options([option('a','猫照片'),option('b','黑白线稿')],'收到',Encoder())
        self.assertEqual(result[0]['semantic_score'],result[1]['semantic_score'])

    def test_current_text_and_image_only_uses_have_grounded_semantics(self):
        r = {'uses':[{'text':'好的','interpretation':'知道了，确认收到'}],'visible_description':'皱眉的卡通人'}
        self.assertIn('确认收到',sticker_ranker.situated_query('','',r,'user_image'))
        self.assertEqual(sticker_ranker.situated_query('这次我是不同意','明白你的意思',r,'user_image'),
                         '这次我是不同意。明白你的意思')

    def test_unknown_medium_stays_open_and_named_redraw_is_excluded(self):
        r = {'visible_description':'姚明的篮球比赛照片'}
        self.assertTrue(sticker_template.known_match(r,{'description':'姚明笑脸加圣诞帽','style':'黑白线稿'}))
        self.assertFalse(sticker_template.known_match(r,{'description':'动画人物大笑'}))


class LiveTests(unittest.TestCase):
    setUp = strength.StrongDIntegration.setUp
    tearDown = strength.StrongDIntegration.tearDown
    event = strength.StrongDIntegration.event

    def test_real_images_go_through_comparison_and_same_face_is_rejected(self):
        self.c.execute("UPDATE assets SET image=? WHERE id='used'",(image(1),))
        self.c.execute("UPDATE generic_stickers SET image=? WHERE id='d_one'",(image(2),))
        self.c.execute("INSERT INTO sticker_intent_reviews VALUES('d_one','[\"ack\"]','after_image_change')")
        seen = []
        def generate(messages,**kwargs):
            self.assertTrue(kwargs['vision'])
            self.assertTrue(messages[1]['images'][0])
            seen.append(kwargs['schema'])
            selection = strength.choice('d_one','used',self.a,self.b)
            selection['template_relation'] = 'same'
            return json.dumps(selection),{}
        aid,audit = stickers.choose(self.c,'a','s','D','收到','好呀','used',generate,rng=lambda _:0)
        self.assertIsNone(aid)
        self.assertEqual(audit['contrast']['template_check'],'same')
        self.assertEqual(audit['ranker_sha256'],sticker_ranker.metric().digest)
        self.assertIn('template_relation',seen[0]['required'])
