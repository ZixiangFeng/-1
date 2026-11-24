import unittest

import numpy as np
import sticker_intent
import sticker_pragmatics as use


class BroadMeaningTests(unittest.TestCase):
    def test_all_legacy_keys_migrate_to_eight_semantic_families(self):
        self.assertEqual(len(use.NAMES),8)
        self.assertTrue(all(use.family(key) in use.NAMES for key in sticker_intent.NAMES))
        self.assertEqual(use.family('ack'),use.family('agree'))
        self.assertEqual(use.family('distress'),use.family('speechless'))
        self.assertEqual(use.family('nonexistent'),'')

    def test_speaker_and_purpose_even_if_a_coarse_label_is_wrong(self):
        reference='自嘲感到孤单无聊，或夸张表达无人理解的空虚感'
        real='一个人待着真寂寞，没人陪我说话'
        self.assertTrue(use.compatible_use('distress', reference, real))
        for different in ['指出对方很变态或不正常。','嗐，真无语','我闲得无聊，出来打发时间',
                          '想一个人安静独处']:
            self.assertFalse(use.compatible_use('distress',reference,different))
        self.assertFalse(use.compatible_use('support','我来安慰你','抱抱我好吗'))
        self.assertFalse(use.compatible_use('daily','我肚子饿','给你尝一口'))
        self.assertFalse(use.compatible_use('joy','这个段子太好笑了','一点都不好笑'))
        self.assertTrue(use.compatible_use('support','鼓励你撑过去','我陪着你，一起加油'))

    def test_distance_only_after_interchangeability_full_group_ties_and_missing(self):
        def run(rows,values,gaps):
            return use.gallery_result(dict(id='q'),rows,['distress']*len(rows),'distress','我觉得孤单',
                                      values,gaps,lambda q,r:None,.60)
        rows=[dict(id=f'img{i:02}',meaning='感到寂寞',quality='clear') for i in range(23)]
        rows.append(dict(id='wrong',meaning='指出对方很变态或不正常',quality='clear'))
        rows.append(dict(id='weak',meaning='孤单',quality='clear'))
        values=np.array([.9]*22+[.61,.99,.30])
        gaps=np.array([.7]*22+[1.06,1.7,1.8])
        result=run(rows,values,gaps)
        self.assertEqual(result['maximum']['id'],'img22')
        self.assertEqual(len(result['divergent']),23)
        self.assertEqual(result['maximum']['meaning_rank'],23)
        self.assertEqual(run(rows[::-1],values[::-1],gaps[::-1])['maximum'],result['maximum'])
        gaps[0]=gaps[22]
        self.assertEqual(run(rows,values,gaps)['maximum']['id'],'img00')
        gaps[1]=np.nan
        self.assertIsNone(run(rows,values,gaps)['maximum'])

    def test_coarse_group_is_not_a_fine_label_gate(self):
        self.assertEqual(use.family('ack'),use.family('agree'))
        self.assertTrue(use.compatible_use('response','收到，我听懂你的安排了','我同意，这样办'))


class ReviewCompatibility(unittest.TestCase):
    def setUp(self):
        import sqlite3
        import generic_stickers
        self.catalog = generic_stickers
        self.c = sqlite3.connect(':memory:')
        self.addCleanup(self.c.close)
        self.c.row_factory = sqlite3.Row
        self.catalog.init(self.c)
        self.c.execute("INSERT INTO generic_stickers VALUES('unit','img','{}',1,'now')")
        self.c.execute("""INSERT INTO sticker_intent_reviews VALUES('unit','["ack"]','old')""")

    def test_saving_displayed_family_preserves_earlier_review_scope(self):
        view=self.catalog.reviewed_catalog(self.c)[0]
        self.assertEqual(view['approved_families'],['response'])
        self.catalog.save_intents(self.c,'unit',['response','daily'],view['review_version'])
        changed=self.catalog.reviewed_catalog(self.c)[0]
        self.assertEqual(changed['approved_groups'],['ack','daily'])
        self.assertEqual(changed['approved_families'],['daily','response'])
        with self.assertRaises(ValueError):
            self.catalog.save_intents(self.c,'unit',['joy'],view['review_version'])

    def test_changing_approval_during_a_reply_invalidates_the_old_winner(self):
        from sticker_identity import _confirmed_maximum
        audit={'selection_rule':sticker_intent.POLICY, 'semantic_policy':use.VERSION,
               'semantic_group':'response','maximum_asset_id':'unit','maximum_style_distance':1.0,
               'semantic_group_size':1,'candidates':[{'asset_id':'unit','review_token':'old'}],
               'style_ranking':[{'id':'unit','distance':1.0}]}
        self.assertTrue(_confirmed_maximum(self.c,'unit',audit))
        self.c.execute("UPDATE sticker_intent_reviews SET token='new' WHERE asset='unit'")
        self.assertFalse(_confirmed_maximum(self.c,'unit',audit))
