import unittest
from research.sticker_encoder.broad import broad_rank


class BroadPoolTests(unittest.TestCase):
    def setUp(self):
        self.q={'id':'a','medium':'photo','pack':'a','families':[],'intent_groups':['ack']}
        self.c={'id':'b','medium':'line','pack':'b','families':[],'intent_groups':['ack'],
                'semantic_score':.6,'context_score':.6,'style_distance':.9}

    def test_coarse_acknowledgement_does_not_need_previous_exact_thresholds(self):
        self.assertEqual(len(broad_rank(self.q,[self.c])['stage2']),1)

    def test_different_meaning_pool_still_excluded(self):
        self.assertFalse(broad_rank(self.q,[dict(self.c,intent_groups=['reject'])])['stage2'])

    def test_same_source_and_medium_still_excluded(self):
        for key,value in [('pack','a'),('medium','photo'),('medium','unknown')]:
            self.assertFalse(broad_rank(self.q,[dict(self.c,**{key:value})])['stage2'])

    def test_explicit_current_intent_overrides_inventory_pool(self):
        self.assertFalse(broad_rank(self.q,[self.c],query_groups=['sad'])['stage2'])

    def test_no_user_owned_or_used_stickers(self):
        self.assertFalse(broad_rank(self.q,[self.c],excluded_ids=['b'])['stage2'])

    def test_semantic_window_precedes_style_ranking(self):
        low=dict(self.c,id='low',semantic_score=.1,context_score=.1,style_distance=1.8)
        result=broad_rank(self.q,[self.c,low],semantic_window=1)
        self.assertEqual(result['stage2'][0]['id'],'b')


if __name__=='__main__':unittest.main()
