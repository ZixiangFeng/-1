import json
import unittest
from unittest.mock import patch

import generic_stickers
import sticker_intent
import stickers
from research.sticker_encoder.reviewed_gallery import reviewed_rows
import test_divergence as fixtures

# Reuse the actual chat selection entry point and a researcher-owned database.
class GroupMaximum(unittest.TestCase):
    setUp=fixtures.StrongDIntegration.setUp
    tearDown=fixtures.StrongDIntegration.tearDown
    event=fixtures.StrongDIntegration.event

    def select(self,generate):
        return stickers.choose(self.c,'a','s','D','收到','好呀','used',generate,rng=lambda n:0)

    def review(self,asset,groups):
        rows={r['id']:r for r in generic_stickers.reviewed_catalog(self.c)}
        return generic_stickers.save_intents(self.c,asset,groups,rows[asset]['review_version'])

    def add(self,asset,groups):
        self.c.execute('INSERT INTO generic_stickers SELECT ?,?,semantics,1,created FROM generic_stickers WHERE id=?',
                       (asset,'test-bytes-'+asset,'d_one'))
        self.c.execute("INSERT INTO sticker_library_metadata VALUES(?,'generic','黑白简笔人物')",(asset,))
        self.review(asset,groups)

    def test_farthest_beyond_eighteen_and_wrong_group_never_reaches_selector(self):
        for i in range(23):
            self.add('extra_%02d'%i,['ack'])
        self.add('wrong',['laugh'])
        seen=[]
        def distances(c,user,ref,options):
            seen.extend(o['asset_id'] for o in options)
            return {o['asset_id']:1.02 if o['asset_id']=='extra_22' else .77 for o in options}
        def generator(messages,**kwargs):
            packet=json.loads(messages[1]['content'])
            self.assertEqual([o['asset_id'] for o in packet['candidates']],['extra_22'])
            return json.dumps(fixtures.choice('extra_22','used',self.a,self.b)),{}
        with patch('sticker_style_choice.distances',side_effect=distances):
            aid,audit=self.select(generator)
            self.assertEqual((aid,len(seen)),('extra_22',24))
            self.assertNotIn('wrong',seen)
            self.assertEqual(audit['maximum_asset_id'],aid)
            self.assertTrue(audit['group_maximum_verified'])
            self.assertIsNone(stickers.sticker_identity.guard_reply(self.c,'a','D','d_one',audit)[0])
            incomplete=dict(audit,style_ranking=audit['style_ranking'][:1])
            self.assertIsNone(stickers.sticker_identity.guard_reply(self.c,'a','D',aid,incomplete)[0])
            veto,a=self.select(lambda *args,**kw:(json.dumps(fixtures.choice('d_one','used',self.a,self.b)),{}))
            self.assertIsNone(veto)
            self.assertEqual(a['maximum_asset_id'],'extra_22')

    def test_unreviewed_and_revised_or_stale_memberships_do_not_qualify(self):
        self.add('new',[])
        old={r['id']:r for r in generic_stickers.reviewed_catalog(self.c)}['new']['review_version']
        self.review('new',['ack','grin'])
        with self.assertRaises(ValueError):
            generic_stickers.save_intents(self.c,'new',['laugh'],old)
        self.c.execute("UPDATE generic_stickers SET image='changed' WHERE id='new'")
        options=stickers.candidates(self.c,'a','s','D','used','收到')
        self.assertEqual([r['asset_id'] for r in options],['d_one'])
        self.review('d_one',[])
        self.assertIsNone(self.select(lambda *a,**k:self.fail('no valid group'))[0])

    def test_equal_or_nearly_equal_distances_and_incomplete_group(self):
        first={'id':'a','style_distance':1.0000001}
        second={'id':'b','style_distance':1.0000002}
        self.assertEqual(sticker_intent.descending_style([first,second])[0]['id'],'b')
        second['style_distance']=first['style_distance']
        self.assertEqual(sticker_intent.descending_style([second,first])[0]['id'],'a')
        with patch('sticker_style_choice.distances',return_value={}):
            aid,audit=self.select(lambda *a,**k:self.fail('partial ranking is forbidden'))
            self.assertIsNone(aid)
            self.assertEqual(audit['error_stage'],'group_style')

    def test_same_medium_can_be_the_global_maximum(self):
        self.add('other_photo',['ack'])
        self.c.execute("UPDATE assets SET description='真人照片' WHERE id='used'")
        self.c.execute("UPDATE sticker_library_metadata SET style='另一张真人照片' WHERE asset='other_photo'")
        # The style edit may invalidate the researcher review.
        self.review('other_photo',['ack'])
        def checker(messages,**kwargs):
            packet=json.loads(messages[1]['content'].split('\n')[0])
            self.assertEqual([o['asset_id'] for o in packet['candidates']],['other_photo'])
            return json.dumps({'asset_id':'','reason':'本轮不选图'}),{}
        with patch('sticker_style_choice.distances',return_value={'other_photo':1.21,'d_one':.91}):
            aid,audit=self.select(checker)
        self.assertIsNone(aid)
        self.assertEqual(audit['maximum_asset_id'],'other_photo')

    def test_known_correction_is_bound_to_one_public_image(self):
        review={'tag':dict(image_sha256='123',previous_meaning='笑',meaning='无语',source='研究者',intent='speechless')}
        rows=[dict(id='tag',image_sha256='123',meaning='笑'),dict(id='other',image_sha256='321',meaning='笑')]
        result=reviewed_rows(rows,review)
        self.assertEqual([r['meaning'] for r in result],['无语','笑'])
        rows[0]['image_sha256']='unexpected'
        with self.assertRaises(ValueError): reviewed_rows(rows,review)
