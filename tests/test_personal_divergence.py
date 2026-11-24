import json
import unittest
import personal_divergence
import stickers
import test_screenshot_library as fixture


class PersonalD(unittest.TestCase):
    event=fixture.ScreenshotLibrary.event
    tearDown=fixture.ScreenshotLibrary.tearDown
    def setUp(self):
        fixture.ScreenshotLibrary.setUp(self)
        self.c.execute("UPDATE users SET arm='D' WHERE id='b'")
        self.c.execute('ALTER TABLE messages ADD COLUMN interpretation TEXT')
        self.c.execute('ALTER TABLE messages ADD COLUMN created TEXT')
        for uid in ('a','b'):
            self.c.execute('INSERT INTO assets VALUES(?,?,?,?,?)',('img_'+uid,uid,'private_'+uid,'描述_'+uid,'{}'))
        for aid in ('g_one','g_two'):
            self.c.execute('INSERT INTO generic_stickers VALUES(?,?,?,?,?)',(aid,'public_'+aid,json.dumps(dict(description='收到',meaning='收到',situations='回应',avoid='危机')),1,'now'))
            self.c.execute('INSERT INTO sticker_library_metadata VALUES(?,?,?)',(aid,'generic','风格_'+aid))

    def use(self,uid):
        self.c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?,?,?)',('src_'+uid,uid,'s','user','用户原话_'+uid,'img_'+uid,'含义_'+uid,'now'))

    def decision(self,uid,target,mid=None,delivered=True,reference=None):
        mid=mid or 'reply_'+uid;ref=reference or 'img_'+uid
        if delivered:self.c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?,?,?)',(mid,uid,'s','assistant','回应',target,'','now'))
        self.event(self.c,uid,'sticker_selection',{'condition':'D','divergence_policy':'chat-context-only-v3','message':mid,
            'asset_id':target,'reference_asset_id':ref,'reason':'依据_'+uid,
            'candidates':[{'asset_id':target,'description':'公共素材'}],
            'usage_context':{'references':[{'asset_id':ref,'uses':[{'source_message':'src_'+uid,'interpretation':'解释_'+uid}]}]}})

    def test_imported_only_and_non_d_have_no_personal_candidates(self):
        d=personal_divergence.report(self.c,'a')
        self.assertEqual(d['used_reference_count'],0);self.assertEqual(d['adopted'],[]);self.assertIsNone(d['latest'])
        self.c.execute("UPDATE users SET arm='G' WHERE id='b'")
        self.assertIsNone(personal_divergence.report(self.c,'b'))

    def test_two_people_have_separate_evidence_and_collections(self):
        for uid,target in [('a','g_one'),('b','g_two')]:self.use(uid);self.decision(uid,target)
        a=personal_divergence.report(self.c,'a');b=personal_divergence.report(self.c,'b')
        self.assertEqual([x['id'] for x in a['adopted']],['g_one'])
        self.assertEqual([x['id'] for x in b['adopted']],['g_two'])
        self.assertNotIn('用户原话_b',json.dumps(a,ensure_ascii=False));self.assertNotIn('img_b',json.dumps(a))
        self.assertEqual(a['latest']['references'][0]['uses'][0]['source_message'],'src_a')

    def test_unfinished_or_foreign_reference_does_not_create_adoption(self):
        self.use('a');self.use('b')
        self.decision('a','g_one',delivered=False)
        self.decision('a','g_two','wrong-ref',reference='img_b')
        self.assertEqual(personal_divergence.report(self.c,'a')['adopted'],[])
        self.decision('a','g_one','real-reply')
        d=personal_divergence.report(self.c,'a')
        self.assertEqual(len(d['adopted']),1);self.assertEqual(d['adopted'][0]['send_count'],1)

    def test_g_is_baseline_and_d_can_retrieve_public_and_legacy_material(self):
        self.use('a')
        self.c.execute("UPDATE messages SET interpretation='收到',text='收到' WHERE id='src_a'")
        self.c.execute("UPDATE generic_stickers SET semantics=(SELECT semantics FROM generic_stickers WHERE id='g_one') WHERE id='d_one'")
        self.c.execute("""INSERT INTO sticker_intent_reviews SELECT id,'["ack"]','fixture' FROM generic_stickers""")
        g={x['asset_id'] for x in stickers.candidates(self.c,'a','s','G')}
        d={x['asset_id'] for x in stickers.candidates(self.c,'a','s','D')}
        self.assertEqual(g,{'g_one','g_two'});self.assertEqual(d,{'g_one','g_two','d_one'})


if __name__=='__main__':unittest.main()
