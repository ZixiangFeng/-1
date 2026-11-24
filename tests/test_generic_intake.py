import json
import unittest
from unittest.mock import Mock
import generic_intake
import generic_stickers
import stickers
import test_screenshot_library as fixture


class GenericIntake(unittest.TestCase):
    event=fixture.ScreenshotLibrary.event
    tearDown=fixture.ScreenshotLibrary.tearDown
    def setUp(self):
        fixture.ScreenshotLibrary.setUp(self)
        generic_intake.init(self.c)
        self.generate=Mock(return_value=(json.dumps(dict(description='卡通猫',meaning='可能表示收到',situations='回应提醒',avoid='需明确承诺时',style='二维彩色卡通、无字')),{}))

    def draft(self,pool='divergent'):
        return generic_intake.preview(self.c,{'image':fixture.screenshot(),'pool':pool})

    def send(self,d,index=0,**kw):
        return generic_intake.import_tile(self.c,dict(draft=d['draft'],index=index,**kw),self.generate,self.event)

    def test_preview_then_individual_analysis_duplicate_and_no_personal_writes(self):
        d=self.draft();self.generate.assert_not_called()
        self.assertEqual(sum(t['selected'] for t in d['tiles']),6)
        result=self.send(d,pool='generic')
        self.assertEqual(result['pool'],'divergent');self.assertTrue(result['id'].startswith('d_'))
        self.assertEqual(self.generate.call_count,1)
        row=self.c.execute('SELECT * FROM sticker_library_metadata WHERE asset=?',(result['id'],)).fetchone()
        self.assertIn('二维',row['style'])
        self.c.execute('UPDATE generic_stickers SET enabled=0 WHERE id=?',(result['id'],))
        repeat=self.send(self.draft());self.assertTrue(repeat['duplicate'])
        self.assertEqual(self.generate.call_count,1)
        self.assertEqual(self.c.execute('SELECT enabled FROM generic_stickers WHERE id=?',(result['id'],)).fetchone()[0],0)
        for table in ('messages','assets','memories'):
            self.assertEqual(self.c.execute('SELECT count(*) FROM '+table).fetchone()[0],0)
        self.assertEqual(stickers.candidates(self.c,'a','s','D'),[])

    def test_failure_retry_preserves_completed_tiles(self):
        d=self.draft('generic');first=self.send(d)
        good=self.generate.return_value
        self.generate.side_effect=ValueError('analysis unavailable')
        with self.assertRaises(ValueError):self.send(d,1)
        self.assertIsNotNone(self.c.execute('SELECT 1 FROM generic_stickers WHERE id=?',(first['id'],)).fetchone())
        self.generate.side_effect=None;self.generate.return_value=good
        next_result=self.send(d,1)
        self.assertNotEqual(first['id'],next_result['id'])
        self.assertFalse(next_result['duplicate'])

    def test_invalid_and_expired_drafts_fail_before_model(self):
        d=self.draft()
        for index,box in [(999,None),(True,None),(0,[0,0,999,90]),(0,[2,2,2,2])]:
            with self.assertRaises(ValueError):self.send(d,index,**({'box':box} if box else {}))
        self.c.execute("UPDATE generic_screenshot_drafts SET created='2000-01-01T00:00:00+00:00'")
        with self.assertRaises(ValueError):self.send(d)
        self.generate.assert_not_called()

    def test_g_and_shared_material_pools_are_independent(self):
        a=self.send(self.draft('generic'));b=self.send(self.draft('divergent'))
        self.assertNotEqual(a['id'],b['id'])
        self.assertEqual([x['asset_id'] for x in generic_stickers.candidates(self.c,set(),pool='generic')],[a['id']])

    def test_candidates_are_relative_to_each_users_actual_sends(self):
        d=self.send(self.draft())['id']
        data=self.c.execute('SELECT image FROM generic_stickers WHERE id=?',(d,)).fetchone()[0]
        self.c.execute('ALTER TABLE messages ADD COLUMN interpretation TEXT')
        self.c.execute('ALTER TABLE messages ADD COLUMN created TEXT')
        for uid,aid,image,desc in [('a','a_used',data,'二维猫'),('b','b_used','another-image','写实人物'),('b','b_inventory',data,'')]:
            self.c.execute('INSERT INTO assets VALUES(?,?,?,?,?)',(aid,uid,image,desc,'{}'))
        for uid,aid in [('a','a_used'),('b','b_used')]:
            self.c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?,?,?)',('m_'+uid,uid,'s','user','收到',aid,'回应提醒','now'))
        a={x['asset_id'] for x in stickers.candidates(self.c,'a','s','D')}
        b={x['asset_id'] for x in stickers.candidates(self.c,'b','s','D')}
        self.assertNotIn(d,a);self.assertNotIn(d,b)
        # B owning the image vetoes reuse, but is still NOT semantic usage evidence.
        self.assertEqual(stickers.expression_usage.context(self.c,'b')['references'][0]['asset_id'],'b_used')


if __name__=='__main__':unittest.main()
