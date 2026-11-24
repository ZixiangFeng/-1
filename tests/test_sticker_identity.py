import base64
import io
import json
import sqlite3
import unittest
from unittest.mock import Mock

from PIL import Image, ImageDraw
import generic_stickers
import sticker_identity as identity
import stickers


def image(kind=0, size=256, fmt='PNG', quality=85):
    im=Image.new('RGB',(256,256),'white');d=ImageDraw.Draw(im)
    if kind==0:
        d.ellipse((28,25,225,223),fill='orange',outline='black',width=4)
        d.ellipse((75,70,90,90),fill='black');d.ellipse((166,70,181,90),fill='black')
        d.arc((78,98,180,180),0,180,fill='black',width=6)
    elif kind==1:
        d.rectangle((20,50,235,205),fill='blue');d.line((20,50,235,205),fill='red',width=16)
    else:
        d.rectangle((0,0,256,256),fill='pink' if kind==2 else 'green')
    im=im.resize((size,size),Image.Resampling.LANCZOS)
    b=io.BytesIO();im.save(b,format=fmt,quality=quality)
    return base64.b64encode(b.getvalue()).decode()


class Identity(unittest.TestCase):
    def setUp(self):
        self.c=sqlite3.connect(':memory:');self.c.row_factory=sqlite3.Row
        self.c.executescript('''CREATE TABLE assets(id,user,image,description,metadata);
            CREATE TABLE messages(id,user,session,role,text,asset,interpretation,created);
            CREATE TABLE memories(id,user,expression,meaning,context,status,source,created);
            CREATE TABLE events(id INTEGER PRIMARY KEY,user,kind,payload);''')
        generic_stickers.init(self.c);stickers.init(self.c)
        for aid,encoded in [('g_same',image()),('g_copy',image(size=90,fmt='JPEG',quality=60)),('g_other',image(1))]:
            self.c.execute('INSERT INTO generic_stickers VALUES(?,?,?,?,?)',(aid,encoded,json.dumps(dict(description='贴纸',meaning='收到',situations='回应',avoid='危机')),1,'now'))
            self.c.execute('INSERT INTO sticker_library_metadata VALUES(?,?,?)',(aid,'generic','彩色卡通'))
            self.c.execute('INSERT INTO sticker_intent_reviews VALUES(?,?,?)',(aid,json.dumps(['ack']),'fixture'))
        self.own('a','personal',image())

    def tearDown(self):self.c.close()

    def own(self,user,aid,encoded,meta=None):
        self.c.execute('INSERT INTO assets VALUES(?,?,?,?,?)',(aid,user,encoded,'',json.dumps(meta or {})))

    def send(self,user='a',aid='personal'):
        self.c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?,?,?)',('sent-'+user,user,'old','user','收到啦',aid,'收到提醒','now'))

    def test_same_bytes_pixels_and_compressed_scaled_copies(self):
        self.assertEqual(identity.same_image(identity.fingerprint(image()),identity.fingerprint(image())),'same_file')
        for size,quality in [(256,85),(90,60),(768,80)]:
            self.assertIsNotNone(identity.same_image(identity.fingerprint(image()),identity.fingerprint(image(size=size,fmt='JPEG',quality=quality))))
        self.assertIsNone(identity.same_image(identity.fingerprint(image()),identity.fingerprint(image(1))))
        self.assertIsNone(identity.same_image(identity.fingerprint(image(2)),identity.fingerprint(image(3))))

    def test_inventory_excludes_without_semantic_learning_and_is_user_relative(self):
        blocked=identity.exclusions(self.c,'a')
        self.assertEqual(set(blocked),{'g_same','g_copy'})
        self.assertEqual({x['asset_id'] for x in stickers.candidates(self.c,'a','s','G')},{'g_other'})
        self.assertEqual(len(stickers.candidates(self.c,'b','s','G')),3)
        self.assertEqual(stickers.candidates(self.c,'a','s','D'),[])
        for table in ('messages','memories','events'):
            self.assertEqual(self.c.execute('SELECT count(*) FROM '+table).fetchone()[0],0)
        self.assertEqual(self.c.execute('SELECT description FROM assets').fetchone()[0],'')

    def test_d_uses_actual_send_but_m_can_still_reuse(self):
        self.send()
        self.assertEqual({x['asset_id'] for x in stickers.candidates(self.c,'a','new','D')},{'g_other'})
        self.assertEqual([x['asset_id'] for x in stickers.candidates(self.c,'a','new','M')],['personal'])
        self.assertEqual(stickers.candidates(self.c,'a','new','T'),[])

    def test_ownership_changes_invalidate_precomputed_options(self):
        options=stickers.candidates(self.c,'b','s','G')
        self.own('b','new_import',image())
        def generate(messages,**kwargs):
            supplied=json.loads(messages[-1]['content'])['candidates']
            self.assertEqual([x['asset_id'] for x in supplied],['g_other'])
            return json.dumps({'asset_id':'g_other','reason':'收到'}),{}
        aid,audit=stickers.choose(self.c,'b','s','G','好','好',None,generate,rng=lambda _:0,options=options)
        self.assertEqual(aid,'g_other');self.assertTrue(audit['personal_exclusion_verified'])
        self.assertEqual(set(audit['personal_exclusions']),{'g_same','g_copy'})

    def test_final_guard_blocks_direct_private_and_changed_image_with_source(self):
        self.own('a','saved',image(3),{'source_asset':'g_other','saved_from_chat':True})
        for arm in ('G','D'):
            for aid in ('g_same','g_copy','g_other','personal','unknown'):
                result,audit=identity.guard_reply(self.c,'a',arm,aid,{})
                self.assertIsNone(result);self.assertEqual(audit['blocked_asset_id'],aid)
        self.assertEqual(identity.guard_reply(self.c,'b','G','g_same',{})[0],'g_same')
        self.assertEqual(identity.guard_reply(self.c,'a','M','personal',{})[0],'personal')

    def test_no_fallback_to_personal_image_when_all_options_excluded(self):
        self.c.execute("UPDATE generic_stickers SET enabled=0 WHERE id='g_other'")
        gen=Mock(side_effect=AssertionError('must not call selector'))
        for arm in ('G','D'):
            self.send() if arm=='D' else None
            aid,audit=stickers.choose(self.c,'a','s',arm,'好','好',None,gen,rng=lambda _:0)
            self.assertIsNone(aid);self.assertEqual(audit['candidates'],[])
        gen.assert_not_called()

    def test_filter_before_top_k_and_pool_rules(self):
        class Store:
            def upsert(inner,docs):
                self.assertEqual([d['id'] for d in docs],['g_other'])
            def search(inner,*args):
                return [{'id':x} for x in ('g_same','g_copy','g_other')][:args[-1]]
        self.assertEqual([x['asset_id'] for x in stickers.candidates(self.c,'a','s','G',store=Store())],['g_other'])
        self.c.execute("UPDATE sticker_library_metadata SET pool='divergent' WHERE asset='g_other'")
        self.assertIsNone(identity.guard_reply(self.c,'a','G','g_other',{})[0])
        self.assertIsNone(identity.guard_reply(self.c,'a','D','g_other',{})[0])


if __name__=='__main__':unittest.main()
