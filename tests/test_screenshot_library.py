import base64
import io
import json
import sqlite3
import unittest
from unittest.mock import patch
from PIL import Image, ImageDraw
import screenshot_library as intake
import generic_stickers
import stickers


def screenshot():
    im=Image.new('RGB',(300,225),'#2c2c2c');d=ImageDraw.Draw(im)
    for r in range(3):
        for col in range(3):
            d.rectangle((10+100*col,10+80*r,89+100*col,min(69+80*r,224)),fill=(90+r*30,110+col*40,160))
    b=io.BytesIO();im.save(b,'PNG')
    return base64.b64encode(b.getvalue()).decode()


class ScreenshotLibrary(unittest.TestCase):
    def setUp(self):
        # These dialogue/intake fixtures deliberately store placeholder image bytes.
        style = patch('sticker_style_choice.distances',side_effect=lambda c,u,r,opts:
                      {o['asset_id']:1.0 for o in opts})
        style.start()
        self.addCleanup(style.stop)
        self.c=sqlite3.connect(':memory:');self.c.row_factory=sqlite3.Row
        self.c.executescript('''PRAGMA foreign_keys=ON;
        CREATE TABLE users(id TEXT PRIMARY KEY,label,arm);
        CREATE TABLE assets(id TEXT PRIMARY KEY,user,image,description,metadata);
        CREATE TABLE messages(id,user,session,role,text,asset);
        CREATE TABLE memories(id,user,expression,meaning,context,status,source,created);
        CREATE TABLE events(id INTEGER PRIMARY KEY,user,kind,payload);
        INSERT INTO users VALUES('a','QA-A','D'),('b','QA-B','M');''')
        intake.init(self.c);generic_stickers.init(self.c);stickers.init(self.c)
        self.c.execute('INSERT INTO generic_stickers VALUES(?,?,?,?,?)',('d_one','image',json.dumps(dict(description='线条猫',meaning='问候',situations='见面',avoid='危机')),1,'now'))
        self.c.execute('INSERT INTO sticker_library_metadata VALUES(?,?,?)',('d_one','divergent','极简线条'))
    def tearDown(self):self.c.close()
    def event(self,c,u,k,p):c.execute('INSERT INTO events(user,kind,payload) VALUES(?,?,?)',(u,k,json.dumps(p)))
    def imported(self):
        d=intake.preview(self.c,'a',screenshot())
        return intake.commit(self.c,'a',d['draft'],[{'index':t['index']} for t in d['tiles'] if t['selected']],self.event)
    def test_detect_partial_and_manual(self):
        im=intake.read_image(screenshot());method,boxes=intake.detect(im)
        self.assertEqual((method,len(boxes)),('background-projection',9))
        self.assertEqual(sum(b['partial'] for b in boxes),3)
        self.assertEqual(len(intake.detect(im,{'columns':3,'rows':3})[1]),9)
        with self.assertRaises(ValueError):intake.detect(im,{'columns':3})
        with self.assertRaises(ValueError):intake.detect(im,{'region':[0,0,999,999]})
    def test_isolation_idempotence_and_no_chat_learning(self):
        d=intake.preview(self.c,'a',screenshot())
        with self.assertRaises(ValueError):intake.commit(self.c,'b',d['draft'],[{'index':0}],self.event)
        result=intake.commit(self.c,'a',d['draft'],[{'index':0},{'index':1}],self.event)
        self.assertEqual(result['imported'],2)
        self.assertTrue(intake.commit(self.c,'a',d['draft'],[{'index':0}],self.event)['replayed'])
        d2=intake.preview(self.c,'a',screenshot())
        self.assertEqual(intake.commit(self.c,'a',d2['draft'],[{'index':0}],self.event)['duplicates'],1)
        for table in ('messages','memories'):self.assertEqual(self.c.execute('SELECT count(*) FROM '+table).fetchone()[0],0)
        self.assertEqual(intake.inventory(self.c,'b')['items'],[])
        self.assertTrue(all(r[0]=='' for r in self.c.execute('SELECT description FROM assets')))
        self.assertEqual(stickers.candidates(self.c,'a','s','M'),[])
    def test_invalid_crop_is_atomic(self):
        d=intake.preview(self.c,'a',screenshot())
        with self.assertRaises(ValueError):intake.commit(self.c,'a',d['draft'],[{'index':0},{'index':1,'box':[0,0,999,90]}],self.event)
        self.assertEqual(self.c.execute('SELECT count(*) FROM assets').fetchone()[0],0)

    def test_import_never_generates_meanings_or_divergence_candidates(self):
        result=self.imported()
        uid='a';aid=result['asset_ids'][0]
        self.c.execute('INSERT INTO sticker_inventory_features VALUES(?,?,?,?,?)',
            (uid,aid,json.dumps({'common_meaning':'wrong default','candidate_ids':['d_one']}),'{}','old'))
        self.c.execute('INSERT INTO sticker_inventory_profiles VALUES(?,?,?,?)',(uid,'old',json.dumps({'analyzed':6,'version':'old'}),'old'))
        answer=intake.analyze_step(self.c,uid,lambda *a,**k:self.fail('import must never invoke AI'),self.event)
        self.assertEqual(answer['policy'],'storage-only')
        d=intake.inventory(self.c,uid)
        self.assertIsNone(d['profile']);self.assertEqual(d['used_assets'],0)
        self.assertTrue(all(x['feature'] is None and not x['usages'] for x in d['items']))
        self.assertEqual(stickers.candidates(self.c,uid,'s','D'),[])
        self.assertIsNone(intake.context(self.c,uid,aid))

    def test_actual_dialogue_is_used_and_d_references_are_validated(self):
        aid=self.imported()['asset_ids'][0]
        self.c.execute('ALTER TABLE messages ADD COLUMN interpretation TEXT')
        self.c.execute('ALTER TABLE messages ADD COLUMN created TEXT')
        self.c.execute('UPDATE assets SET description=? WHERE id=?',('彩色写实人物手举到额边，无字',aid))
        self.c.execute('UPDATE generic_stickers SET semantics=? WHERE id=?',
            (json.dumps(dict(description='黑白简笔猫咪，无字',meaning='收到',situations='回应提醒',avoid='危机')),'d_one'))
        self.c.execute("INSERT INTO sticker_intent_reviews VALUES('d_one','[\"ack\"]','fixture')")
        self.c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?,?,?)',('prior','a','s','assistant','记得保存文件',None,'','now'))
        self.c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?,?,?)',('send','a','s','user','',aid,'收到、听懂提醒','now'))
        self.c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?,?,?)',('foreign','b','s','user','私密内容',aid,'wrong','now'))
        d=intake.inventory(self.c,'a');use=d['items'][0]['usages'][0]
        self.assertEqual(use['interpretation'],'收到、听懂提醒')
        self.assertEqual(use['preceding'][0]['text'],'记得保存文件')
        self.assertEqual(d['total_uses'],1)
        calls=[]
        def choose(messages,**kwargs):
            from test_divergence import choice, profile
            calls.append(messages)
            return json.dumps(choice('d_one',aid,'彩色写实人物手举到额边，无字','黑白简笔猫咪，无字',
                profile('photo','human','none','color','彩色写实人物手举到额边，无字'),
                profile('line_drawing','animal','none','monochrome','黑白简笔猫咪，无字'))),{}
        selected,audit=stickers.choose(self.c,'a','s','D','','好呀',aid,choose,rng=lambda n:0)
        self.assertEqual(selected,'d_one');self.assertEqual(audit['source_messages'],['send'])
        self.assertEqual(audit['divergence_policy'],'chat-context-only-v3')
        self.assertNotIn('wrong default',str(calls))
        self.assertNotIn('私密内容',str(calls))
        self.assertIn('收到',str(calls))
        with self.assertRaises(ValueError):
            stickers.choose(self.c,'a','s','D','','好呀',aid,lambda *a,**k:(json.dumps({'asset_id':'d_one','reference_asset_id':'unseen','reason':'invalid'}),{}),rng=lambda n:0)

    def test_legacy_d_mapping_is_not_shown_as_a_live_decision(self):
        aid=self.imported()['asset_ids'][0]
        self.event(self.c,'a','sticker_selection',{'asset_id':'d_one','reference_asset_id':aid,'divergence_policy':'inventory-plus-observed-v2'})
        self.assertEqual(intake.inventory(self.c,'a')['items'][0]['decisions'],[])

if __name__=='__main__':unittest.main()
