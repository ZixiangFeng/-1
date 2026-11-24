import json
import sqlite3
import unittest
import learning
from cloud_api import generate
from test_cloud_api import Opener

class LearningTests(unittest.TestCase):
    def setUp(self):
        self.c = sqlite3.connect(':memory:')
        self.c.row_factory = sqlite3.Row
        self.c.executescript('CREATE TABLE memories(id,user,expression,meaning,context,status,source,created); CREATE TABLE events(user,kind,payload,created);')
        learning.init(self.c)
        self.samples = [{'id':'m1','text':'又改论文了🙂','expressions':['emoji:🙂'],'image_description':''}, {'id':'m2','text':'我是开心，老师认可我了','expressions':[],'image_description':''}]
        self.item = dict(source='m1',expression='emoji:🙂',meaning='开心',context='老师认可论文',evidence='我是开心',evidence_message='m2')

    def tearDown(self):
        self.c.close()

    def test_automatic_learning_and_revision_no_confirmation(self):
        learning.apply(self.c,'u','m2',self.samples,{'observations':[self.item]})
        self.assertEqual(self.c.execute("SELECT count(*) FROM memories WHERE status='inferred'").fetchone()[0],1)
        learning.apply(self.c,'u','m2',self.samples,{'observations':[dict(self.item,meaning='高兴得到认可')]})
        self.assertEqual(self.c.execute("SELECT count(*) FROM memories WHERE status='inferred'").fetchone()[0],1)
        self.assertEqual(self.c.execute("SELECT count(*) FROM memories WHERE status='superseded'").fetchone()[0],1)
        self.assertEqual(self.c.execute('SELECT count(*) FROM learning_evidence').fetchone()[0],2)

    def test_fabricated_or_foreign_evidence_rejected_before_mutation(self):
        for item in [dict(self.item,evidence='你很无奈'),dict(self.item,source='other-user'),dict(self.item,expression='emoji:😭')]:
            with self.assertRaises(ValueError):
                learning.apply(self.c,'u','m2',self.samples,{'observations':[item]})
        self.assertEqual(self.c.execute('SELECT count(*) FROM memories').fetchone()[0],0)

    def test_abstention_retracts_only_own_recent_hypothesis(self):
        learning.apply(self.c,'u','m2',self.samples,{'observations':[self.item]})
        learning.apply(self.c,'other','m2',self.samples,{'observations':[]})
        self.assertEqual(self.c.execute("SELECT count(*) FROM memories WHERE status='inferred'").fetchone()[0],1)
        learning.apply(self.c,'u','m2',self.samples,{'observations':[]})
        self.assertEqual(self.c.execute("SELECT count(*) FROM memories WHERE status='inferred'").fetchone()[0],0)

    def test_cloud_image_wire_and_required_vision_model(self):
        config={'api_key':'TEST','model':'text','vision_model':'vision'}
        op=Opener({'choices':[{'message':{'content':'笑脸'},'finish_reason':'stop'}]})
        generate(config,[{'role':'user','content':'描述','images':['YWJj']}],vision=True,opener=op)
        body=json.loads(op.request.data)
        self.assertEqual(body['model'],'vision')
        self.assertEqual(body['messages'][0]['content'][1]['image_url']['url'],'data:image/jpeg;base64,YWJj')
        with self.assertRaises(ValueError):
            generate(dict(config,vision_model=''),[],vision=True,opener=op)

    def test_prior_assistant_context_can_support_image_only_acknowledgement(self):
        samples=[{'id':'a1','role':'assistant','text':'记得把文件保存后再关闭。','expressions':[]},
                 {'id':'u1','role':'user','text':'','expressions':['image:salute']}]
        item=dict(source='u1',expression='image:salute',meaning='收到并听懂提醒',context='回应保存文件的提醒',
                  evidence='把文件保存后再关闭',evidence_message='a1')
        learning.apply(self.c,'u','u1',samples,{'observations':[item]})
        row=self.c.execute('SELECT meaning,status FROM memories').fetchone()
        self.assertEqual(tuple(row),('收到并听懂提醒','inferred'))
        samples.append({'id':'a2','role':'assistant','text':'你看起来很头疼','expressions':[]})
        with self.assertRaises(ValueError):
            learning.apply(self.c,'u','u1',samples,{'observations':[dict(item,evidence_message='a2',evidence='很头疼')]})
        with self.assertRaises(ValueError):
            learning.apply(self.c,'u','u1',samples,{'observations':[dict(item,source='a1')]})

    def test_learning_excludes_generated_reply_after_current_user_send(self):
        self.c.executescript('CREATE TABLE messages(id,user,session,role,text,asset,expressions,interpretation,created); CREATE TABLE assets(id,user,description);')
        for mid,role,text in [('before','assistant','先保存文件'),('image','user',''),('reply','assistant','你是不是头疼')]:
            self.c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?)',(mid,'u','s',role,text,None,'["image:salute"]' if mid=='image' else '[]','','now'))
        self.c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?)',('foreign','other','s','user','私人内容',None,'[]','','now'))
        samples=learning.collect(self.c,'u','s','image')
        self.assertEqual([x['id'] for x in samples],['before','image'])
