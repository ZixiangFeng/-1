import sqlite3
import unittest
from datetime import datetime, timezone

import behavior
import expression_reuse as reuse


class ExpressionReuse(unittest.TestCase):
    def setUp(self):
        self.c = sqlite3.connect(':memory:')
        self.c.row_factory = sqlite3.Row
        self.c.executescript('''CREATE TABLE users(id,label,arm,is_test);
          CREATE TABLE messages(id PRIMARY KEY,user,session,role,text,asset,created);
          CREATE TABLE surveys(user,wave,created); CREATE TABLE assets(id,user,description);''')
        behavior.init(self.c); reuse.init(self.c)
        self.start = datetime(2026,1,1,tzinfo=behavior.TZ).timestamp()
        self.c.executemany('INSERT INTO users VALUES(?,?,?,?)',[('a','P001','M',0),('b','P002','G',0)])
        self.c.execute('INSERT INTO assets VALUES(?,?,?)',('x','a','图'))

    def tearDown(self):self.c.close()
    def iso(self,s):return datetime.fromtimestamp(self.start+s,timezone.utc).isoformat()
    def msg(self,mid,s,role='user',asset=None,user='a',session='one'):
        self.c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?,?)',(mid,user,session,role,'话',asset,self.iso(s)))
    def view(self,mid,s,page='p',asset='x'):
        self.c.execute('INSERT INTO image_views VALUES(?,?,?,?,?)',('a',page,mid,asset,self.iso(s)))
    def report(self,s=29*86400):return reuse.report(self.c,'a',clock=self.start+s)

    def test_week_one_robot_week_four_user_no_short_time_or_send_cap(self):
        self.msg('u1',0,asset='x');self.msg('bot',86400,'assistant','x');self.view('bot',86410)
        for i in range(60):self.msg('text'+str(i),2*86400+i)
        self.msg('u2',23*86400,asset='x',session='another');self.msg('u3',27*86400,asset='x')
        d=self.report();a=d['assets'][0]
        self.assertEqual(a['user_reuses_after_robot'],2)
        self.assertEqual(a['user_reuses_after_seen'],2)
        self.assertEqual(d['reuse_events'][0]['sends_from_latest_robot'],61)
        self.assertEqual(d['weekly'][3]['user_reuses_after_robot'],2)
        self.assertEqual(d['period']['status'],'completed')
        self.assertIsNone(d['period']['send_limit'])

    def test_multiple_bots_do_not_reset_or_duplicate_user_reuses(self):
        self.msg('u1',0,asset='x');self.msg('b1',10,'assistant','x');self.msg('b2',20,'assistant','x')
        self.msg('u2',30,asset='x');self.msg('u3',40,asset='x');self.msg('b3',50,'assistant','x');self.msg('u4',60,asset='x')
        d=self.report();a=d['assets'][0]
        self.assertEqual((a['assistant_reuse_sends'],a['user_reuses_after_robot']),(3,3))
        self.assertEqual([e['latest_assistant_message_id'] for e in d['reuse_events']],['b2','b2','b3'])
        self.assertEqual(len({e['user_message_id'] for e in d['reuse_events']}),3)
        self.assertEqual(d['summary']['user_reused_images'],1)
        self.assertEqual(d['summary']['asset_reuse_rate'],1)

    def test_historical_sequence_counts_without_fabricating_views(self):
        self.msg('u1',0,asset='x');self.msg('b1',10,'assistant','x');self.msg('u2',10*86400,asset='x')
        d=self.report()
        self.assertEqual(d['summary']['user_reuses_after_robot'],1)
        self.assertIsNone(d['assets'][0]['user_reuses_after_seen'])
        self.assertIsNone(d['summary']['user_reuses_after_seen'])
        self.assertFalse(d['reuse_events'][0]['has_prior_image_view'])

    def test_per_asset_before_after_totals_and_coverage_denominator(self):
        self.msg('u1',0,asset='x');self.msg('u2',1,asset='x');self.msg('b1',10,'assistant','x');self.msg('u3',20,asset='x')
        self.msg('y1',25,asset='y');self.msg('by',30,'assistant','y');self.msg('z1',40,asset='z')
        d=self.report();a={a['asset_id']:a for a in d['assets']}
        self.assertEqual((a['x']['user_sends_before_first_robot'],a['x']['user_reuses_after_robot']),(2,1))
        self.assertEqual(d['summary']['robot_reused_images'],2)
        self.assertEqual(d['summary']['user_reused_images'],1)
        self.assertEqual(d['summary']['asset_reuse_rate'],0.5)
        for r in a.values():self.assertEqual(r['user_sends'],r['user_sends_before_first_robot']+r['user_reuses_after_robot'])

    def test_observation_ongoing_then_completed_and_day29_excluded(self):
        self.msg('u1',0,asset='x');self.msg('b1',10,'assistant','x');self.msg('u28',28*86400-1,asset='x');self.msg('u29',28*86400,asset='x')
        self.assertEqual(self.report(5*86400)['period']['status'],'ongoing')
        self.assertEqual(self.report(5*86400)['summary']['user_reuses_after_robot'],0)
        d=self.report();self.assertEqual(d['summary']['user_reuses_after_robot'],1)
        self.assertNotIn('u29',{r['message_id'] for r in d['timeline']})

    def test_weekly_totals_match_whole_study(self):
        self.msg('u1',0,asset='x');self.msg('b1',1,'assistant','x')
        for w in range(4):self.msg('hit'+str(w),w*7*86400+10,asset='x')
        d=self.report()
        for weekly_key,total_key in [('user_image_sends','user_sends'),('assistant_reuse_sends','assistant_reuse_sends'),('user_reuses_after_robot','user_reuses_after_robot')]:
            self.assertEqual(sum(w[weekly_key] for w in d['weekly']),d['summary'][total_key])
        self.assertEqual(self.report(1*86400)['weekly'][1]['observed'],False)

    def test_participant_isolation_imports_and_bot_first(self):
        self.msg('text',0);self.msg('bot',10,'assistant','x');self.msg('other',1,asset='x',user='b')
        self.msg('first',20,asset='x');self.msg('again',30,asset='x')
        d=self.report();self.assertEqual(d['summary']['assistant_reuse_sends'],0)
        self.assertEqual(d['summary']['user_reuses_after_robot'],0)
        self.assertIsNone(d['summary']['asset_reuse_rate'])
        self.assertTrue(all(r['user']=='a' for r in d['timeline']))

    def test_late_view_does_not_relabel_past_user_use(self):
        self.msg('u1',0,asset='x');self.msg('b1',10,'assistant','x');self.view('b1',40)
        self.msg('u2',30,asset='x');self.msg('u3',50,asset='x')
        d=self.report();self.assertEqual(d['assets'][0]['user_reuses_after_seen'],1)
        self.assertEqual([e['has_prior_image_view'] for e in d['reuse_events']],[False,True])

    def test_old_visible_bot_still_qualifies_when_latest_bot_unconfirmed(self):
        self.msg('u1',0,asset='x');self.msg('b1',10,'assistant','x');self.view('b1',15)
        self.msg('b2',20,'assistant','x');self.msg('u2',30,asset='x')
        e=self.report()['reuse_events'][0]
        self.assertEqual(e['latest_assistant_message_id'],'b2')
        self.assertEqual(e['latest_seen_assistant_message_id'],'b1')

    def test_receipt_time_takes_priority_and_views_dedup(self):
        self.msg('u1',0,asset='x');self.msg('b1',10,'assistant','x');self.view('b1',20);self.view('b1',28,page='other')
        self.msg('u2',100,asset='x');behavior.record_send(self.c,'a','u2',self.iso(35),{})
        e=self.report()['reuse_events'][0]
        self.assertEqual(e['seconds_from_latest_robot'],25)
        self.assertEqual(e['seconds_from_latest_view'],15)
        self.assertEqual(self.report()['assets'][0]['confirmed_robot_sends'],1)

    def test_observation_auth_and_idempotence(self):
        self.msg('b1',10,'assistant','x');data={'page':'p','message':'b1','asset':'x'}
        reuse.observe(self.c,'a',data);reuse.observe(self.c,'a',data)
        self.assertEqual(self.c.execute('SELECT count(*) FROM image_views').fetchone()[0],1)
        for u,change in [('b',{}),('a',{'asset':'bad'}),('a',{'message':'missing'}),('a',{'page':''})]:
            with self.assertRaises(ValueError):reuse.observe(self.c,u,{**data,**change})

    def test_empty_and_unknown_user(self):
        d=self.report();self.assertEqual(d['assets'],[])
        self.assertEqual(d['period']['status'],'not_started')
        self.assertTrue(all(not w['observed'] for w in d['weekly']))
        with self.assertRaises(ValueError):reuse.report(self.c,'missing')
