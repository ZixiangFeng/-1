import sqlite3,unittest,json
from datetime import datetime,timedelta,timezone
import behavior

class Behavior(unittest.TestCase):
 def setUp(self):
  self.c=sqlite3.connect(':memory:');self.c.row_factory=sqlite3.Row
  self.c.executescript('''CREATE TABLE users(id,label,arm,is_test,created);CREATE TABLE accounts(user,username);CREATE TABLE messages(id PRIMARY KEY,user,session,role,text,asset,created);CREATE TABLE surveys(user,wave,created);CREATE TABLE assets(id,user,description);CREATE TABLE visits(id,user,session,greeting,created);''')
  behavior.init(self.c)
  self.start=datetime(2026,1,1,tzinfo=behavior.TZ).timestamp()
  self.c.execute("UPDATE behavior_meta SET value=?",(self.iso(0),))
  for uid,arm,test in [('a','M',0),('b','G',0),('qa','M',1)]:
   self.c.execute('INSERT INTO users VALUES(?,?,?,?,?)',(uid,uid,arm,test,self.iso(0)))
   self.c.execute('INSERT INTO accounts VALUES(?,?)',(uid,'P-'+uid))
 def tearDown(self):self.c.close()
 def iso(self,seconds):return datetime.fromtimestamp(self.start+seconds,timezone.utc).isoformat()
 def message(self,id,seconds,role='user',text='你好',asset=None,user='a'):
  self.c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?,?)',(id,user,'s',role,text,asset,self.iso(seconds)))
 def test_mix_latency_first_reply_and_missing_historical_time(self):
  self.message('u1',0,text='嗨🙂');self.message('bot',10,'assistant')
  self.message('u2',30,text='',asset='img');self.message('u3',40,text='好的',asset='img')
  self.c.execute('INSERT INTO message_views VALUES(?,?,?,?)',('a','page','bot',self.iso(22)))
  behavior.record_send(self.c,'a','u2',self.iso(30),{'page':'page','reply_to':'bot','response_ms':8000})
  behavior.record_send(self.c,'a','u3',self.iso(40),{'page':'page','reply_to':'bot','response_ms':18000})
  d=behavior.report(self.c,user='a',clock=self.start+86400+100);s=d['summary']
  self.assertEqual((s['user_sends'],s['text_only'],s['image_only'],s['mixed']),(3,1,1,1))
  self.assertAlmostEqual(s['image_send_share'],2/3,5);self.assertEqual(s['image_text_ratio'],1)
  self.assertEqual((s['emoji_sends'],s['response_samples'],s['response_median_seconds']),(1,1,8))
  self.assertIsNone(s['active_seconds']);self.assertEqual(s['replies_to_bot'],1)
  self.assertEqual(len(d['user_days']),2)
 def test_union_activity_dedup_zero_and_midnight(self):
  self.message('u',0)
  data={'page':'p1','seq':1,'visible_seconds':15,'active_seconds':10}
  behavior.heartbeat(self.c,'a',data,self.start+20);behavior.heartbeat(self.c,'a',data,self.start+20)
  behavior.heartbeat(self.c,'a',{**data,'page':'p2','active_seconds':15},self.start+25)
  behavior.heartbeat(self.c,'a',{**data,'seq':2,'active_seconds':0},self.start+40)
  behavior.heartbeat(self.c,'a',{**data,'seq':3,'visible_seconds':10,'active_seconds':10},self.start+86405)
  d=behavior.report(self.c,user='a',clock=self.start+86410)
  self.assertEqual(d['summary']['active_seconds'],25)
  self.assertEqual([r['active_seconds'] for r in d['user_days']],[20,5])
  self.assertEqual(self.c.execute('SELECT count(*) FROM activity_intervals').fetchone()[0],4)
 def test_filters_surveys_and_28_day_window(self):
  self.message('a1',0);self.message('a28',27*86400);self.message('outside',28*86400)
  self.message('b1',0,user='b');self.message('q1',0,user='qa')
  d=behavior.report(self.c,clock=self.start+30*86400)
  self.assertEqual(d['summary']['participants'],2);self.assertEqual(d['summary']['user_sends'],3)
  self.assertEqual(behavior.report(self.c,arm='M',week=4,clock=self.start+30*86400)['summary']['user_sends'],1)
  self.assertEqual(behavior.schedule(self.c,'a',self.start+6*86400)['waves'][0]['open'],True)
  self.assertEqual(behavior.schedule(self.c,'a',self.start+6*86400)['waves'][1]['open'],False)
  self.assertEqual(behavior.schedule(self.c,'a',self.start+27*86400)['current_day'],28)
 def test_invalid_telemetry_and_wrong_user_message(self):
  self.message('bot',0,'assistant',user='b')
  with self.assertRaises(ValueError):behavior.observe(self.c,'a',{'page':'p','message':'bot'})
  for n in [float('nan'),-1,30]:
   with self.assertRaises(ValueError):behavior.heartbeat(self.c,'a',{'page':'p','seq':1,'visible_seconds':n,'active_seconds':0})
  behavior.record_send(self.c,'a','u',self.iso(10),{'page':'p','reply_to':'bot','response_ms':1000})
  self.assertIsNone(self.c.execute('SELECT reply_to FROM message_sends').fetchone()[0])
 def test_no_samples_is_null_and_emoji_count_is_grapheme(self):
  self.message('a1',0,text='👨‍👩‍👧‍👦🙂好')
  s=behavior.report(self.c,user='a',clock=self.start+50)['summary']
  self.assertEqual(s['emoji_count'],2);self.assertEqual(s['text_characters'],1)
  self.assertIsNone(s['response_median_seconds']);self.assertEqual(s['response_samples'],0)
 def test_receipt_time_prevents_generation_delay_from_distorting_day(self):
  self.message('u',86405);behavior.record_send(self.c,'a','u',self.iso(86395),{})
  self.assertEqual(behavior.schedule(self.c,'a',self.start+86410)['current_day'],2)
  d=behavior.report(self.c,user='a',clock=self.start+86410)
  self.assertEqual(d['user_days'][0]['user_sends'],1)

 def test_completed_chat_rounds_exclude_greetings_bursts_and_unanswered_messages(self):
  self.message('opening',0,'assistant')
  self.c.execute('INSERT INTO visits VALUES(?,?,?,?,?)',('v0','a','s','你好',self.iso(0)))
  self.message('u1',1);self.message('u2',2,text='',asset='image')
  self.message('bot1',3,'assistant',text='第一句\n\n第二句',asset='image')
  self.message('bot-extra',4,'assistant')
  self.message('unanswered',5)
  self.message('return-greeting',700,'assistant')
  self.c.execute('INSERT INTO visits VALUES(?,?,?,?,?)',('v1','a','s','你好',self.iso(700)))
  self.message('bot-only',701,'assistant')
  self.message('u3',702);self.message('bot2',703,'assistant')
  self.message('still-waiting',704)
  self.message('b-user',2,user='b');self.message('b-bot',3,'assistant',user='b')
  d=behavior.report(self.c,user='a',clock=self.start+800)
  self.assertEqual(d['summary']['chat_rounds'],2)
  self.assertEqual(d['user_days'][0]['chat_rounds'],2)
  self.assertEqual(d['participants'][0]['summary']['chat_rounds'],2)
  self.assertEqual(d['weekly'][0]['chat_rounds'],2)
  self.c.execute("DELETE FROM messages WHERE id='bot2'")
  self.assertEqual(behavior.report(self.c,user='a',clock=self.start+800)['summary']['chat_rounds'],1)
 def test_rounds_use_completion_day_across_week_and_study_boundaries(self):
  self.message('start',0)
  self.message('reply1',7*86400+1,'assistant')
  self.message('u28',28*86400-10)
  self.message('reply-outside',28*86400+1,'assistant')
  d=behavior.report(self.c,user='a',clock=self.start+29*86400)
  self.assertEqual(d['summary']['chat_rounds'],1)
  self.assertEqual(d['user_days'][7]['chat_rounds'],1)
  self.assertEqual(behavior.report(self.c,user='a',week=1,clock=self.start+29*86400)['summary']['chat_rounds'],0)
  self.assertEqual(behavior.report(self.c,user='a',week=2,clock=self.start+29*86400)['summary']['chat_rounds'],1)
