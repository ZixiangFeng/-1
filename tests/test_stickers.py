import sqlite3,json,unittest
import stickers
class Stickers(unittest.TestCase):
 def setUp(self):
  self.c=sqlite3.connect(':memory:');self.c.row_factory=sqlite3.Row
  self.c.executescript('CREATE TABLE assets(id,user,description,image,metadata);CREATE TABLE memories(id,user,expression,meaning,context,status,source,created);CREATE TABLE messages(id,user,session,role,text,asset);CREATE TABLE events(id INTEGER PRIMARY KEY,user,kind,payload);')
  stickers.init(self.c)
  import generic_stickers
  generic_stickers.init(self.c)
  for u,a in [('a','cat'),('b','private')]:
   self.c.execute('INSERT INTO assets(id,user,description) VALUES(?,?,?)',(a,u,'猫猫庆祝表情包'))
   self.c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?)',('src'+u,u,'old','user','终于搞定了',a))
   self.c.execute('INSERT INTO memories VALUES(?,?,?,?,?,?,?,?)',('mem'+u,u,'image:'+a,'庆祝','完成任务','inferred','src'+u,'1'))
 def tearDown(self):self.c.close()
 def test_isolation_and_control(self):
  self.assertEqual([x['asset_id'] for x in stickers.candidates(self.c,'a','new','memory')],['cat'])
  self.assertEqual(stickers.candidates(self.c,'a','new','control'),[])
  self.assertEqual(len(stickers.candidates(self.c,'a','old','control')),1)
 def test_no_echo_repeat_or_superseded(self):
  self.assertEqual(stickers.candidates(self.c,'a','new','memory','cat'),[])
  self.c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?)',('reply','a','new','assistant','好耶','cat'))
  self.assertEqual(stickers.candidates(self.c,'a','new','memory'),[])
 def test_allowlist_and_abstain(self):
  for aid in ['cat','']:
   out,_=stickers.choose(self.c,'a','new','memory','完成啦','太好啦',None,lambda *a,**kw:(json.dumps({'asset_id':aid,'reason':'庆祝'}),{}),rng=lambda n:0)
   self.assertEqual(out,aid or None)
  with self.assertRaises(ValueError):
   stickers.choose(self.c,'a','new','memory','完成啦','太好啦',None,lambda *a,**kw:(json.dumps({'asset_id':'private','reason':'test'}),{}),rng=lambda n:0)

 def test_personal_frequency_gate_skips_model_when_not_selected(self):
  calls=[]
  out,audit=stickers.choose(self.c,'a','new','memory','完成啦','太好啦',None,lambda *a,**kw:calls.append(1),rng=lambda n:9999)
  self.assertIsNone(out)
  self.assertFalse(audit['probability_passed'])
  self.assertEqual(audit['send_probability'],3/7)
  self.assertEqual(calls,[])

 def test_personal_frequency_gate_allows_semantic_selection_on_text_turn(self):
  out,audit=stickers.choose(self.c,'a','new','memory','完成啦','太好啦',None,lambda *a,**kw:(json.dumps({'asset_id':'cat','reason':'庆祝'}),{}),rng=lambda n:0)
  self.assertEqual(out,'cat')
  self.assertTrue(audit['probability_passed'])

 def test_probability_adapts_to_users_observed_sticker_rate(self):
  initial=stickers.usage_probability(self.c,'a')
  for i in range(9):self.c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?)',(f't{i}','a','old','user','纯文字',None))
  low=stickers.usage_probability(self.c,'a')
  for i in range(18):self.c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?)',(f's{i}','a','old','user','', 'cat'))
  high=stickers.usage_probability(self.c,'a')
  self.assertLess(low['probability'],initial['probability'])
  self.assertGreater(high['probability'],initial['probability'])
  self.assertEqual(high['sample_size'],28)

 def test_three_conditions_sources_and_allowlist(self):
  import generic_stickers
  generic_stickers.init(self.c)
  self.c.execute('INSERT INTO generic_stickers VALUES(?,?,?,?,?)',('g_test','image',json.dumps(dict(description='庆祝贴纸',meaning='开心',situations='完成任务',avoid='难过')),1,'now'))
  self.assertEqual(stickers.candidates(self.c,'a','new','T'),[])
  self.assertEqual([x['asset_id'] for x in stickers.candidates(self.c,'a','new','G')],['g_test'])
  self.assertEqual([x['asset_id'] for x in stickers.candidates(self.c,'a','new','M')],['cat'])
  with self.assertRaises(ValueError):
   stickers.choose(self.c,'a','new','G','完成','好耶',None,lambda *a,**kw:(json.dumps({'asset_id':'cat','reason':'wrong source'}),{}),rng=lambda n:0)
  self.c.execute('UPDATE generic_stickers SET enabled=0')
  self.assertEqual(stickers.candidates(self.c,'a','new','G'),[])

 def test_t_never_calls_selector_even_with_injected_candidates(self):
  calls=[]
  result,audit=stickers.choose(self.c,'a','new','T','hi','hi',None,lambda *a,**kw:calls.append(1),rng=lambda n:0,options=[{'asset_id':'cat'}])
  self.assertIsNone(result);self.assertEqual(calls,[])

 def test_streak_counts_completed_replies_and_resets_across_sessions(self):
  def reply(i,user='a',asset=None,kind='sticker_selection',exists=True):
   if exists:self.c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?)',(i,user,'day2','assistant','回复',asset))
   self.c.execute('INSERT INTO events(user,kind,payload) VALUES(?,?,?)',(user,kind,json.dumps({'message':i})))
  reply('a1');reply('a2');reply('greeting',kind='proactive_greeting');reply('withdrawn',exists=False);reply('other','b','private')
  self.assertEqual(stickers.no_sticker_streak(self.c,'a'),2)
  reply('sent',asset='cat');self.assertEqual(stickers.no_sticker_streak(self.c,'a'),0)
  reply('next');self.assertEqual(stickers.no_sticker_streak(self.c,'a'),1)

 def test_boost_has_grace_cap_and_remains_random(self):
  s=stickers.DEFAULT_FREQUENCY
  self.assertAlmostEqual(stickers.weighted_probability(1/3,1,s),1/3)
  self.assertAlmostEqual(stickers.weighted_probability(1/3,2,s),1/3+.1)
  self.assertEqual(stickers.weighted_probability(1/3,200,s),.85)
  self.assertEqual(stickers.weighted_probability(.9,200,s),.9)
  self.assertEqual(stickers.weighted_probability(.2,200,{**s,'enabled':False}),.2)

 def test_boost_shared_and_semantic_abstention_preserved(self):
  for i in range(5):
   self.c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?)',(str(i),'a','old','assistant','回复',None))
   self.c.execute('INSERT INTO events(user,kind,payload) VALUES(?,?,?)',('a','sticker_selection',json.dumps({'message':str(i)})))
  self.c.execute('INSERT INTO generic_stickers VALUES(?,?,?,?,?)',('g_streak','unrelated',json.dumps(dict(description='庆祝',meaning='开心',situations='庆祝',avoid='痛苦')),1,'now'))
  probabilities=[]
  for arm in ['G','M']:
   out,a=stickers.choose(self.c,'a','new',arm,'难过','我在',None,lambda *a,**kw:(json.dumps({'asset_id':'','reason':'不合适'}),{}),rng=lambda n:6000,options=[{'asset_id':'g_streak' if arm=='G' else 'cat'}])
   self.assertIsNone(out);self.assertTrue(a['probability_passed']);probabilities.append(a['send_probability'])
  self.assertEqual(*probabilities)
  self.assertEqual(stickers.no_sticker_streak(self.c,'a'),5)

 def test_frequency_settings_validate_and_persist(self):
  s={**stickers.DEFAULT_FREQUENCY,'increment':.05}
  stickers.save_frequency_settings(self.c,s);self.assertEqual(stickers.frequency_settings(self.c),s)
  for bad in [{**s,'cap':1},{**s,'increment':float('nan')},{**s,'start_after':True},{**s,'enabled':'true'}]:
   with self.assertRaises(ValueError):stickers.save_frequency_settings(self.c,bad)

 def test_cooldown_blocks_all_image_arms_and_resumes_after_committed_text(self):
  from unittest.mock import patch
  self.c.execute('INSERT INTO generic_stickers VALUES(?,?,?,?,?)',('g_gap','other',json.dumps(dict(description='庆祝',meaning='开心',situations='庆祝',avoid='痛苦')),1,'now'))
  def record(mid,asset=None,user='a',event=True):
   self.c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?)',(mid,user,'yesterday','assistant','回复',asset))
   if event:self.c.execute('INSERT INTO events(user,kind,payload) VALUES(?,?,?)',(user,'sticker_selection',json.dumps({'message':mid})))
  record('sent','old-image')
  record('greeting',event=False)  # greetings never consume the gap
  record('foreign',user='b')
  record('withdrawn');self.c.execute("DELETE FROM messages WHERE id='withdrawn'")
  with patch('stickers.expression_usage.context',return_value={'references':[]}):
   for arm in ['G','M','D']:
    calls=[]
    out,a=stickers.choose(self.c,'a','today',arm,'开心','好耶',None,lambda *a,**kw:calls.append('model'),rng=lambda n:0,options=[{'asset_id':'cat' if arm=='M' else 'g_gap'}])
    self.assertIsNone(out);self.assertEqual(calls,[])
    self.assertEqual(a['cooldown_remaining'],1);self.assertEqual(a['send_probability'],0)
    self.assertNotIn('probability_draw',a)
   record('text-gap')
   out,a=stickers.choose(self.c,'a','today','M','开心','好耶',None,lambda *a,**kw:(json.dumps({'asset_id':'cat','reason':'庆祝'}),{}),rng=lambda n:0,options=[{'asset_id':'cat'}])
   self.assertEqual(out,'cat');self.assertEqual(a['cooldown_remaining'],0)
   self.assertEqual(a['no_sticker_streak'],1)

 def test_first_reply_has_no_cooldown_and_boost_toggle_cannot_remove_gap(self):
  self.assertEqual(stickers.cooldown_remaining(self.c,'a',0),0)
  self.c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?)',('sent','a','old','assistant','回复','cat'))
  self.c.execute('INSERT INTO events(user,kind,payload) VALUES(?,?,?)',('a','sticker_selection',json.dumps({'message':'sent'})))
  stickers.save_frequency_settings(self.c,{**stickers.DEFAULT_FREQUENCY,'enabled':False})
  out,a=stickers.choose(self.c,'a','new','M','hello','hi',None,lambda *a,**kw:self.fail('must skip model'),rng=lambda n:0,options=[{'asset_id':'cat'}])
  self.assertEqual(a['cooldown_remaining'],1)

 def test_old_sticker_retrievable_beyond_recent_learning_window(self):
  for i in range(70):
   aid='new'+str(i);mid='source'+str(i)
   self.c.execute('INSERT INTO assets(id,user,description) VALUES(?,?,?)',(aid,'a','早安问候表情'))
   self.c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?)',(mid,'a','today','user','早安',aid))
   self.c.execute('INSERT INTO memories VALUES(?,?,?,?,?,?,?,?)',('memory'+str(i),'a','image:'+aid,'早安','早晨','inferred',mid,str(i+10)))
  options=stickers.candidates(self.c,'a','future','M',query='终于搞定了，一起庆祝！')
  self.assertEqual(options[0]['asset_id'],'cat')
  self.assertEqual(options[0]['eligible_pool_size'],71)
  self.assertEqual(len(options),12)

 def test_sent_without_inference_allowed_but_import_only_excluded(self):
  self.c.execute('INSERT INTO assets(id,user,description) VALUES(?,?,?)',('sent','a','晚安睡觉贴纸'))
  self.c.execute('INSERT INTO assets(id,user,description) VALUES(?,?,?)',('import_only','a','晚安睡觉贴纸'))
  self.c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?)',('sent-old','a','last_week','user','晚安','sent'))
  options=stickers.candidates(self.c,'a','today','M',query='晚安')
  found=next(x for x in options if x['asset_id']=='sent')
  self.assertEqual(found['usages'],[]);self.assertEqual(found['observed_contexts'],['晚安'])
  self.assertNotIn('import_only',[x['asset_id'] for x in options])
  self.assertNotIn('private',[x['asset_id'] for x in options])

 def test_reuse_after_cooldown_and_all_history_vector_search(self):
  self.c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?)',('bot-cat','a','old','assistant','庆祝','cat'))
  for i in range(3):self.c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?)',('bot'+str(i),'a','old','assistant','文字',None))
  class Store:
   def upsert(inner,docs):inner.docs=docs
   def search(inner,query,user,kind,limit):
    self.assertEqual((user,kind),('a','personal_sticker'))
    self.assertTrue(all(x['user']=='a' for x in inner.docs))
    return [{'id':'personal_sticker:b:private'},{'id':inner.docs[0]['id']}]
  options=stickers.candidates(self.c,'a','next_day','M',query='庆祝',store=Store())
  self.assertEqual([x['asset_id'] for x in options],['cat'])
  self.assertEqual(options[0]['retrieval_scope'],'all_user_history')

 def test_no_store_still_searches_old_pool(self):
  class BrokenStore:
   def upsert(self,docs):raise RuntimeError('offline')
  self.assertEqual(stickers.candidates(self.c,'a','new','M',query='庆祝',store=BrokenStore())[0]['asset_id'],'cat')
