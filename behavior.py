"""28-day research metrics. SQLite messages are authoritative; timing is prospective."""
import json, math, re, statistics, time
from datetime import datetime, timezone, timedelta
import regex

TZ = timezone(timedelta(hours=8))
DAYS = 28
WAVES = (7, 14, 28)
VERSION = 'behavior-28d-v2'
DEFINITIONS = {
 'study_day': '首次用户消息所在的北京时间日期为第1天；分析窗口为第1–28天，未来天不计零值。',
 'user_sends': '数据库中成功保存的用户发送次数。图文一起发送算1次，不按界面气泡数重复计数。',
 'content_mix': '纯文字（含Emoji）、纯图片、图文混合三类互斥，总和等于用户发送次数；图片含普通照片，不能自动等同于表情包。',
 'image_send_share': '含图片的发送次数 ÷ 全部用户发送次数。Emoji发送占比单列，可能与图片占比重叠。',
 'image_text_ratio': '图片数 ÷ 含文字（含Emoji）的发送次数；图文混合同时计入分子与分母。分母为0时不计算。',
 'active_seconds': '页面可见且最近120秒发生用户触摸/点击/输入等活动的累计时长；15秒心跳采集，跨标签/设备区间合并去重。浏览器关闭、后台、长时间静置不计入。属于有效前台使用时长估计，不等于注意或情感投入。',
 'response_seconds': '机器人完整文字可见（历史消息进入可视区也算）至该页面首次回复发送的单调时钟时差；一个机器人回复仅匹配首次用户回复。未采到可见事件的数据记缺失，不用模型生成耗时或历史消息间隔替代。',
 'chat_rounds': '完成的用户→机器人往返回合：一次或连续多次用户发送，随后有一条成功保存的机器人回复计1回合；分段/图文气泡不重复计数，主动问候、仅机器人连续消息、未完成或已撤回的回复不计。跨日回合归于机器人回复完成日。',
 'replies_to_bot': '按消息顺序，机器人回复后的首次用户发送；连续追加消息仍计入发送总量，但不重复计为回复机器人。',
 'surveys': '第7、14、28天开放对应阶段问卷。旧baseline记录保留为历史数据，不充当本方案基线。现有题目仍为原型占位题。',
 'group_summary': '组间汇总仅为描述统计，不是显著性检验；发送/人的分母为筛选内全部已分配被试（含尚未开始者），图片占比按发送次数汇总；多日、多条消息属于同一被试的重复观测。',
 'timing_coverage': '有效时长与可见后响应时延仅从本版本启用后可测。历史消息数和内容比例可补算，历史精确时长不可恢复。'
}

def init(c):
 c.executescript('''
 CREATE TABLE IF NOT EXISTS behavior_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
 CREATE TABLE IF NOT EXISTS message_sends(message TEXT PRIMARY KEY,user TEXT NOT NULL,received_at TEXT NOT NULL,page TEXT,reply_to TEXT,response_ms REAL);
 CREATE TABLE IF NOT EXISTS message_views(user TEXT NOT NULL,page TEXT NOT NULL,message TEXT NOT NULL,seen_at TEXT NOT NULL,PRIMARY KEY(user,page,message));
 CREATE TABLE IF NOT EXISTS activity_intervals(user TEXT NOT NULL,page TEXT NOT NULL,seq INTEGER NOT NULL,ended REAL NOT NULL,visible_seconds REAL NOT NULL,active_seconds REAL NOT NULL,PRIMARY KEY(user,page,seq));
 CREATE INDEX IF NOT EXISTS activity_user_time ON activity_intervals(user,ended);
 CREATE INDEX IF NOT EXISTS sends_user ON message_sends(user);
 ''')
 c.execute('INSERT OR IGNORE INTO behavior_meta VALUES(?,?)', ('collection_started_at',datetime.now(timezone.utc).isoformat()))

def stamp(value):
 try:
  d=datetime.fromisoformat(value)
  return d.replace(tzinfo=timezone.utc).timestamp() if d.tzinfo is None else d.timestamp()
 except (ValueError,TypeError): return None

def page_id(value):
 return value if isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9_-]{1,80}',value) else None

def record_send(c,user,message,received_at,data):
 data=data if isinstance(data,dict) else {}
 page=page_id(data.get('page'));reply=data.get('reply_to');ms=data.get('response_ms')
 if not (page and isinstance(reply,str) and type(ms) in (int,float) and math.isfinite(ms) and 0<=ms<=7*86400000):
  page=reply=ms=None
 elif not c.execute("SELECT 1 FROM messages WHERE id=? AND user=? AND role='assistant'",(reply,user)).fetchone():
  page=reply=ms=None
 c.execute('INSERT OR IGNORE INTO message_sends VALUES(?,?,?,?,?,?)',(message,user,received_at,page,reply,ms))

def observe(c,user,data):
 page=page_id(data.get('page'));message=data.get('message')
 if not page or not isinstance(message,str) or not c.execute("SELECT 1 FROM messages WHERE id=? AND user=? AND role='assistant'",(message,user)).fetchone():
  raise ValueError('消息可见记录无效')
 c.execute('INSERT OR IGNORE INTO message_views VALUES(?,?,?,?)',(user,page,message,datetime.now(timezone.utc).isoformat()))

def heartbeat(c,user,data,clock=None):
 page=page_id(data.get('page'));seq=data.get('seq');visible=data.get('visible_seconds');active=data.get('active_seconds')
 if not page or type(seq) is not int or not 0<=seq<10**9:
  raise ValueError('活动记录无效')
 if any(type(v) not in (int,float) or not math.isfinite(v) for v in (visible,active)) or not 0<=active<=visible<=20:
  raise ValueError('活动时长无效')
 c.execute('INSERT OR IGNORE INTO activity_intervals VALUES(?,?,?,?,?,?)',(user,page,seq,time.time() if clock is None else clock,visible,active))

def schedule(c,user,clock=None):
 rows=c.execute("SELECT m.created,s.received_at FROM messages m LEFT JOIN message_sends s ON s.message=m.id AND s.user=m.user WHERE m.user=? AND m.role='user' ORDER BY m.rowid LIMIT 1",(user,)).fetchone()
 start=stamp((rows['received_at'] or rows['created'])) if rows else None
 today=datetime.fromtimestamp(clock if clock is not None else time.time(),TZ).date()
 day=(today-datetime.fromtimestamp(start,TZ).date()).days+1 if start is not None else None
 completed={r['wave']:r['created'] for r in c.execute('SELECT wave,created FROM surveys WHERE user=?',(user,))}
 start_date=datetime.fromtimestamp(start,TZ).date() if start is not None else None
 return {'duration_days':DAYS,'timezone':'Asia/Shanghai','start_date':str(start_date) if start_date else None,'current_day':day,
         'waves':[{'wave':'day'+str(n),'day':n,'due_date':str(start_date+timedelta(days=n-1)) if start_date else None,'submitted_at':completed.get('day'+str(n)), 'open':day is not None and day>=n} for n in WAVES]}

def union_seconds(intervals):
 total=0;end=None
 for a,b in sorted(intervals):
  if b<=a:continue
  if end is None or a>end:total+=b-a;end=b
  elif b>end:total+=b-end;end=b
 return total

def parts(text):
 chars=regex.findall(r'\X',text or '')
 emoji=[x for x in chars if regex.search(r'\p{Extended_Pictographic}|\p{Regional_Indicator}|\u20e3',x)]
 return len(emoji),len(''.join(x for x in chars if x not in emoji).strip())

def ratio(a,b):return round(a/b,6) if b else None

def summarize(rows):
 sums={k:sum(r[k] for r in rows) for k in ('user_sends','text_sends','image_sends','text_only','image_only','mixed','emoji_sends','emoji_count','text_characters','replies_to_bot','chat_rounds','assistant_replies','assistant_images')}
 samples=[x for r in rows for x in r['_response_samples']]
 active=[r['active_seconds'] for r in rows if r['active_seconds'] is not None]
 visible=[r['visible_seconds'] for r in rows if r['visible_seconds'] is not None]
 return {**sums,'active_seconds':round(sum(active),2) if active else None,'visible_seconds':round(sum(visible),2) if visible else None,
         'image_send_share':ratio(sums['image_sends'],sums['user_sends']),'emoji_send_share':ratio(sums['emoji_sends'],sums['user_sends']),
         'image_text_ratio':ratio(sums['image_sends'],sums['text_sends']), 'image_bubble_share':ratio(sums['image_sends'],sums['image_sends']+sums['text_sends']),
         'response_median_seconds':round(statistics.median(samples),3) if samples else None,'response_samples':len(samples),
         'active_user_days':sum(r['user_sends']>0 for r in rows),'timing_user_days':len(active)}

def report(c,arm='',include_test=False,user='',week=0,clock=None):
 clock=time.time() if clock is None else clock
 if arm not in ('','T','G','M','memory','control') or week not in range(5):raise ValueError('筛选条件无效')
 users=[dict(r) for r in c.execute('SELECT u.id,u.label,u.arm,u.is_test,a.username FROM users u LEFT JOIN accounts a ON a.user=u.id ORDER BY u.created') if (include_test or not r['is_test']) and (not arm or r['arm']==arm) and (not user or r['id']==user)]
 all_rows=[];people=[]
 meta=c.execute("SELECT value FROM behavior_meta WHERE key='collection_started_at'").fetchone()
 collection=stamp(meta[0]) if meta else clock
 for u in users:
  uid=u['id'];plan=schedule(c,uid,clock);u['schedule']=plan
  if not plan['start_date']:
   people.append({**u,'summary':summarize([])});continue
  begin=datetime.fromisoformat(plan['start_date']).replace(tzinfo=TZ).timestamp()
  rows=[]
  for d in range(1,29):
   if week and (d-1)//7+1!=week:continue
   start=begin+(d-1)*86400;end=start+86400
   if start>clock:continue
   rows.append({'user':uid,'label':u['label'],'username':u['username'],'arm':u['arm'],'is_test':u['is_test'],'day':d,'week':(d-1)//7+1,'date':datetime.fromtimestamp(start,TZ).date().isoformat(),
    **{k:0 for k in ('user_sends','text_sends','image_sends','text_only','image_only','mixed','emoji_sends','emoji_count','text_characters','replies_to_bot','chat_rounds','assistant_replies','assistant_images')},
    'active_seconds':None,'visible_seconds':None,'_response_samples':[],'_active':[],'_visible':[]})
  byday={r['day']:r for r in rows}
  greeting_rows={(g['session'],g['created'],g['greeting']) for g in c.execute('SELECT session,created,greeting FROM visits WHERE user=?',(uid,))}
  messages=[dict(m) for m in c.execute('''SELECT m.*,s.received_at,s.page,s.reply_to,s.response_ms,v.seen_at FROM messages m
   LEFT JOIN message_sends s ON s.message=m.id AND s.user=m.user
   LEFT JOIN message_views v ON v.user=s.user AND v.page=s.page AND v.message=s.reply_to
   WHERE m.user=? ORDER BY m.rowid''',(uid,))]
  # Use receipt time for user messages from the new instrumentation. Historic
  # message times predate this change and may include generation delay.
  messages.sort(key=lambda m:stamp(m['received_at'] or m['created']) or 0)
  existing_assistant={m['id'] for m in messages if m['role']=='assistant'}
  answered=set();pending=None;user_waiting=False
  for m in messages:
   ts=stamp(m['received_at'] or m['created'])
   if ts is None:continue
   day=int((ts-begin)//86400)+1;r=byday.get(day)
   if m['role']=='assistant':
    if r and user_waiting and (m['session'],m['created'],m['text']) not in greeting_rows:r['chat_rounds']+=1
    # A greeting never completes an unanswered user turn from an earlier visit.
    user_waiting=False
    pending=m['id']
    if r:r['assistant_replies']+=1;r['assistant_images']+=bool(m['asset'])
    continue
   if m['role']!='user':continue
   user_waiting=True
   first_reply_to=pending;is_reply=pending is not None;pending=None
   if not r:continue
   text=bool((m['text'] or '').strip());image=bool(m['asset']);emoji,chars=parts(m['text'])
   r['user_sends']+=1;r['text_sends']+=text;r['image_sends']+=image;r['replies_to_bot']+=is_reply
   r['text_only']+=text and not image;r['image_only']+=image and not text;r['mixed']+=text and image
   r['emoji_sends']+=emoji>0;r['emoji_count']+=emoji;r['text_characters']+=chars
   if m['reply_to']==first_reply_to and m['reply_to'] in existing_assistant and m['seen_at'] and m['response_ms'] is not None and m['reply_to'] not in answered:
    seen=stamp(m['seen_at'])
    # Receipt race may put seen a moment after send; permit a 2s transport margin.
    if seen is not None and seen<=ts+2:
     r['_response_samples'].append(m['response_ms']/1000);answered.add(m['reply_to'])
  for a in c.execute('SELECT * FROM activity_intervals WHERE user=? AND ended>=? AND ended<=?',(uid,begin,begin+28*86400+20)):
   for r in rows:
    start=begin+(r['day']-1)*86400;end=min(start+86400,clock)
    if start<=a['ended']<=end or a['ended']-a['visible_seconds']<end<a['ended']:
     # Split intervals at midnight. One heartbeat proves collection even with 0 active seconds.
     if a['ended']>=start and a['ended']-a['visible_seconds']<=end:
      r['active_seconds']=0;r['visible_seconds']=0
      r['_active'].append((max(start,a['ended']-a['active_seconds']),min(end,a['ended'])))
      r['_visible'].append((max(start,a['ended']-a['visible_seconds']),min(end,a['ended'])))
  for r in rows:
   if r['active_seconds'] is not None:r['active_seconds']=round(union_seconds(r['_active']),2);r['visible_seconds']=round(union_seconds(r['_visible']),2)
   r['timing_status']='partial' if r['active_seconds'] is not None and begin+(r['day']-1)*86400<collection else 'collected' if r['active_seconds'] is not None else 'not_collected'
   r.update({k:v for k,v in summarize([r]).items() if k not in r})
  people.append({**u,'summary':summarize(rows)})
  all_rows.extend(rows)
 groups=[]
 for group in sorted({u['arm'] for u in users}):
  subset=[r for r in all_rows if r['arm']==group];n=sum(u['arm']==group for u in users)
  summary=summarize(subset)
  groups.append({'arm':group,'participants':n,**summary,'sends_per_participant':ratio(summary['user_sends'],n)})
 daily=[{'day':d,**summarize([r for r in all_rows if r['day']==d])} for d in sorted({r['day'] for r in all_rows})]
 weekly=[{'week':w,**summarize([r for r in all_rows if r['week']==w])} for w in sorted({r['week'] for r in all_rows})]
 clean=[{k:v for k,v in r.items() if not k.startswith('_')} for r in all_rows]
 return {'version':VERSION,'generated_at':datetime.fromtimestamp(clock,timezone.utc).isoformat(),'collection_started_at':meta[0] if meta else None,'duration_days':28,'survey_days':list(WAVES),'timezone':'Asia/Shanghai','filters':{'arm':arm,'user':user,'week':week,'include_test':include_test},'definitions':DEFINITIONS,
         'participants':people,'summary':{'participants':len(users),'active_participants':sum(p['summary']['user_sends']>0 for p in people),**summarize(all_rows)},'groups':groups,'daily':daily,'weekly':weekly,'user_days':clean}

def transcript(c,user,offset=0,limit=100):
 if not c.execute('SELECT 1 FROM users WHERE id=?',(user,)).fetchone():raise ValueError('用户不存在')
 total=c.execute('SELECT count(*) FROM messages WHERE user=?',(user,)).fetchone()[0]
 rows=[dict(r) for r in c.execute('''SELECT m.id,m.role,m.text,m.asset,m.created,m.session,s.received_at,a.description
 FROM messages m LEFT JOIN message_sends s ON s.message=m.id AND s.user=m.user
 LEFT JOIN assets a ON a.id=m.asset AND a.user=m.user WHERE m.user=? ORDER BY m.rowid LIMIT ? OFFSET ?''',(user,limit,offset))]
 return {'user':user,'total':total,'offset':offset,'messages':rows,'next_offset':offset+len(rows) if offset+len(rows)<total else None}
