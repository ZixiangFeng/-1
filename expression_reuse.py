"""Whole-study, per-participant exact-image reuse; no short response window."""
import bisect
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone

import behavior

VERSION = 'expression-reuse-28d-v2'
DEFINITIONS = {
    'period': '固定统计每位被试第1–28天。起点为首次用户发送所在北京时间日期的00:00，第29天00:00截止；未结束时展示截至当前的累计值。与上方每日行为的周筛选独立，不设24小时或后续发送次数上限。',
    'identity': '按该被试的同一原始asset_id匹配，跨会话、跨日、跨周累积；库导入不算聊天使用。照片尚未自动排除，不同ID的近似图片不自动合并。',
    'sequence': '同图在用户已经使用过之后，被机器人发出才构成机器人复用；随后用户在28天内每次再用均计入累计回用次数。机器人重复复用不会清零或截断累计统计。',
    'attribution': '每条用户同图回用消息只计一次。明细关联最近一次此前的机器人复用，同时保留首次机器人复用ID；连续机器人复用不能让同一条用户消息被重复计数。',
    'rate': '同图回用覆盖率＝机器人复用后用户再次使用过的不同图片数÷机器人复用过的不同图片数。分母为0时不计算；进行中的四周研究显示截至当前值，不能将其当作完整28天结论。',
    'counts': '逐图保留用户总使用次数、机器人复用次数、首次机器人复用前用户使用次数、首次复用后用户累计回用次数，以及首次使用/首次复用/首次回用/最近回用时间。全部使用次数以已保存消息计，不按拆分气泡重复。',
    'visibility': '发送后回用和可见后回用分列。图片解码成功且前台进入视口才有image_views记录；历史可见缺失不补造，也不阻止整段发送时序统计。没有可见样本显示未确认，不把它当0%。可见代理不等于实际注意。',
    'timing': '每次回用保留距首次及最近一次机器人复用的秒数、最近一次复用后第几次用户发送，并保留可见前7天使用次数。用户优先使用服务器接收时间，旧记录使用保存时间；遥测时间受网络延迟影响。',
    'weekly': '四周拆分按实际使用发生所在的实验周计数，四周之和等于28天累计；未来周显示未开始，未满28天显示进行中，机器人从第1周开始复用的图在第4周回用仍计入。',
    'interpretation': '这些是探索性的双向使用过程描述。语境、原有习惯和选择偏差均可能导致回用，不能单独证明信任提高或因果适应；G组通用图与用户个人图不跨ID归并。',
}

def init(c):
    c.executescript('''
    CREATE TABLE IF NOT EXISTS image_views(
      user TEXT NOT NULL, page TEXT NOT NULL, message TEXT NOT NULL,
      asset TEXT NOT NULL, seen_at TEXT NOT NULL,
      PRIMARY KEY(user,page,message));
    CREATE INDEX IF NOT EXISTS image_views_user_message ON image_views(user,message);
    CREATE INDEX IF NOT EXISTS messages_user_asset ON messages(user,asset);
    ''')
    c.execute('INSERT OR IGNORE INTO behavior_meta VALUES(?,?)',
              ('image_visibility_started_at', datetime.now(timezone.utc).isoformat()))


def observe(c, user, data):
    page = behavior.page_id(data.get('page'))
    message, asset = data.get('message'), data.get('asset')
    if not page or not isinstance(message, str) or not isinstance(asset, str):
        raise ValueError('图片可见记录无效')
    if not c.execute("SELECT 1 FROM messages WHERE user=? AND id=? AND role='assistant' AND asset=?",
                     (user, message, asset)).fetchone():
        raise ValueError('图片可见记录无效')
    c.execute('INSERT OR IGNORE INTO image_views VALUES(?,?,?,?,?)',
              (user, page, message, asset, datetime.now(timezone.utc).isoformat()))


def _iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat() if ts is not None else None


def report(c, user, clock=None):
    person = c.execute('SELECT id,label,arm,is_test FROM users WHERE id=?', (user,)).fetchone()
    if not person:
        raise ValueError('请选择一位存在的被试')
    now = time.time() if clock is None else clock
    plan = behavior.schedule(c, user, now)
    begin = (datetime.fromisoformat(plan['start_date']).replace(tzinfo=behavior.TZ).timestamp()
             if plan['start_date'] else None)
    end = begin + 28*86400 if begin is not None else None
    ongoing = begin is not None and now < end
    rows = []
    for r in c.execute('''SELECT m.id,m.session,m.role,m.asset,m.created,s.received_at
                          FROM messages m LEFT JOIN message_sends s ON s.user=m.user AND s.message=m.id
                          WHERE m.user=? ORDER BY m.rowid''', (user,)):
        ts = behavior.stamp(r['received_at'] or r['created'])
        if ts is not None and begin is not None and begin <= ts < end and ts <= now:
            rows.append({**dict(r), '_ts':ts})
    rows.sort(key=lambda r:r['_ts'])
    for index,r in enumerate(rows):
        r['_index'] = index
    all_user = [r for r in rows if r['role']=='user']
    user_indices = [r['_index'] for r in all_user]
    by_asset = defaultdict(list)
    for r in rows:
        if r['asset'] and r['role'] in ('user','assistant'):
            by_asset[r['asset']].append(r)
    seen = {}
    for r in c.execute('SELECT message,asset,MIN(seen_at) AS seen_at FROM image_views WHERE user=? GROUP BY message,asset', (user,)):
        ts = behavior.stamp(r['seen_at'])
        if ts is not None and begin is not None and begin <= ts < end and ts <= now:
            seen[(r['message'],r['asset'])] = ts
    assets, reuse_events, timeline = [], [], []
    weekly = []
    for w in range(1,5):
        started = begin is not None and begin+(w-1)*7*86400 <= now
        weekly.append({'week':w, 'observed':started, 'user_image_sends':0,
                       'assistant_reuse_sends':0, 'user_reuses_after_robot':0,
                       'user_reuses_after_seen':0})
    def week_of(ts):
        return int((ts-begin)//(7*86400))
    for asset,items in sorted(by_asset.items()):
        users = [r for r in items if r['role']=='user']
        bots = [r for r in items if r['role']=='assistant']
        first_user = users[0] if users else None
        eligible = [b for b in bots if first_user and first_user['_index'] < b['_index']]
        first_bot = eligible[0] if eligible else None
        bot_indices = [b['_index'] for b in eligible]
        confirmed = [(seen[(b['id'],asset)],b) for b in eligible
                     if (b['id'],asset) in seen and seen[(b['id'],asset)]>=b['_ts']]
        confirmed.sort(key=lambda v:(v[0],v[1]['_index']))
        first_seen = confirmed[0][0] if confirmed else None
        hits = []
        for r in items:
            timeline.append({'user':user, 'asset_id':asset, 'message_id':r['id'],
                             'role':r['role'], 'session':r['session'], 'sent_at':_iso(r['_ts']),
                             'image_seen_at':_iso(seen.get((r['id'],asset))) if r['role']=='assistant' else None,
                             'week':week_of(r['_ts'])+1})
        for r in eligible:
            weekly[week_of(r['_ts'])]['assistant_reuse_sends'] += 1
        for r in users:
            weekly[week_of(r['_ts'])]['user_image_sends'] += 1
            if not first_bot or r['_index'] <= first_bot['_index']:
                continue
            latest = eligible[bisect.bisect_left(bot_indices,r['_index'])-1]
            viewed = [v for v in confirmed if v[0]<=r['_ts'] and v[1]['_index']<r['_index']]
            latest_view = viewed[-1] if viewed else None
            n = bisect.bisect_right(user_indices,r['_index'])-bisect.bisect_right(user_indices,latest['_index'])
            # A row is one actual user message, never one row per preceding bot.
            e = {'user':user, 'asset_id':asset, 'user_message_id':r['id'], 'user_session':r['session'],
                 'user_reused_at':_iso(r['_ts']), 'week':week_of(r['_ts'])+1,
                 'first_assistant_message_id':first_bot['id'], 'first_assistant_reused_at':_iso(first_bot['_ts']),
                 'latest_assistant_message_id':latest['id'], 'latest_assistant_reused_at':_iso(latest['_ts']),
                 'seconds_from_first_robot':round(r['_ts']-first_bot['_ts'],3),
                 'seconds_from_latest_robot':round(r['_ts']-latest['_ts'],3), 'sends_from_latest_robot':n,
                 'has_prior_image_view':bool(latest_view),
                 'latest_seen_assistant_message_id':latest_view[1]['id'] if latest_view else None,
                 'latest_image_seen_at':_iso(latest_view[0]) if latest_view else None,
                 'seconds_from_latest_view':round(r['_ts']-latest_view[0],3) if latest_view else None,
                 'prior_7d_asset_sends':sum(latest['_ts']-7*86400<=u['_ts']<latest['_ts'] for u in users),
                 'prior_7d_user_sends':sum(latest['_ts']-7*86400<=u['_ts']<latest['_ts'] for u in all_user)}
            reuse_events.append(e)
            hits.append(e)
            weekly[e['week']-1]['user_reuses_after_robot'] += 1
            weekly[e['week']-1]['user_reuses_after_seen'] += bool(latest_view)
        personal = c.execute('SELECT description FROM assets WHERE user=? AND id=?',(user,asset)).fetchone()
        gaps = [h['seconds_from_latest_robot'] for h in hits]
        assets.append({'user':user, 'asset_id':asset,
                       'description':personal['description'] if personal else '',
                       'source':'personal' if personal else 'generic_or_missing',
                       'user_sends':len(users), 'assistant_sends':len(bots),
                       'assistant_reuse_sends':len(eligible),
                       'user_sends_before_first_robot':sum(not first_bot or u['_index']<first_bot['_index'] for u in users),
                       'user_reuses_after_robot':len(hits), 'reused_after_robot':bool(hits),
                       'confirmed_robot_sends':len(confirmed),
                       'user_reuses_after_seen':sum(h['has_prior_image_view'] for h in hits) if confirmed else None,
                       'first_user_used_at':_iso(first_user['_ts']) if first_user else None,
                       'first_robot_reused_at':_iso(first_bot['_ts']) if first_bot else None,
                       'first_image_seen_at':_iso(first_seen),
                       'first_user_reused_at':hits[0]['user_reused_at'] if hits else None,
                       'last_user_reused_at':hits[-1]['user_reused_at'] if hits else None,
                       'median_seconds_from_latest_robot':statistics.median(gaps) if gaps else None})
    fields = ('user_sends','assistant_sends','assistant_reuse_sends','user_reuses_after_robot','confirmed_robot_sends')
    summary = {k:sum(a[k] for a in assets) for k in fields}
    summary.update(image_count=len(assets),
                   robot_reused_images=sum(a['assistant_reuse_sends']>0 for a in assets),
                   user_reused_images=sum(a['reused_after_robot'] for a in assets),
                   confirmed_images=sum(a['confirmed_robot_sends']>0 for a in assets))
    summary['asset_reuse_rate'] = behavior.ratio(summary['user_reused_images'],summary['robot_reused_images'])
    summary['user_reuses_after_seen'] = (sum(a['user_reuses_after_seen'] or 0 for a in assets)
                                        if summary['confirmed_images'] else None)
    reuse_events.sort(key=lambda r:r['user_reused_at'])
    timeline.sort(key=lambda r:r['sent_at'])
    meta = c.execute("SELECT value FROM behavior_meta WHERE key='image_visibility_started_at'").fetchone()
    return {'version':VERSION, 'generated_at':_iso(now), 'participant':dict(person), 'schedule':plan,
            'collection_started_at':meta[0] if meta else None,
            'period':{'start_at':_iso(begin), 'end_at_exclusive':_iso(end),
                      'observed_until':_iso(min(now,end)) if end is not None else None,
                      'status':'ongoing' if ongoing else 'completed' if begin is not None else 'not_started',
                      'duration_days':28, 'scope':'whole_study', 'send_limit':None},
            'definitions':DEFINITIONS, 'summary':summary, 'assets':assets,
            'reuse_events':reuse_events, 'weekly':weekly, 'timeline':timeline}
