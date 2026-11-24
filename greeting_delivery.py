"""Short-lived, user-scoped greeting drafts. Unseen drafts are never history."""
import secrets
import threading
import time


FIRST_OPENING = '嗨，我是小屿～很高兴认识你！\n想先认识一下你，我可以怎么称呼你呀？'


def return_opening(local_hour):
    # Safe timeout fallback: no claims about earlier plans, sleep or mood.
    prefix = '早上好呀' if 5 <= local_hour < 11 else ('下午好呀' if 11 <= local_hour < 18 else ('晚上好呀' if local_hour >= 18 else '你回来啦'))
    return prefix + '～我在呢，想接着聊还是换个话题都可以。'


def first_opening(c, user, created, event):
    """An authored, immediate introduction, shared by all experimental arms.

    Call only after consent and on entering an empty conversation. The write
    lock makes simultaneous mobile/desktop arrivals idempotent. Never insert a
    retroactive greeting once any conversation message already exists.
    """
    c.execute('BEGIN IMMEDIATE')
    if c.execute('SELECT 1 FROM messages WHERE user=? LIMIT 1',(user,)).fetchone():
        return {'created':False,'reason':'conversation_exists'}
    session=c.execute('SELECT id FROM sessions WHERE user=? ORDER BY created DESC LIMIT 1',(user,)).fetchone()
    sid=session['id'] if session else secrets.token_hex(12)
    if not session:
        c.execute('INSERT INTO sessions VALUES(?,?,?)',(sid,user,created))
        event(c,user,'session_started',{'session':sid,'source':'first_opening'})
    mid=secrets.token_hex(16);visit='initial_'+user
    c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?)',
              (mid,user,sid,'assistant',FIRST_OPENING,None,'[]','',created))
    c.execute('INSERT INTO visits VALUES(?,?,?,?,?)',(visit,user,sid,FIRST_OPENING,created))
    event(c,user,'proactive_greeting',{'session':sid,'visit':visit,'message':mid,
        'kind':'first_opening','policy':'first-opening-v1',
        'stats':{'source':'authored-opening','model_used':False}})
    return {'created':True,'id':mid,'session':sid}


class GreetingDelivery:
    def __init__(self, ttl=180):
        self.lock = threading.RLock()
        self.pending = {}
        self.cancelled = {}
        self.ttl = ttl

    def begin(self, user, visit, session, baseline, fallback=False):
        with self.lock:
            self.cancelled = {k: t for k, t in self.cancelled.items() if time.monotonic()-t < self.ttl}
            self.pending = {u: d for u, d in self.pending.items()
                            if time.monotonic() - d['started'] < self.ttl}
            if (user, visit) in self.cancelled:
                return None
            # Multiple tabs must not queue competing opening messages.
            if user in self.pending:
                prior=self.pending[user]
                if not fallback or prior['visit'] != visit or prior['baseline'] != baseline:
                    return None
                if 'reply' in prior:
                    return prior
                # Replace the slow draft. Its eventual result cannot be committed.
            draft = dict(id=secrets.token_hex(16), user=user, visit=visit,
                         session=session, baseline=baseline,
                         started=time.monotonic())
            self.pending[user] = draft
            return draft

    def active(self, draft):
        with self.lock:
            return (self.pending.get(draft['user']) is draft
                    and time.monotonic() - draft['started'] < self.ttl)

    def cancel(self, user, visit=None):
        with self.lock:
            draft = self.pending.get(user)
            target=visit or (draft['visit'] if draft else None)
            if target:
                self.cancelled[(user,target)]=time.monotonic()
            if draft and (visit is None or draft['visit'] == visit):
                self.pending.pop(user, None)


delivery = GreetingDelivery()
