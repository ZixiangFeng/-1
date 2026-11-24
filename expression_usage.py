"""Read actual image sends and their situated interpretations, never inventory guesses."""
import json

POLICY = 'chat-context-only-v3'


def examples(c, user, asset, limit=3):
    rows = c.execute('''SELECT rowid AS seq,* FROM messages WHERE user=?
        AND role='user' AND asset=? ORDER BY rowid DESC LIMIT ?''', (user, asset, limit)).fetchall()
    result = []
    has_evidence = c.execute("SELECT 1 FROM sqlite_master WHERE name='learning_evidence'").fetchone()
    for row in rows:
        r = dict(row)
        before = [dict(x) for x in c.execute('''SELECT id,role,text FROM messages
            WHERE user=? AND session=? AND rowid<? ORDER BY rowid DESC LIMIT 3''',
            (user, r['session'], r['seq']))][::-1]
        after = [dict(x) for x in c.execute('''SELECT id,role,text FROM messages
            WHERE user=? AND session=? AND rowid>? AND role='user' ORDER BY rowid LIMIT 2''',
            (user, r['session'], r['seq']))]
        memory = c.execute('''SELECT * FROM memories WHERE user=? AND source=? AND expression=?
            AND status IN ('inferred','confirmed') ORDER BY CASE status WHEN 'confirmed' THEN 0 ELSE 1 END,
            created DESC,rowid DESC LIMIT 1''', (user, r['id'], 'image:'+asset)).fetchone()
        memory = dict(memory) if memory else None
        evidence = None
        if memory and has_evidence:
            item = c.execute('''SELECT evidence_message,evidence FROM learning_evidence
                WHERE user=? AND memory=?''', (user, memory['id'])).fetchone()
            evidence = dict(item) if item else None
        # An abstained/retracted memory must not resurrect its initial planner guess.
        superseded = c.execute('''SELECT 1 FROM memories WHERE user=? AND source=? AND expression=?
            AND status='superseded' LIMIT 1''',(user,r['id'],'image:'+asset)).fetchone()
        meaning = memory['meaning'] if memory else ('' if superseded else r.get('interpretation',''))
        result.append({'source_message':r['id'], 'created':r.get('created',''), 'session':r['session'],
            'text':r['text'], 'interpretation':meaning or '',
            'status':memory['status'] if memory else ('tentative' if meaning else 'uninterpreted'),
            'context':memory['context'] if memory else '', 'supporting_evidence':evidence,
            'preceding':[dict(x,text=x['text'][:600]) for x in before],
            'subsequent_user':[dict(x,text=x['text'][:600]) for x in after]})
    return result


def context(c, user, current_asset=None):
    rows = [dict(r) for r in c.execute('''SELECT a.id,a.description,MAX(m.rowid) AS latest
        FROM assets a JOIN messages m ON m.asset=a.id AND m.user=a.user
        WHERE a.user=? AND m.role='user' GROUP BY a.id ORDER BY latest DESC''', (user,))]
    if current_asset:
        rows.sort(key=lambda r:r['id'] != current_asset)
    references = []
    for row in rows[:8]:
        references.append({'asset_id':row['id'], 'visible_description':(row['description'] or '')[:700],
                           'uses':examples(c,user,row['id'])})
    if not references:
        return None
    return {'policy':POLICY, 'basis':'actual_user_sends_only', 'references':references}


def divergence_decisions(c, user):
    result = {}
    for row in c.execute("SELECT payload FROM events WHERE user=? AND kind='sticker_selection' ORDER BY rowid DESC",(user,)):
        event = json.loads(row['payload'])
        if event.get('divergence_policy') != POLICY or not event.get('asset_id'):
            continue
        reference = event.get('reference_asset_id')
        context_data = event.get('usage_context') or {}
        if reference not in {r['asset_id'] for r in context_data.get('references',[])}:
            continue
        reply = c.execute("SELECT * FROM messages WHERE id=? AND user=? AND role='assistant' AND asset=?",
                          (event.get('message'),user,event['asset_id'])).fetchone()
        if not reply or not c.execute("SELECT 1 FROM messages WHERE user=? AND role='user' AND asset=?",(user,reference)).fetchone():
            continue
        entries = result.setdefault(reference,[])
        if len(entries)<3:
            entries.append({'message':event['message'],'created':dict(reply).get('created',''),
                'asset_id':event['asset_id'],'reason':event.get('reason',''),
                'source_messages':event.get('source_messages',[])})
    return result
