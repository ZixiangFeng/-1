"""Per-participant D library views derived from completed, evidenced chat turns.

Retrieval candidates are not treated as confirmed divergence. No model calls,
inventory guesses, or cross-participant choices are made when viewing a report.
"""
import json
import expression_usage
import divergence


def report(c, uid):
    participant=c.execute('SELECT arm FROM users WHERE id=?',(uid,)).fetchone()
    if not participant or participant['arm']!='D':return None
    actual={r['id']:dict(r) for r in c.execute('''SELECT a.id,a.image,a.description,COUNT(*) AS uses
        FROM assets a JOIN messages m ON m.user=a.user AND m.asset=a.id
        WHERE a.user=? AND m.role='user' GROUP BY a.id''',(uid,))}
    result={'basis':'actual_user_sends_only','source':'G_public_plus_legacy_materials',
            'used_reference_count':len(actual),'latest':None,'adopted':[],
            'current_contrast_policy':divergence.POLICY,'contrast_threshold':divergence.MIN_SCORE}
    if not actual:return result
    materials={r['id']:dict(r) for r in c.execute('''SELECT g.id,g.image,g.enabled,
        COALESCE(m.style,'') AS style FROM generic_stickers g
        LEFT JOIN sticker_library_metadata m ON m.asset=g.id''')}
    adopted={}
    for row in c.execute("SELECT payload FROM events WHERE user=? AND kind='sticker_selection' ORDER BY rowid DESC",(uid,)):
        try:entry=json.loads(row['payload'])
        except (ValueError,TypeError):continue
        if entry.get('condition')!='D' or entry.get('divergence_policy')!=expression_usage.POLICY:continue
        message=entry.get('message')
        reply=c.execute("SELECT * FROM messages WHERE id=? AND user=? AND role='assistant'",(message,uid)).fetchone()
        if not reply:continue
        refs=[]
        for ref in (entry.get('usage_context') or {}).get('references',[]):
            aid=ref.get('asset_id')
            if aid not in actual:continue
            uses=[]
            for use in ref.get('uses',[]):
                source=c.execute("SELECT id,text FROM messages WHERE id=? AND user=? AND role='user' AND asset=?",
                                 (use.get('source_message'),uid,aid)).fetchone()
                if source:uses.append({'source_message':source['id'],'text':source['text'],
                                      'interpretation':use.get('interpretation',''),'status':use.get('status','tentative')})
            if uses:refs.append({'asset_id':aid,'image':actual[aid]['image'],
                                 'description':actual[aid]['description'],'uses':uses})
        if not refs:continue
        refids={x['asset_id'] for x in refs}
        options=[]
        for option in entry.get('candidates',[]):
            aid=option.get('asset_id')
            if aid in materials:
                options.append(dict(materials[aid],description=option.get('description',''),
                                    usages=option.get('usages',[])))
        chosen=entry.get('asset_id')
        reference=entry.get('reference_asset_id')
        valid_chosen=chosen in materials and chosen==dict(reply).get('asset') and reference in refids and any(o['id']==chosen for o in options)
        if result['latest'] is None:
            result['latest']={'message':message,'created':dict(reply).get('created',''),
                'reply':reply['text'],'references':refs,'candidates':options,
                'selected_asset':chosen if valid_chosen else None,'reason':entry.get('reason',''),
                'contrast_policy':entry.get('contrast_policy'),'contrast':entry.get('contrast'),
                'rejected_asset_id':entry.get('rejected_asset_id'),
                'failed':entry.get('failed',False),'error_stage':entry.get('error_stage'),
                'error_type':entry.get('error_type'),'selection_attempts':entry.get('selection_attempts',[])}
        if valid_chosen:
            target=adopted.setdefault(chosen,dict(materials[chosen],send_count=0,decisions=[]))
            target['send_count']+=1
            target['decisions'].append({'message':message,'created':dict(reply).get('created',''),
                'reference_asset':reference,'source_messages':[u['source_message'] for r in refs if r['asset_id']==reference for u in r['uses']],
                'reason':entry.get('reason',''),'contrast':entry.get('contrast'),
                'contrast_policy':entry.get('contrast_policy')})
    result['adopted']=list(adopted.values())
    return result
