"""Test new queries against the real 1,250-item gallery. Blind max/semantic-first pairs."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import sticker_intent as policy
import sticker_ranker
from scripts.evaluate_external_sticker_test import OUT,DATA,source_meaning,read,write
from scripts.evaluate_context_ranker import CORPUS,features
from research.sticker_encoder.reviewed_gallery import reviewed_rows,meaning_vectors
from research.sticker_encoder.disentangled import identity_reason
from scripts.prepare_sticker_v4 import extract_families

def main():
    model=sticker_ranker.metric()
    old=reviewed_rows(read(CORPUS)['rows'])
    z=model.project(meaning_vectors(old,features(old)))
    labels=[r.get('reviewed_intent') or g for r,g in zip(old,policy.classify(z))]
    q=read(OUT/'manifest.json')['rows']
    summary=read(OUT/'results.json')
    for r,s in zip(q,summary['sample']):
        r.update(medium=s['medium'],pack=r['source_pack'],families=extract_families(r['source_caption']))
    qz=model.project(sticker_ranker.local_encoder().encode([source_meaning(r) for r in q]))
    qs=np.load(OUT/'aladin-features.npz')
    oldstyle=np.load(ROOT/'outputs/sticker-semantic-v3-local-20261118/all-style-features.npz')
    assert list(oldstyle['ids'])==[r['id'] for r in old] and list(qs['ids'])==[r['id'] for r in q]
    d=1-qs['style'].astype('float64')@oldstyle['style'].astype('float64').T
    records={}; keys=[]
    for i,(r,s) in enumerate(zip(q,summary['sample'])):
        result=policy.gallery_result(r,old,labels,s['predicted'],z@qz[i],d[i],identity_reason)
        records[r['id']]=result
        winners={a[0]['id'] for a in (result['semantic'],result['divergent']) if a}
        keys.extend((r['id'],c) for c in sorted(winners))
    np.random.default_rng(397467).shuffle(keys)
    write(OUT/'gallery-rankings.json',records)
    old_keys = read(OUT/'previous-medium-filter/gallery-review-key.json')
    prior = {(p['query'],p['candidate']):p['blind_id'] for p in old_keys}
    novel = {}
    pairs = []
    for a,b in keys:
        if (a,b) not in prior:
            novel[a,b] = 'N%03d'%(len(novel)+1)
        pairs.append({'blind_id':prior.get((a,b)) or novel[a,b],'query':a,'candidate':b})
    write(OUT/'gallery-review-key.json',pairs)
    tmp=ROOT/'tmp/holdout-nov19/sheets'
    extra = [p for p in pairs if p['blind_id'].startswith('N')]
    for i in range(0,len(extra),16):
        sheet=Image.new('RGB',(920,4*158),'#eff2ef');draw=ImageDraw.Draw(sheet)
        for j,item in enumerate(extra[i:i+16]):
            x=j%4*230; y=j//4*158
            for col,path in enumerate((OUT/'images'/(item['query']+'.png'),
                                      ROOT/'outputs/sticker-context-v6/web/images'/(item['candidate']+'.jpg'))):
                with Image.open(path) as im:
                    im.thumbnail((109,125));sheet.paste(im,(x+col*112+(112-im.width)//2,y+3+(125-im.height)//2))
            draw.text((x+82,y+130),item['blind_id'],fill='black')
        sheet.save(tmp/('strict-pairs-%02d.jpg'%i),quality=94)
    print('queries',len(q),'blind pairs',len(pairs),'with winner',sum(bool(r['maximum']) for r in records.values()))
    with (DATA/'gallery-pair-judgments.tsv').open() as f:
        labels={r['blind_id']:r for r in csv.DictReader(f,delimiter='\t')}
    review=DATA/'gallery-pair-judgments-full-group.tsv'
    if extra and not review.exists():return
    if extra:
        with review.open() as f:
            new_labels={r['blind_id']:r for r in csv.DictReader(f,delimiter='\t')}
        if set(new_labels)!={p['blind_id'] for p in extra}: raise ValueError('Incomplete new blind review')
        labels.update(new_labels)
    judged={(r['query'],r['candidate']):labels[r['blind_id']] for r in pairs}
    evaluation={}
    for kind in ('semantic','divergent'):
        out=[]
        for qid,r in records.items():
            if r[kind]:
                top=r[kind][0];label=judged[qid,top['id']]
                out.append({'query':qid,**top,**label})
        evaluation[kind]={'returned':len(out),'meaning_accepted':sum(r['intent']=='yes' for r in out),
                          'both_accepted':sum(r['intent']=='yes' and r['style']=='yes' for r in out),
                          'unclear_meaning':sum(r['intent']=='uncertain' for r in out),'choices':out}
    write(OUT/'gallery-evaluation.json',evaluation)
    print({k:{a:b for a,b in v.items() if a!='choices'} for k,v in evaluation.items()})

if __name__=='__main__':main()
