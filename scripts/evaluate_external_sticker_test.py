"""Locked external image-to-image retrieval with independent, pre-scored visual labels."""
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import sticker_intent as policy
import sticker_ranker
from research.sticker_encoder.aladin_features import StyleEncoder
from research.sticker_encoder.contrast_retrieval import retrieve
from research.sticker_encoder.disentangled import identity_reason
from research.sticker_encoder.reviewed_gallery import reviewed_rows,meaning_vectors
from scripts.evaluate_context_ranker import CORPUS, EXAMPLES, features
from scripts.prepare_sticker_v4 import extract_families

OUT=ROOT/'outputs/sticker-external-holdout-20261119'
DATA=ROOT/'research/sticker_encoder/external'
ARCH=ROOT/'outputs/sticker-divergence-v4-20261119'
FROZEN='a9076a1d854163c019472a5b67f3608bf526129cd242059a691e0613a457f454'
VECFILE=OUT/'aladin-features.npz'


def read(p):
    return json.loads(p.read_text())


def write(p,d):
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def source_meaning(row):
    # Inputs come only from the publisher. Use the communicative gloss when
    # explicitly provided. The independent image review is never put into BGE.
    text=row['source_caption']
    parts=re.split(r'(?<=[，。；;])\s*(?:适合(?:用于|用来)?|常用于|用来|用于|表达(?:了|出)?)',text)
    return (parts[-1] if len(parts)>1 else text).strip('，。；; ') or text


def style_vectors(rows):
    digest=sha(OUT/'manifest.json')
    if VECFILE.exists():
        with np.load(VECFILE,allow_pickle=False) as f:
            if str(f['manifest'])==digest and list(f['ids'])==[r['id'] for r in rows]:
                return f['style']
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    torch.set_num_threads(3)
    encoder=StyleEncoder(ROOT,device)
    vectors=[]
    for i in range(0,len(rows),12):
        batch=[]
        for row in rows[i:i+12]:
            path=OUT/'images'/(row['id']+'.png')
            if sha(path)!=row['image_sha256']:
                raise ValueError('Image changed')
            with Image.open(path) as im:
                batch.append(im.convert('RGB'))
        vectors.extend(encoder.encode(batch))
    vec=np.stack(vectors)
    np.savez_compressed(VECFILE,manifest=digest,ids=[r['id'] for r in rows],style=vec)
    return vec


def metrics(records,ids,truth,rows):
    by={r['id']:r for r in rows}
    valid=[i for i in ids if truth[i]['uses']]
    def correct(q,c,style):
        a,b=truth[q],truth[c]
        if not a['uses'] or not b['uses'] or not a['uses']&b['uses']:
            return False
        if not style:
            return True
        return (a['medium']!=b['medium'] and 'unknown' not in (a['medium'],b['medium'])
                and not(a['family'] and a['family']==b['family']))
    out={}
    for stage,k in [('semantic',5),('divergent',1),('divergent',5)]:
        key=stage+'@'+str(k); end=stage=='divergent'
        possible={q for q in valid if any(q!=c and not identity_reason(by[q],by[c]) and correct(q,c,end) for c in ids)}
        hits={}; count=0; accepted=0; empty=0; unjudged=0; returned_query=0; semonly=0
        for q in valid:
            choices=[v['id'] for v in records[q][stage][:k]]
            successes=sum(correct(q,c,end) for c in choices)
            hits[q]=int(successes>0)
            count+=len(choices);accepted+=successes;empty+=int(not choices);returned_query+=int(bool(choices))
            unjudged+=sum(not truth[c]['uses'] for c in choices)
            semonly+=sum(correct(q,c,False) for c in choices)
        out[key]=dict(judged_queries=len(valid),all_queries=len(ids),answerable=len(possible),
                      with_verified_hit=sum(hits[q] for q in possible),returned_queries=returned_query,
                      returned=count,correct=accepted,returned_with_uncertain_truth=unjudged,
                      meaning_correct=semonly,accepted_per_all_judged_queries=sum(hits.values()),
                      empty=empty,precision=accepted/count if count else None)
    return out


def main():
    manifest=read(OUT/'manifest.json')
    rows=manifest['rows']; ids=[r['id'] for r in rows]
    truth={}
    with (DATA/'visual-judgments.tsv').open() as f:
        for row in csv.DictReader(f,delimiter='\t'):
            row['uses']=set(row['uses'].split(',')) if row['uses'] else set()
            truth[row['id']]=row
    if set(ids)!=set(truth) or len(ids)!=len(set(ids)):
        raise ValueError('Incomplete blind review')
    evidence=[OUT/'manifest.json',DATA/'visual-judgments.tsv',DATA/'context-checks.json',ROOT/'sticker_intent.py']
    lock={p.name:sha(p) for p in evidence}
    lock['checkpoint']=sticker_ranker.metric().digest
    if lock['checkpoint']!=FROZEN: raise ValueError('Wrong model')
    # Keep the original cross-medium test lock for provenance. Only the selection
    # program changed in v2. Neither annotations, sampling nor weights may change.
    first=read(OUT/'test-lock.json')
    if any(lock[k] != first[k] for k in first if k!='sticker_intent.py'):
        raise ValueError('Changed holdout or weights in the full-group recheck')
    lock_path=OUT/'test-lock-full-group.json'
    if lock_path.exists() and read(lock_path)!=lock: raise ValueError('Test inputs or labels changed after freeze')
    if not lock_path.exists(): write(lock_path,lock)
    style=style_vectors(rows)
    m=load_file(str(ARCH/'medium-classifier.safetensors'))
    with torch.inference_mode():
        probs=(torch.tensor(style)@m['weight'].T+m['bias']).softmax(1).numpy()
    cats=read(ARCH/'medium-training-report.json')['categories']
    for r,p in zip(rows,probs):
        r.update(pack=r['source_pack'],families=extract_families(r['source_caption']),quality='clear',
                 meaning=source_meaning(r),medium=cats[int(p.argmax())] if p.max()>=.60 else 'unknown')
    encoder=sticker_ranker.local_encoder(); model=sticker_ranker.metric()
    z=model.project(encoder.encode([r['meaning'] for r in rows]))
    labels=policy.classify(z)
    scores=z@z.T; dist=1-style.astype(np.float64)@style.astype(np.float64).T
    systems={'original':{},'full_group_max':{}};checks=0;perm=np.random.default_rng(572341)
    for i,row in enumerate(rows):
        new=policy.gallery_result(row,rows,labels,labels[i],scores[i],dist[i],identity_reason)
        systems['full_group_max'][row['id']]=new
        systems['original'][row['id']]=retrieve(row,rows,scores[i],dist[i],model.floor)
        eligible=[j for j,r in enumerate(rows) if labels[i] and labels[j]==labels[i] and
                  not identity_reason(row,r) and dist[i,j]>=policy.MIN_STYLE]
        winner=min(eligible,key=lambda j:(-dist[i,j],ids[j])) if eligible else None
        expected=ids[winner] if winner is not None else None
        for _ in range(10):
            order=perm.permutation(len(rows))
            rerun=policy.gallery_result(row,[rows[j] for j in order],[labels[j] for j in order],labels[i],
                                       scores[i,order],dist[i,order],identity_reason)
            if (rerun['maximum'] or {}).get('id')!=expected:
                raise AssertionError('Maximum depends on input order')
            checks+=1
    stats={k:metrics(v,ids,truth,rows) for k,v in systems.items()}
    contexts=read(DATA/'context-checks.json')
    proj=model.project(encoder.encode([s[1] for s in contexts]))
    preds=policy.classify(proj)
    ctx=[dict(id=a,text=b,expected=c,predicted=p) for (a,b,c),p in zip(contexts,preds)]
    old=read(CORPUS)['rows']; corrected=reviewed_rows(old); ix=next(i for i,r in enumerate(old) if r['id']=='E0264')
    if ids.count('E0264') or any(r['pixel_hash'] in {q['pixel_hash'] for q in old} for r in rows):
        raise AssertionError('Known image leaked into external queries')
    oldz=model.project(features(old)); newz=model.project(meaning_vectors(corrected,features(old),encoder))
    diagnostic_text=[EXAMPLES['laugh_together'],contexts[0][1],contexts[1][1]]
    testq=model.project(encoder.encode(diagnostic_text))
    regression=[dict(text=t,old_score=float(a),new_score=float(b),query_group=g,
                     E0264_group=corrected[ix]['reviewed_intent'],same_group=g==corrected[ix]['reviewed_intent'])
                for t,a,b,g in zip(diagnostic_text,testq@oldz[ix],testq@newz[ix],policy.classify(testq))]
    by={r['id']:r for r in rows}
    for r,g in zip(rows,labels):
        r['predicted_group']=g
    sample=[]
    for q in ids:
        pick=systems['full_group_max'][q]['maximum']
        sample.append(dict(id=q,source=by[q]['source'],meaning=by[q]['meaning'],gold=sorted(truth[q]['uses']),
                           visual_read=truth[q]['visual_read'],predicted=by[q]['predicted_group'],medium=by[q]['medium'],
                           actual_medium=truth[q]['medium'],winner=pick['id'] if pick else None,
                           distance=pick['style_distance'] if pick else None,
                           semantic_group_count=len(systems['full_group_max'][q]['semantic'])))
    report=dict(policy=policy.POLICY,images=len(ids),sources=dict(Counter(r['source'] for r in rows)),
                use_group_threshold=policy.MIN_AFFINITY,margin=policy.MIN_LEAD,minimum_style=policy.MIN_STYLE,
                queries_clear=sum(bool(t['uses']) for t in truth.values()),classified=sum(bool(g) for g in labels),
                correctly_classified=sum(g in truth[r['id']]['uses'] for r,g in zip(rows,labels) if truth[r['id']]['uses']),
                judged_classified=sum(bool(g) and bool(truth[r['id']]['uses']) for r,g in zip(rows,labels)),
                stability_checks=checks,stability_query_winners=sum(bool(v['maximum']) for v in systems['full_group_max'].values()),
                tests=stats,context_checks=ctx,regression=regression,sample=sample,
                scoring='All 77 images used as queries against the other 76; visual use labels written by Codex before scoring and independent of source captions, not participant ratings. Same-source series and known duplicates excluded. Unclear query excluded from pragmatic accuracy, see all-query coverage.')
    write(OUT/'predictions.json',systems); write(OUT/'results.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('sample','context_checks','scoring','regression')},ensure_ascii=False),flush=True)


if __name__=='__main__': main()
