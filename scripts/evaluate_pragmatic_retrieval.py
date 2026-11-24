"""Paired evaluation on the frozen new sample, with method-blind visual review."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sticker_intent as old
import sticker_pragmatics as new
import sticker_ranker
from research.sticker_encoder.disentangled import identity_reason
from research.sticker_encoder.reviewed_gallery import reviewed_rows, meaning_vectors
from scripts.evaluate_external_sticker_test import source_meaning
from scripts.prepare_sticker_v4 import extract_families
from scripts.train_context_ranker import read, write, CORPUS, features
from scripts.train_pragmatic_intent import OUT, digest

HOLDOUT = OUT/'holdout'
JUDGMENTS = new.DIRECTORY/'holdout-pair-review.tsv'


def style_vectors(rows, directory=HOLDOUT):
    from research.sticker_encoder.aladin_features import StyleEncoder
    import torch
    manifest = digest(directory/'manifest.json')
    store = directory/'style.npz'
    ids = [r['id'] for r in rows]
    if store.exists():
        with np.load(store) as f:
            if str(f['manifest']) != manifest or list(f['ids']) != ids:
                raise ValueError('Mismatched holdout image list')
            return f['style']
    torch.set_num_threads(3)
    encoder = StyleEncoder(ROOT,'mps' if torch.backends.mps.is_available() else 'cpu')
    vectors = []
    for i in range(0,len(rows),8):
        batch = []
        for r in rows[i:i+8]:
            p = directory/'images'/(r['id']+'.png')
            if digest(p) != r['image_sha256']:
                raise ValueError('Unexpected image')
            with Image.open(p) as im:
                batch.append(im.convert('RGBA'))
        vectors.extend(encoder.encode(batch))
    s = np.asarray(vectors)
    np.savez_compressed(store,manifest=manifest,ids=ids,style=s)
    return s


def make_review(key):
    tmp = ROOT/'tmp/holdout-nov19/v7-review'
    tmp.mkdir(parents=True,exist_ok=True)
    for i in range(0,len(key),12):
        sheet = Image.new('RGB',(960,4*172),'#eeefe9')
        draw = ImageDraw.Draw(sheet)
        for j,pair in enumerate(key[i:i+12]):
            x,y = j%3*320,j//3*172
            for offset,base,ext,column in [(0,HOLDOUT/'images','png','query'),(158,ROOT/'outputs/sticker-context-v6/web/images','jpg','candidate')]:
                with Image.open(base/(pair[column]+'.'+ext)) as im:
                    rgba=im.convert('RGBA')
                    rgba.thumbnail((155,143))
                    at=(x+offset+(155-rgba.width)//2,y+3+(143-rgba.height)//2)
                    sheet.paste(rgba,at,rgba)
            draw.text((x+137,y+147),pair['blind_id'],fill='black')
            draw.line((x+319,y,x+319,y+171),fill='white',width=2)
        sheet.save(tmp/('p%02d.jpg'%(i//12+1)),quality=96)
    return (len(key)+11)//12


def check_maximum(result):
    if result['incomplete_style']:
        return result['maximum'] is None
    eligible = [v for v in result['semantic'] if old.distance_gate(v['style_distance']) is None]
    m = min(eligible,key=lambda v:(-v['style_distance'],v['id']),default=None)
    return (m and m['id']) == (result['maximum'] and result['maximum']['id'])


def score_queries(qs, directory, texts):
    if len(texts) != len(qs):
        raise ValueError('Each query requires its own use text')
    gallery = reviewed_rows(read(CORPUS)['rows'])
    base = sticker_ranker.metric()
    z = base.project(meaning_vectors(gallery,features(gallery)))
    coarse = new.model().groups(z,[r['meaning'] for r in gallery])
    fine = old.classify(z)
    labels_old = [r.get('reviewed_intent') or g for r,g in zip(gallery,fine)]
    labels_new = [new.family(r.get('reviewed_intent')) or g for r,g in zip(gallery,coarse)]
    for r in qs:
        r.update(pack=r['source_pack'],families=extract_families(r['source_caption']))
    qz = base.project(sticker_ranker.local_encoder().encode(texts))
    qold = old.classify(qz)
    qnew = new.model().groups(qz,texts)
    qstyle = style_vectors(qs, directory)
    with np.load(ROOT/'outputs/sticker-semantic-v3-local-20261118/all-style-features.npz') as a:
        if list(a['ids']) != [r['id'] for r in gallery]:
            raise ValueError('Unexpected style matrix')
        distances = 1-qstyle.astype('float64') @ a['style'].astype('float64').T
    records = {}
    pairs = set()
    for i,r in enumerate(qs):
        scores = z @ qz[i]
        versions = {
            '27':old.gallery_result(r,gallery,labels_old,qold[i],scores,distances[i],identity_reason),
            '8':new.gallery_result(r,gallery,labels_new,qnew[i],texts[i],scores,distances[i],identity_reason,new.model().pair_floor)}
        records[r['id']] = {}
        for name,v in versions.items():
            if not check_maximum(v):
                raise AssertionError('Not a full-group maximum')
            row = {'group':v['group'],'family_candidates':v.get('broad_count',v['group_size']),
                   'interchangeable':v['group_size'],'style_eligible':len(v['divergent'])}
            for stage in ('semantic','divergent'):
                top = v[stage][0] if v[stage] else None
                row[stage] = top
                if top:
                    pairs.add((r['id'],top['id']))
            records[r['id']][name] = row
    return records, pairs


def generate():
    lock = read(OUT/'training.json')
    if (lock['frozen_external_sample'] != digest(HOLDOUT/'manifest.json') or
        lock['sha256'] != new.model().digest or lock['inputs']['sticker_pragmatics.py'] != digest(ROOT/'sticker_pragmatics.py')):
        raise ValueError('Evaluation requires locked inputs')
    qs = read(HOLDOUT/'manifest.json')['rows']
    records, pairs = score_queries(qs, HOLDOUT, [source_meaning(r) for r in qs])
    pairs = sorted(pairs)
    np.random.default_rng(51938026).shuffle(pairs)
    key = [dict(blind_id='J%03d'%(i+1),query=q,candidate=c) for i,(q,c) in enumerate(pairs)]
    for path,obj in ((OUT/'blind-key.json',key),(OUT/'rankings.json',records)):
        if path.exists() and read(path) != obj:
            raise ValueError('Prediction changed after lock')
        if not path.exists():
            write(path,obj)
    print('Blind sheets:',make_review(key),'pairs:',len(key))


def summarize():
    key=read(OUT/'blind-key.json')
    records=read(OUT/'rankings.json')
    with JUDGMENTS.open() as f:
        labels=list(csv.DictReader(f,delimiter='\t'))
    if len(labels)!=len(key) or {r['blind_id'] for r in labels}!={r['blind_id'] for r in key}:
        raise ValueError('Incomplete blind judgments')
    by_blind={r['blind_id']:r for r in labels}
    judged={(r['query'],r['candidate']):by_blind[r['blind_id']] for r in key}
    result={}
    for method in ('27','8'):
        table=result[method]={'assigned':sum(bool(s[method]['group']) for s in records.values())}
        for stage in ('semantic','divergent'):
            rr=[judged[q,s[method][stage]['id']] for q,s in records.items() if s[method][stage]]
            if any(v[k] not in ('yes','no','uncertain') for v in rr for k in ('intent','style')):
                raise ValueError('Invalid coding')
            table[stage]=dict(returned=len(rr),same_meaning=sum(r['intent']=='yes' for r in rr),
                different_meaning=sum(r['intent']=='no' for r in rr),unclear_meaning=sum(r['intent']=='uncertain' for r in rr),
                both=sum(r['intent']=='yes' and r['style']=='yes' for r in rr))
    report=dict(queries=len(records),gallery=1250,method='Frozen paired holdout; same public images and ALADIN features. Independent files from the same two public source repositories. One Codex visual judgment, blind to method, rank and scores, with no participant context.',
                model=digest(new.DIRECTORY/'language-intent.npz'),sample=digest(HOLDOUT/'manifest.json'),
                key_sha256=digest(OUT/'blind-key.json'),judgments_sha256=digest(JUDGMENTS),
                full_maximum_verified_on_all=len(records),results=result)
    write(OUT/'evaluation.json',report)
    print(json.dumps(report['results'],ensure_ascii=False))


if __name__=='__main__':
    if sys.argv[1:]==['--summarize']:
        summarize()
    else:
        generate()
