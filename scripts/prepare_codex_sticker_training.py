# -*- coding: utf-8 -*-
"""Build an offline Codex-AI annotated corpus without touching archived labels.

No network calls. Fresh test labels are deliberately absent until checkpoint lock.
"""
import json, os, shutil, sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
from research.sticker_encoder.weak_model import digest,save
from research.sticker_encoder.continuation import validate_continuation

def prepare(out):
    archive=ROOT/'outputs/sticker-corpus-v2-20261112'
    parent=ROOT/'outputs/sticker-semantic-v1-20261111'
    old=json.loads((parent/'dataset.json').read_text())
    manifest=json.loads((archive/'manifest.json').read_text())
    raw={}
    for path in sorted((out/'annotations').glob('*.tsv')):
        expected=json.loads((out/'contact-sheets'/(path.stem+'.json')).read_text())
        records=[]
        for line in path.read_text().splitlines():
            aid,meaning,quality=line.split('|')
            if aid in raw or quality not in ('clear','uncertain') or not meaning: raise ValueError('Invalid annotation '+aid)
            raw[aid]={'id':aid,'meaning':meaning,'quality':quality,'label_source':'codex_direct_visual_ai',
                      'source':'Codex direct image inspection; AI weak annotation, NOT human gold',
                      'human_reviewed':False,'context':'No personal chat context; possible generic communicative meaning only',
                      'source_sheet':path.stem,'label_version':'2026-11-18-v1'}
            records.append(aid)
        if records!=expected: raise ValueError('Sheet IDs mismatch '+str(path))
    rows=[]; missing=[]
    (out/'images').mkdir(exist_ok=True)
    for item in manifest['items']:
        r=dict(item); aid=r['id']; target=out/r['image']
        if not target.exists(): os.link(archive/r['image'],target)
        ap=out/'annotations'/(aid+'.json')
        if aid.startswith('B'):
            if not ap.exists(): shutil.copy2(archive/'annotations'/(aid+'.json'),ap)
            r['quality']='legacy_weak'
        elif aid in raw:
            if not ap.exists(): save(ap,raw[aid])
            elif json.loads(ap.read_text())!=raw[aid]: raise ValueError('Annotation edit requires new run: '+aid)
            r['quality']=raw[aid]['quality']; r['label_source']='codex_direct_visual_ai'; r['human_reviewed']=False
        else:
            if r['split']!='test': raise ValueError('Missing development label '+aid)
            missing.append(aid); continue
        a=json.loads(ap.read_text()); r.update(text=a['meaning'],annotation_sha256=digest(ap),image_sha256=digest(target))
        rows.append(r)
    splits={s:[r['id'] for r in rows if r['split']==s] for s in ['train','validation','test']}
    data={**old,'source':str(out.resolve()),'rows':rows,'splits':splits,'counts':{s:len(v) for s,v in splits.items()},
          'unlabeled_fresh_test':missing,'label_source':'Archived500 Doubao + NEW Codex direct visual AI labels. No new paid API calls.',
          'quality_counts':dict(Counter((r['split']+':'+r['quality']) for r in rows)),
          'annotation_protocol_sha256':digest(out/'labeling-protocol.json'),
          'test_note':'New E test labels remain absent until checkpoint selection is locked; old B test is previously inspected.'}
    validate_continuation(data,old); save(out/'dataset.json',data)
    import subprocess
    subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/local_caption_features.py'),str(out)],check=True)
    print(json.dumps({'counts':data['counts'],'quality':data['quality_counts'],'unlabeled_fresh_test':len(missing)},ensure_ascii=False),flush=True)

if __name__=='__main__': prepare(ROOT/'outputs/sticker-codex-local-20261118')
