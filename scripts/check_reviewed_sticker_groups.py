"""Diagnostic with the use group supplied, separating visual maximization from use inference."""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import sticker_intent as policy
from research.sticker_encoder.disentangled import identity_reason
from scripts.evaluate_external_sticker_test import OUT,DATA,read,write
from scripts.prepare_sticker_v4 import extract_families

def main():
    truth={}
    with (DATA/'visual-judgments.tsv').open() as f:
        for r in csv.DictReader(f,delimiter='\t'):
            truth[r['id']]=dict(r,uses=r['uses'].split(',') if r['uses'] else [])
    assert hashlib.sha256((DATA/'visual-judgments.tsv').read_bytes()).hexdigest()==read(OUT/'test-lock.json')['visual-judgments.tsv']
    rows=read(OUT/'manifest.json')['rows']; measured=read(OUT/'results.json')['sample']
    with np.load(OUT/'aladin-features.npz') as f:
        assert list(f['ids'])==[r['id'] for r in rows]
        style=f['style'].astype('float64')
    dist=1-style@style.T
    for r,s in zip(rows,measured):
        r.update(medium=s['medium'],pack=r['source_pack'],families=extract_families(r['source_caption']))
    cases=[];checks=0; reruns=0; rng=np.random.default_rng(345625)
    for i,q in enumerate(rows):
        a=truth[q['id']]
        for group in a['uses']:
            if group not in policy.NAMES:continue
            members=[j for j,r in enumerate(rows) if q['id']!=r['id'] and group in truth[r['id']]['uses']]
            if not members:continue
            eligible=[j for j in members if not identity_reason(q,rows[j]) and np.isfinite(dist[i,j]) and dist[i,j]>=policy.MIN_STYLE]
            expected=min(eligible,key=lambda j:(-dist[i,j],rows[j]['id'])) if eligible else None
            choices=[dict(id=rows[j]['id'],style_distance=float(dist[i,j])) for j in eligible]
            for _ in range(10):
                rng.shuffle(choices)
                sorted_choices=policy.descending_style(choices)
                assert ([r['id'] for r in sorted_choices[:1]]==([rows[expected]['id']] if expected is not None else []))
                checks+=1
                reruns+=int(bool(sorted_choices))
            winner=rows[expected]['id'] if expected is not None else None
            style_ok=None
            if winner:
                b=truth[winner]
                style_ok=(a['medium']!=b['medium'] and 'unknown' not in (a['medium'],b['medium'])
                          and not(a['family'] and a['family']==b['family']))
            cases.append(dict(query=q['id'],group=group,group_size=len(members),eligible=len(eligible),winner=winner,
                              style_distance=float(dist[i,expected]) if winner else None,visual_contrast=style_ok))
    d=dict(cases=len(cases),with_winner=sum(bool(r['winner']) for r in cases),unique_queries=len({r['query'] for r in cases}),
           visual_contrast=sum(r['visual_contrast'] is True for r in cases),stable_checks=checks,nonempty_stable_checks=reruns,
           basis='Given visual-code group for reference and gallery; no contribution to model fitting or automatic semantic accuracy.',rows=cases)
    write(OUT/'known-group-test.json',d);print({k:v for k,v in d.items() if k not in ('rows','basis')})

if __name__=='__main__':main()
