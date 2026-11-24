"""Recompute the broad-policy outputs without modifying any fitted checkpoint."""
import json,sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from research.sticker_encoder.broad import broad_rank
from research.sticker_encoder.disentangled import identity_reason
from research.sticker_encoder.weak_model import save,digest
out=ROOT/'outputs/sticker-broad-v5-20261119';v4=ROOT/'outputs/sticker-divergence-v4-20261119'
rows=json.loads((out/'display-dataset.json').read_text())['rows'];by={r['id']:r for r in rows}
archive=np.load(v4/'retrieval-matrices.npz');m={k:archive[k] for k in ('v4','meaning','style')}
results={};counts=Counter();pairs=0
for i,q in enumerate(rows):
    cs=[{**r,'semantic_score':float(m['v4'][i,j]),'context_score':float(m['meaning'][i,j]),'style_distance':float(m['style'][i,j])} for j,r in enumerate(rows) if j!=i]
    r=broad_rank(q,cs)
    for c in r['stage2']:
        assert identity_reason(q,c) is None
        assert q['medium']!=c['medium'] and 'unknown' not in (q['medium'],c['medium'])
        assert c['style_distance']>=.65
        assert set(q['intent_groups'][:1])&set(c['intent_groups'])
    results[q['id']]={'stage1_count':len(r['stage1']), 'stage2':[{k:c[k] for k in ('id','semantic_score','context_score','style_distance','recall_score','semantic_rank_before_style')} for c in r['stage2']],
                     'rejected_counts':dict(Counter(c['reason'] for c in r['rejected']))}
    pairs+=len(r['stage2']);counts.update(c['reason'] for c in r['rejected'])
assert not {c['id'] for c in results['B0032']['stage2']}&{'B0128','B0314','B0403','B0416'}
save(out/'two-stage-results.json',{'policy':'Primary broad pool by default; optional explicit current intent; top20 semantic window then ALADIN rank', 'queries':results})
save(out/'policy-validation.json',{'queries':len(rows),'queries_with_candidates':sum(bool(r['stage2']) for r in results.values()),'returned_pairs':pairs,
     'forbidden_output_violations':0,'rejection_counts':dict(counts),'screenshot_query':results['B0032'],
     'checkpoint_sha256':digest(out/'broad-intent-head.safetensors'),'semantic_adapter_sha256':digest(v4/'semantic-adapter.safetensors'),
     'human_semantic_accuracy':None,'warning':'This validates code invariants and availability, not contextual semantic equivalence.'})
print(json.dumps({'queries':len(rows),'covered':sum(bool(r['stage2']) for r in results.values()),'B0032':results['B0032']}))
