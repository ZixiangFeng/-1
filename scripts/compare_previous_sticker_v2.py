"""Add the previously deployed v2 as a frozen comparator; never retrain v3."""
import json,sys,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,torch
import torch.nn.functional as F
from transformers import ChineseCLIPModel,ChineseCLIPProcessor
from research.sticker_encoder.weak_model import digest,rgb,save,retrieval
from evaluate_local_sticker_v3 import common_metrics
out=ROOT/'outputs/sticker-semantic-v3-local-20261118';old=ROOT/'outputs/sticker-semantic-v2-500-20261112';source=ROOT/'outputs/sticker-codex-local-20261118'
if (out/'v2-comparison.json').exists():raise ValueError('Comparator already evaluated')
prior=json.loads((old/'training-report.json').read_text())
for n,h in prior['checkpoint_files'].items():
 if digest(old/'checkpoint'/n)!=h:raise ValueError('v2 changed')
lock=json.loads((out/'checkpoint-lock.json').read_text())
for n,h in lock['checkpoint_files'].items():
 if digest(out/'checkpoint'/n)!=h:raise ValueError('v3 changed')
rows=json.loads((out/'full-dataset.json').read_text())['rows'];f=np.load(out/'evaluation-features.npz');ids=[r['id'] for r in rows];torch.set_num_threads(4);os.environ['TOKENIZERS_PARALLELISM']='false'
model=ChineseCLIPModel.from_pretrained(old/'checkpoint',local_files_only=True,use_safetensors=True).eval().to('mps');proc=ChineseCLIPProcessor.from_pretrained(old/'checkpoint',local_files_only=True);values=[]
with torch.inference_mode():
 for i in range(0,len(rows),32):
  inp=proc(images=[rgb(source/r['image']) for r in rows[i:i+32]],return_tensors='pt').to('mps');values.append(F.normalize(model.get_image_features(**inp),dim=-1).cpu().numpy())
v2=np.concatenate(values);np.savez_compressed(out/'v2-evaluation-features.npz',ids=np.array(ids),v2_image=v2)
original=json.loads((out/'evaluation-report.json').read_text());index={aid:i for i,aid in enumerate(ids)};metrics={}
for group,g in original['metrics'].items():
 ix=[index[a] for a in g['ids']];metrics[group]={'common1250':common_metrics(v2,f['text'],ix),'within_subset':retrieval(v2[ix],f['text'][ix])}
oldix=[index[a] for a in original['metrics']['old_test_all75']['ids']];gallery=[i for i,r in enumerate(rows) if r['id'].startswith('B')];old500={m:common_metrics(v[gallery],f['text'][gallery],oldix) for m,v in [('v1',f['v1_image']),('v2',v2),('v3',f['v3_image'])]}
ix=[index[a] for a in original['metrics']['fresh_test_all75']['ids']];packs=sorted({rows[i]['pack'] for i in ix});groups=[np.array([j for j,i in enumerate(ix) if rows[i]['pack']==p]) for p in packs];rng=np.random.default_rng(20261118);changes={};a=np.array(metrics['fresh_test_all75']['common1250']['ranks']);b=np.array(original['metrics']['fresh_test_all75']['v3']['common1250']['ranks'])
for metric in ['recall_at_1','recall_at_5','mrr']:
 delta=1/b-1/a if metric=='mrr' else (b<=(1 if metric=='recall_at_1' else 5)).astype(float)-(a<=(1 if metric=='recall_at_1' else 5)).astype(float)
 samples=[float(delta[np.concatenate([groups[k] for k in rng.integers(0,len(packs),len(packs))])].mean()) for _ in range(5000)]
 changes[metric]={'delta':float(delta.mean()),'pack_bootstrap95':np.quantile(samples,[.025,.975]).tolist(),'packs':len(packs)}
result={'comparator':'Previously deployed v2, frozen. v3 selection remained locked.','v2_checkpoint_files':prior['checkpoint_files'],'v3_lock_sha256':digest(out/'checkpoint-lock.json'),'metrics':metrics,'old_test_common500':old500,'fresh75_v3_minus_v2':changes}
save(out/'v2-comparison.json',result);print(json.dumps({'fresh75_v2':metrics['fresh_test_all75'],'fresh75_changes':changes,'old_test_common500':old500}),flush=True)
