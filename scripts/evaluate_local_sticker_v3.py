# -*- coding: utf-8 -*-
"""Evaluate the locked local v3 once; never fit parameters with test images."""
import json,os,sys,time,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from transformers import ChineseCLIPModel,ChineseCLIPProcessor
from research.sticker_encoder.weak_model import digest,retrieval,rgb,save

def common_metrics(images,texts,indices):
 score=images[indices]@texts.T
 ranks=np.argsort(np.argsort(-score,axis=1,kind='stable'),axis=1)[np.arange(len(indices)),indices]+1
 return {'n':len(indices),'gallery':len(texts),'recall_at_1':float(np.mean(ranks==1)),'recall_at_5':float(np.mean(ranks<=5)),'mrr':float(np.mean(1/ranks)),'ranks':ranks.tolist()}

def run():
 out=ROOT/'outputs/sticker-semantic-v3-local-20261118';source=ROOT/'outputs/sticker-codex-local-20261118';parent=ROOT/'outputs/sticker-semantic-v1-20261111'
 lock=json.loads((out/'checkpoint-lock.json').read_text())
 for n,h in lock['checkpoint_files'].items():
  if digest(out/'checkpoint'/n)!=h:raise ValueError('Selected model changed')
 if digest(out/'training-report.json')!=lock['report_sha256']:raise ValueError('Training report changed')
 if (out/'evaluation-report.json').exists():raise ValueError('Evaluation already frozen; inspect existing results instead')
 data=json.loads((source/'dataset.json').read_text());manifest=json.loads((ROOT/'outputs/sticker-corpus-v2-20261112/manifest.json').read_text())
 rows=copy.deepcopy(data['rows']);existing={r['id'] for r in rows}
 raw={}
 for path in sorted((source/'annotations').glob('test-*.tsv')):
  expected=json.loads((source/'contact-sheets'/(path.stem+'.json')).read_text());parsed=[]
  for line in path.read_text().splitlines():
   aid,text,quality=line.split('|');parsed.append(aid)
   if aid in raw or quality not in ('clear','uncertain'):raise ValueError('Invalid test annotation')
   raw[aid]={'id':aid,'meaning':text,'quality':quality,'label_source':'codex_direct_visual_ai','human_reviewed':False,'source':'Codex direct visual AI annotation after checkpoint lock; not human gold','checkpoint_lock_sha256':digest(out/'checkpoint-lock.json')}
  if parsed!=expected:raise ValueError('Test sheet mismatch')
 missing={r['id'] for r in manifest['items'] if r['id'] not in existing}
 if missing!=set(raw) or len(missing)!=75:raise ValueError('All75 fresh holdouts required')
 for row in manifest['items']:
  if row['id'] in existing:continue
  r=dict(row);a=raw[r['id']];ap=source/'annotations'/(r['id']+'.json');save(ap,a)
  r.update(text=a['meaning'],quality=a['quality'],label_source='codex_direct_visual_ai',human_reviewed=False,image_sha256=digest(source/r['image']),annotation_sha256=digest(ap));rows.append(r)
 rows.sort(key=lambda r:r['id']);ids=[r['id'] for r in rows]
 for r in rows:
  if digest(source/r['image'])!=r['image_sha256'] or digest(source/'annotations'/(r['id']+'.json'))!=r['annotation_sha256']:raise ValueError('Source changed')
 full={**data,'rows':rows,'counts':{s:sum(r['split']==s for r in rows) for s in ['train','validation','test']},'unlabeled_fresh_test':[]};save(out/'full-dataset.json',full)
 os.environ['TOKENIZERS_PARALLELISM']='false';torch.set_num_threads(4);device='mps';processor=ChineseCLIPProcessor.from_pretrained(parent/'checkpoint',local_files_only=True)
 pixels=processor(images=[rgb(source/r['image']) for r in rows],return_tensors='pt')['pixel_values'];features={};text=None
 with torch.inference_mode():
  for name,folder in [('v1',parent/'checkpoint'),('v3',out/'checkpoint')]:
   model=ChineseCLIPModel.from_pretrained(folder,local_files_only=True,use_safetensors=True).eval().to(device)
   vals=[]
   for i in range(0,len(rows),32):vals.append(F.normalize(model.get_image_features(pixel_values=pixels[i:i+32].to(device)),dim=-1).cpu().numpy())
   features[name]=np.concatenate(vals)
   tv=[]
   for i in range(0,len(rows),32):
    inp=processor(text=[r['text'] for r in rows[i:i+32]],padding=True,truncation=True,max_length=64,return_tensors='pt').to(device);tv.append(F.normalize(model.get_text_features(**inp),dim=-1).cpu().numpy())
   current=np.concatenate(tv)
   if text is None:text=current
   elif not np.allclose(text,current,atol=1e-6):raise ValueError('Text tower changed unexpectedly')
   del model;torch.mps.empty_cache();print(json.dumps({'encoded':name,'n':len(rows)}),flush=True)
 np.savez_compressed(out/'evaluation-features.npz',ids=np.array(ids),v1_image=features['v1'],v3_image=features['v3'],text=text)
 groups={'train_eligible':[i for i,r in enumerate(rows) if r['split']=='train' and r['quality']!='uncertain'], 'validation_eligible':[i for i,r in enumerate(rows) if r['split']=='validation' and r['quality']!='uncertain'], 'old_test_all75':[i for i,r in enumerate(rows) if r['split']=='test' and r['id'].startswith('B')], 'fresh_test_all75':[i for i,r in enumerate(rows) if r['split']=='test' and r['id'].startswith('E')], 'fresh_test_clear':[i for i,r in enumerate(rows) if r['split']=='test' and r['id'].startswith('E') and r['quality']=='clear']}
 metrics={}
 for group,indices in groups.items():
  metrics[group]={'ids':[ids[i] for i in indices]}
  for name,v in features.items():metrics[group][name]={'common1250':common_metrics(v,text,indices),'within_subset':retrieval(v[indices],text[indices])}
 # Descriptive source-pack bootstrap on new75; no new hyperparameter choices.
 ix=groups['fresh_test_all75'];packs=sorted({rows[i]['pack'] for i in ix});rng=np.random.default_rng(20261118);changes={}
 for metric in ['recall_at_1','recall_at_5','mrr']:
  a=np.array(metrics['fresh_test_all75']['v1']['common1250']['ranks']);b=np.array(metrics['fresh_test_all75']['v3']['common1250']['ranks'])
  delta=(1/b-1/a) if metric=='mrr' else ((b<=(1 if metric=='recall_at_1' else 5)).astype(float)-(a<=(1 if metric=='recall_at_1' else 5)).astype(float))
  packidx=[np.array([j for j,i in enumerate(ix) if rows[i]['pack']==p]) for p in packs]
  samples=[float(delta[np.concatenate([packidx[k] for k in rng.integers(0,len(packs),len(packs))])].mean()) for _ in range(5000)]
  changes[metric]={'delta':float(delta.mean()),'pack_bootstrap95':np.quantile(samples,[.025,.975]).tolist(),'packs':len(packs)}
 report={'created_unix':time.time(),'checkpoint_lock_sha256':digest(out/'checkpoint-lock.json'),'full_dataset_sha256':digest(out/'full-dataset.json'),'metrics':metrics,'fresh75_paired_pack_changes':changes,'test_label_quality':{q:sum(rows[i]['quality']==q for i in groups['fresh_test_all75']) for q in ['clear','uncertain']},'interpretation':'Paired AI-caption retrieval proxy only. Exact-caption R@5 is not human semantic P@5 and cannot prove conversational synonymy.','limits':['Fresh75 images come from previously held-out source packs, not a new external source.','Old75 were previously inspected; show as secondary only.','Uncertain labels retained in all75 proxy; clear-only result is a descriptive subgroup chosen before scoring.','No test-based threshold or checkpoint tuning.','AI pair review is not independent human validation.'],'paid_api_calls':0}
 save(out/'evaluation-report.json',report);print(json.dumps({'event':'evaluation_frozen','fresh75':metrics['fresh_test_all75'],'changes':changes},ensure_ascii=False),flush=True)
if __name__=='__main__':run()
