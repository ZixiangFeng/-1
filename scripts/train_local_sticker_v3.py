# -*- coding: utf-8 -*-
"""Offline v3 experiment: new Codex weak captions + AI-reviewed image pairs.

No test labels, no test metrics, no network access, no production model change.
"""
import copy,json,os,random,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from transformers import ChineseCLIPModel,ChineseCLIPProcessor
from research.sticker_encoder.weak_model import contrastive_loss,digest,retrieval,rgb,save
from research.sticker_encoder.continuation import validate_continuation
from research.sticker_encoder.local_pairs import validate_pairs,image_pair_loss

def run():
 source=ROOT/'outputs/sticker-codex-local-20261118';out=ROOT/'outputs/sticker-semantic-v3-local-20261118';parent=ROOT/'outputs/sticker-semantic-v1-20261111'
 if out.exists():raise ValueError('New output directory required')
 data=json.loads((source/'dataset.json').read_text());old=json.loads((parent/'dataset.json').read_text());prior=json.loads((parent/'training-report.json').read_text())
 validate_continuation(data,old)
 pairs=json.loads((source/'pair-review.json').read_text())['accepted'];validate_pairs(pairs,data['rows'])
 if not pairs:raise ValueError('No accepted auxiliary pairs')
 for n,h in prior['checkpoint_files'].items():
  if digest(parent/'checkpoint'/n)!=h:raise ValueError('Parent modified')
 for r in data['rows']:
  if digest(source/r['image'])!=r['image_sha256'] or digest(source/'annotations'/(r['id']+'.json'))!=r['annotation_sha256']:raise ValueError('Data modified')
 # Exclude old test entirely from training process, including preprocessing.
 rows=[r for r in data['rows'] if r['split'] in ('train','validation') and r['quality']!='uncertain']
 if any(r['split']=='test' for r in rows):raise ValueError('Test leakage')
 ix={r['id']:i for i,r in enumerate(rows)}
 protocol={'seed':20261118,'parent':'v1','epochs_max':10,'patience':3,'pair_weights':[0.,.2],
 'backbone_lr':3e-6,'projection_lr':3e-5,'batch_size':32,'pair_batch_size':16,'memory_execution':'Sequential caption backward then pair backward, at most32 images per forward; same weighted objective','negative_margin':.2,
 'selection':'Highest validation image-to-paired-caption MRR among epoch0 and both trials; earlier retained on ties.',
 'validation_gallery':'All eligible validation captions, same gallery in all epochs and trials.',
 'test_access':'No old or new test image encoded, labeled or scored by this trainer. Fresh75 evaluated after immutable checkpoint lock.',
 'supervision':'Archived Doubao captions + new Codex direct visual AI captions and Codex AI pair judgments; not human gold.',
 'quality_policy':'New uncertain captions excluded from fitting and checkpoint selection; retained for audit.',
 'data_sha256':digest(source/'dataset.json'),'pairs_sha256':digest(source/'pair-review.json'),
 'counts':{s:sum(r['split']==s for r in rows) for s in ['train','validation']},'paid_api_calls':0}
 out.mkdir();save(out/'protocol.json',protocol);shutil.copy2(source/'dataset.json',out/'dataset.json');shutil.copy2(source/'pair-review.json',out/'pair-review.json')
 started=time.time();torch.set_num_threads(4);device='mps';os.environ['TOKENIZERS_PARALLELISM']='false'
 model=ChineseCLIPModel.from_pretrained(parent/'checkpoint',local_files_only=True,use_safetensors=True).to(device)
 processor=ChineseCLIPProcessor.from_pretrained(parent/'checkpoint',local_files_only=True)
 for n,p in model.named_parameters():p.requires_grad_(n in prior['trainable_parameter_names'])
 trainable={n:p for n,p in model.named_parameters() if p.requires_grad};initial={n:p.detach().cpu().clone() for n,p in trainable.items()}
 model.eval();pixels=processor(images=[rgb(source/r['image']) for r in rows],return_tensors='pt')['pixel_values'];text=[]
 with torch.inference_mode():
  for a in range(0,len(rows),32):
   token=processor(text=[r['text'] for r in rows[a:a+32]],padding=True,truncation=True,max_length=64,return_tensors='pt').to(device)
   text.append(F.normalize(model.get_text_features(**token),dim=-1).cpu())
 texts=torch.cat(text).clone();tf=np.load(source/'teacher-features.npz');ti={str(a):i for i,a in enumerate(tf['ids'])};teacher=torch.from_numpy(np.array([tf['meaning'][ti[r['id']]] for r in rows]));sim=teacher@teacher.T
 captionmap={t:i for i,t in enumerate(sorted({r['text'] for r in rows}))};caption_ids=[captionmap[r['text']] for r in rows]
 train=[i for i,r in enumerate(rows) if r['split']=='train'];val=[i for i,r in enumerate(rows) if r['split']=='validation']
 @torch.inference_mode()
 def encode(indices):
  model.eval();return torch.cat([F.normalize(model.get_image_features(pixel_values=pixels[indices[a:a+32]].to(device)),dim=-1).cpu() for a in range(0,len(indices),32)]).numpy()
 def restore(state):
  with torch.no_grad():
   for n,p in trainable.items():p.copy_(state[n].to(device))
 baseline=encode(list(range(len(rows))));bv=retrieval(baseline[val],texts[val].numpy());best={'score':bv['image_to_caption']['mrr'],'trial':0,'epoch':0,'weight':0.};beststate=copy.deepcopy(initial);trials=[]
 print(json.dumps({'event':'start','counts':protocol['counts'],'pairs':len(pairs),'baseline_validation':bv}),flush=True)
 for trial,weight in enumerate(protocol['pair_weights'],1):
  restore(initial);random.seed(protocol['seed']);np.random.seed(protocol['seed']);torch.manual_seed(protocol['seed'])
  opt=torch.optim.AdamW([{'params':[p for n,p in trainable.items() if not n.startswith('visual_projection')],'lr':protocol['backbone_lr']},{'params':list(model.visual_projection.parameters()),'lr':protocol['projection_lr']}],weight_decay=.01)
  stale=0;localbest=bv['image_to_caption']['mrr'];records=[]
  for epoch in range(1,protocol['epochs_max']+1):
   order=train.copy();random.shuffle(order);pairorder=pairs.copy();random.shuffle(pairorder);pt=0;losses=[]
   for start in range(0,len(order),32):
    batch=order[start:start+32]
    if len(batch)<2:continue
    model.eval()
    for layer in model.vision_model.encoder.layers[-2:]:layer.train()
    opt.zero_grad(set_to_none=True)
    picked=[pairorder[(pt+k)%len(pairorder)] for k in range(16)] if weight else [];pt+=16
    image=F.normalize(model.get_image_features(pixel_values=pixels[batch].to(device)),dim=-1)
    caption=contrastive_loss(image,texts[batch].to(device),[caption_ids[i] for i in batch],sim[batch][:,batch].to(device),model.logit_scale.detach().exp().clamp(max=100))
    if not torch.isfinite(caption):raise ValueError('Nonfinite caption loss')
    caption_value=caption.item();caption.backward();del image,caption
    pair_value=0.
    if picked:
     extra=sorted({ix[p[a]] for p in picked for a in ['left','right']});lookup={v:k for k,v in enumerate(extra)}
     image=F.normalize(model.get_image_features(pixel_values=pixels[extra].to(device)),dim=-1)
     pairloss=image_pair_loss(image[[lookup[ix[p['left']]] for p in picked]],image[[lookup[ix[p['right']]] for p in picked]],torch.tensor([p['label'] for p in picked],device=device),protocol['negative_margin'])
     if not torch.isfinite(pairloss):raise ValueError('Nonfinite pair loss')
     pair_value=pairloss.item();(weight*pairloss).backward();del image,pairloss
    norm=torch.nn.utils.clip_grad_norm_(list(trainable.values()),1.)
    if not torch.isfinite(norm):raise ValueError('Nonfinite gradient')
    opt.step();losses.append((caption_value+weight*pair_value,caption_value,pair_value));torch.mps.empty_cache()
   vv=encode(val);metrics=retrieval(vv,texts[val].numpy());score=metrics['image_to_caption']['mrr'];record={'epoch':epoch,'losses':np.mean(losses,axis=0).tolist(),'validation':metrics};records.append(record)
   if score>best['score']+1e-8:best={'score':score,'trial':trial,'epoch':epoch,'weight':weight};beststate={n:p.detach().cpu().clone() for n,p in trainable.items()};torch.save(beststate,out/'best-trainable-state.pt')
   if score>localbest+1e-8:localbest=score;stale=0
   else:stale+=1
   save(out/'progress.json',{'state':'training','trial':trial,'epoch':epoch,'best':best,'latest':record});print(json.dumps({'event':'epoch','trial':trial,'weight':weight,**record,'best':best}),flush=True)
   if stale>=protocol['patience']:break
  trials.append({'trial':trial,'weight':weight,'records':records});save(out/'validation-trials.json',trials)
 restore(beststate);trained=encode(list(range(len(rows))));model.save_pretrained(out/'checkpoint',safe_serialization=True);processor.save_pretrained(out/'checkpoint')
 from safetensors.torch import save_file
 save_file({n:p.detach().cpu().contiguous() for n,p in trainable.items()},str(out/'visual-finetune.safetensors'))
 np.savez_compressed(out/'development-features.npz',ids=np.array([r['id'] for r in rows]),baseline=baseline,trained=trained,text=texts.numpy())
 report={'implementation':'xiaoyu-chinese-clip-codex-weak-v3','selected':best,'protocol':protocol,'baseline_validation':bv,'validation_trials':trials,'seconds':round(time.time()-started,2),'checkpoint_files':{p.name:digest(p) for p in (out/'checkpoint').iterdir() if p.is_file()},'test_results':None,'limitations':['AI annotations are weak supervision, not human truth.','Source images lack personal chat context.','This is an own implementation, not author GSE.','No production model changed.'],'code_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'research/sticker_encoder/local_pairs.py',ROOT/'scripts/prepare_codex_sticker_training.py',ROOT/'scripts/mine_local_sticker_pairs.py']}}
 save(out/'training-report.json',report);save(out/'checkpoint-lock.json',{'checkpoint_files':report['checkpoint_files'],'report_sha256':digest(out/'training-report.json'),'selected':best,'test_exposure':'New test not viewed or annotated before this lock','locked_unix':time.time()});save(out/'progress.json',{'state':'selected_locked','selected':best,'seconds':report['seconds']});print(json.dumps({'event':'selected_locked','selected':best,'seconds':report['seconds']}),flush=True)
if __name__=='__main__':run()
