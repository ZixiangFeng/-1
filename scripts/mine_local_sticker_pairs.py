# -*- coding: utf-8 -*-
"""Offline candidate mining. Candidate labels are NOT automatically approved."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from research.sticker_encoder.weak_model import digest,save

def run():
 out=ROOT/'outputs/sticker-codex-local-20261118';data=json.loads((out/'dataset.json').read_text());rows=data['rows'];ids=[r['id'] for r in rows]
 path=out/'style-features.npz'
 if not path.exists():
  import torch
  from torchvision import transforms
  from PIL import Image
  from vendor.aladin.vit_networks import VisionTransformer
  from style_lab import PROVENANCE
  torch.set_num_threads(4)
  ckpt=ROOT/'models/style-lab/aladinVIT_bamfg_64.47.pt'
  if digest(ckpt)!=PROVENANCE['style']['weights_sha256'] or digest(ROOT/'vendor/aladin/vit_networks.py')!=PROVENANCE['style']['code_sha256']:raise ValueError('Author model checksum mismatch')
  class AViT(torch.nn.Module):
   def __init__(self):super().__init__();self.enc_style=VisionTransformer()
   def forward(self,x):return self.enc_style(x,just_style_code=True)[0]
  model=AViT();model.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True),strict=True);model.eval().to('mps')
  transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((.5,.5,.5),(.5,.5,.5))])
  old=np.load(ROOT/'outputs/sticker-benchmark-500/features.npz');oldmap={str(k):i for i,k in enumerate(old['ids'])};vectors={k:old['style'][i] for k,i in oldmap.items()}
  pending=[r for r in rows if r['id'] not in vectors]
  with torch.inference_mode():
   for start in range(0,len(pending),16):
    group=pending[start:start+16];batch=[]
    for r in group:
     with Image.open(out/r['image']) as im:batch.append(transform(im.convert('RGB')))
    v=model(torch.stack(batch).to('mps'));v=torch.nn.functional.normalize(v,dim=-1).cpu().numpy()
    if not np.isfinite(v).all():raise ValueError('Invalid style feature')
    vectors.update({r['id']:a for r,a in zip(group,v)})
    if start%160==0:print(json.dumps({'style_done':start+len(group),'total':len(pending)}),flush=True)
  np.savez_compressed(path,ids=np.array(ids),style=np.array([vectors[k] for k in ids]))
 style=np.load(path);teacher=np.load(out/'teacher-features.npz')
 if list(style['ids'])!=ids or list(teacher['ids'])!=ids:raise ValueError('ID alignment')
 ss=1-style['style']@style['style'].T; sem=teacher['meaning']@teacher['meaning'].T
 train=[i for i,r in enumerate(rows) if r['split']=='train' and r['id'].startswith('E') and r['quality']=='clear']
 values=ss[np.ix_(train,train)][np.triu_indices(len(train),1)];low,high=np.quantile(values,[.25,.5])
 candidates=[]
 for i in train:
  for j in train:
   if j<=i:continue
   if sem[i,j]>=.72 and ss[i,j]>=high and rows[i]['pack']!=rows[j]['pack']:kind='positive'
   elif sem[i,j]<=.45 and ss[i,j]<=low:kind='negative'
   else:continue
   candidates.append({'left':ids[i],'right':ids[j],'kind':kind,'semantic_cosine':float(sem[i,j]),'style_distance':float(ss[i,j]),'left_meaning':rows[i]['text'],'right_meaning':rows[j]['text'],'human_reviewed':False,'approved':False})
 chosen=[]
 for kind in ['positive','negative']:
  subset=[p for p in candidates if p['kind']==kind]
  subset.sort(key=lambda p:(-p['semantic_cosine'] if kind=='positive' else p['style_distance']))
  counts={}
  for p in subset:
   if any(counts.get(a,0)>=2 for a in [p['left'],p['right']]):continue
   p['pair_id']=f'{kind[0].upper()}{1+sum(x["kind"]==kind for x in chosen):03}'
   chosen.append(p)
   for a in [p['left'],p['right']]:counts[a]=counts.get(a,0)+1
   if sum(x['kind']==kind for x in chosen)>=50:break
 save(out/'pair-candidates.json',{'thresholds':{'meaning_positive_min':.72,'meaning_negative_max':.45,'style_low_train_quantile25':float(low),'style_high_train_median':float(high)},'split':'train_only','pairs':chosen,'policy':'Codex AI must review proposed pair relations before use; no human gold claims.'})
 for p in chosen:print(p['pair_id'],p['left'],p['right'],round(p['style_distance'],3),p['left_meaning'],' / ',p['right_meaning'])
if __name__=='__main__':run()
