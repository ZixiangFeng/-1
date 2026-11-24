"""Append held-out ALADIN vectors after semantic model lock; no training."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from PIL import Image
from torchvision import transforms
from vendor.aladin.vit_networks import VisionTransformer
from research.sticker_encoder.weak_model import digest
from style_lab import PROVENANCE
source=ROOT/'outputs/sticker-codex-local-20261118';out=ROOT/'outputs/sticker-semantic-v3-local-20261118'
if not (out/'checkpoint-lock.json').exists():raise ValueError('Model not locked')
rows=json.loads((out/'full-dataset.json').read_text())['rows'];old=np.load(source/'style-features.npz');vectors={str(k):v for k,v in zip(old['ids'],old['style'])};pending=[r for r in rows if r['id'] not in vectors]
ckpt=ROOT/'models/style-lab/aladinVIT_bamfg_64.47.pt'
if digest(ckpt)!=PROVENANCE['style']['weights_sha256'] or digest(ROOT/'vendor/aladin/vit_networks.py')!=PROVENANCE['style']['code_sha256']:raise ValueError('Author checkpoint/source mismatch')
class AViT(torch.nn.Module):
 def __init__(self):super().__init__();self.enc_style=VisionTransformer()
 def forward(self,x):return self.enc_style(x,just_style_code=True)[0]
torch.set_num_threads(4);model=AViT();model.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True));model.eval().to('mps');tr=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((.5,.5,.5),(.5,.5,.5))])
with torch.inference_mode():
 for i in range(0,len(pending),16):
  group=pending[i:i+16];batch=[]
  for r in group:
   with Image.open(source/r['image']) as im:batch.append(tr(im.convert('RGB')))
  v=torch.nn.functional.normalize(model(torch.stack(batch).to('mps')),dim=-1).cpu().numpy()
  vectors.update({r['id']:value for r,value in zip(group,v)})
np.savez_compressed(out/'all-style-features.npz',ids=np.array([r['id'] for r in rows]),style=np.array([vectors[r['id']] for r in rows]))
print(json.dumps({'style_count':len(rows),'newly_encoded':len(pending),'author_weight_sha256':digest(ckpt)}))
