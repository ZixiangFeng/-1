"""Train/evaluate an explicitly named GSE-inspired image-text CLIP reproduction.

Requires reviewed train.jsonl, validation.jsonl, test.jsonl and a local pretrained
HF CLIP directory. Nothing is downloaded, bought, or fitted to participant data.
"""
import argparse
import json
import os
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from semantic_training import IMPLEMENTATION,PAPER,validate_dataset,calibrate,metrics,sha


def train(args):
    train_rows,val_rows,test_rows,data_manifest=validate_dataset(args.dataset)
    import torch
    from PIL import Image
    from transformers import CLIPModel,CLIPProcessor
    root=Path(args.dataset).resolve();base=Path(args.base).resolve();out=Path(args.output).resolve()
    if not base.is_dir(): raise ValueError('Provide a local pretrained CLIP directory')
    if out.exists(): raise ValueError('Use a new output directory to preserve previous runs')
    if args.epochs<1 or args.batch_size<2 or args.lr<=0: raise ValueError('Invalid training configuration')
    torch.manual_seed(args.seed);random.seed(args.seed)
    torch.set_num_threads(1)
    device=torch.device(args.device)
    model=CLIPModel.from_pretrained(base,local_files_only=True,use_safetensors=True).to(device)
    processor=CLIPProcessor.from_pretrained(base,local_files_only=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    def rgb(name):
        with Image.open(root/name) as image:
            rgba=image.convert('RGBA');bg=Image.new('RGBA',rgba.size,'white');bg.alpha_composite(rgba)
            return bg.convert('RGB')
    @torch.inference_mode()
    def pair_scores(rows):
        model.eval();vectors={}
        for name in sorted({r[k] for r in rows for k in ('left','right')}):
            inputs=processor(images=rgb(name),return_tensors='pt').to(device)
            v=model.get_image_features(**inputs);v=torch.nn.functional.normalize(v,dim=-1)
            vectors[name]=v[0].cpu()
        return [float(torch.dot(vectors[r['left']],vectors[r['right']]).clamp(-1,1)) for r in rows]
    out.mkdir(parents=True)
    losses=[]
    for epoch in range(args.epochs):
        model.train();indices=list(range(len(train_rows)));random.shuffle(indices);total=0;steps=0
        for start in range(0,len(indices),args.batch_size):
            batch=[train_rows[i] for i in indices[start:start+args.batch_size]]
            if len(batch)<2: continue
            inputs=processor(text=[r['text'] for r in batch],images=[rgb(r['image']) for r in batch],
                             return_tensors='pt',padding=True,truncation=True).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=model(**inputs,return_loss=True).loss
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            with torch.no_grad(): model.logit_scale.clamp_(0,4.6052)
            total+=loss.item();steps+=1
        if not steps: raise ValueError('No valid training batches')
        losses.append(total/steps);print(json.dumps({'epoch':epoch+1,'loss':losses[-1]}),flush=True)
    # Fit threshold ONLY on validation pairs; freeze it before looking at test labels.
    validation=calibrate(pair_scores(val_rows),[r['label'] for r in val_rows])
    test=metrics(pair_scores(test_rows),[r['label'] for r in test_rows],validation['threshold'])
    model.save_pretrained(out,safe_serialization=True);processor.save_pretrained(out)
    manifest={'implementation':IMPLEMENTATION,'author_original':False,'paper':PAPER,'state':'trained_evaluated',
              'objective':'symmetric image-text CLIP contrastive loss; paper section 4.1 interpretation',
              'base':str(base),'base_files':{str(p.relative_to(base)):sha(p) for p in base.rglob('*') if p.is_file()},
              'base_config':model.config.to_dict(),'epochs':args.epochs,'batch_size':args.batch_size,
              'lr':args.lr,'optimizer':'AdamW; weight_decay=0.01; grad_clip=1; fixed LR (our choices)',
              'seed':args.seed,'losses':losses,'data':data_manifest,'validation':validation,'test':test,
              'limitations':['Not original GSE weights or exact reproduction; original training data unavailable.',
                              'Paper does not identify CLIP variant, detailed captions, preprocessing or optimizer.',
                              'Image-only general semantics; cannot infer a particular user’s contextual meaning.',
                              'Similar/duplicate captions in a batch may be false negatives under standard CLIP loss.'],
              'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
    (out/'reproduction.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps({'output':str(out),'validation':validation,'test':test,'author_original':False}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--base',required=True);p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=5);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cpu')
    train(p.parse_args())
