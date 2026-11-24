"""Train 8 broad expression pools on local weak labels; separate from synonymy."""
import copy, json, sys, time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from safetensors.torch import load_file,save_file
from research.sticker_encoder.disentangled import SemanticAdapter
from research.sticker_encoder.weak_model import digest,save
from research.sticker_encoder.broad import broad_rank
from scripts.train_sticker_v4 import align

OUT=ROOT/'outputs/sticker-broad-v5-20261119';V4=ROOT/'outputs/sticker-divergence-v4-20261119';V3=ROOT/'outputs/sticker-semantic-v3-local-20261118'


def classification_metrics(logits,labels):
    p=logits.argmax(1);y=labels.argmax(1)
    fs=[]
    for c in range(labels.shape[1]):
        tp=((p==c)&(y==c)).sum();fp=((p==c)&(y!=c)).sum();fn=((p!=c)&(y==c)).sum()
        if (y==c).any():fs.append(float(2*tp/max(1,2*tp+fp+fn)))
    return {'top1_pool_accuracy':float((p==y).mean()),'macro_f1':float(np.mean(fs)),
            'n':len(labels),'interpretation':'Accuracy against BGE-generated broad labels; not human synonymy accuracy'}


def main():
    if (OUT/'checkpoint-lock.json').exists():raise ValueError('Locked run exists')
    start=time.time();torch.set_num_threads(4);torch.manual_seed(20261119)
    rows=json.loads((V4/'dataset.json').read_text())['rows'];ids=[r['id'] for r in rows]
    cats=json.loads((OUT/'categories.json').read_text());prototypes=np.load(OUT/'prototypes.npz')['vectors']
    y=align(np.load(V4/'all-caption-features.npz'),'meaning',ids);x=align(np.load(V3/'evaluation-features.npz'),'v3_image',ids)
    adapter=SemanticAdapter();adapter.load_state_dict(load_file(str(V4/'semantic-adapter.safetensors')))
    with torch.no_grad():z=adapter(torch.from_numpy(x)).numpy()
    score=y@prototypes.T;order=np.argsort(-score,axis=1)
    # Primary broad labels do not depend on artwork, person or source pack.
    labels=np.eye(len(cats),dtype='float32')[order[:,0]]
    for i,r in enumerate(rows):
        selected=[int(order[i,0])]
        if score[i,order[i,0]]-score[i,order[i,1]]<=.035:selected.append(int(order[i,1]))
        r['intent_groups']=[cats[j]['id'] for j in selected]
        r['intent_label_source']='local_BGE_broad_prototypes_AI_weak'
    train=np.array([i for i,r in enumerate(rows) if r['split']=='train' and r['quality']!='uncertain'])
    val=np.array([i for i,r in enumerate(rows) if r['split']=='validation' and r['quality']!='uncertain'])
    test=np.array([i for i,r in enumerate(rows) if r['split']=='test'])
    xt=torch.tensor(z);target=torch.tensor(order[:,0],dtype=torch.long)
    model=torch.nn.Linear(512,len(cats));opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.02)
    counts=torch.bincount(target[train],minlength=len(cats)).clamp(min=1)
    weights=(len(train)/counts.float()).sqrt();weights/=weights.mean()
    best=-1.;state=None;records=[];stale=0;selected_epoch=None
    protocol={'reason':'User requested broader expression pools after V4; no fine-grained semantic taxonomy',
              'groups':[c['name'] for c in cats],'seed':20261119,'train_n':len(train),'validation_n':len(val),
              'test_n':len(test),'test_used_in_fit':0,'epoch_max':500,'eval_every':10,'patience_evaluations':10,
              'selection':'validation broad-pool macro-F1','input':'frozen trained V4 semantic projection',
              'trainable_parameters':512*8+8,'labels':'BGE descriptions matched to fixed eight broad prototypes; AI weak',
              'test_policy':'historical fixed holdout; already evaluated in previous versions, not blind',
              'retrieval_policy':'Broad meaning pools then semantic ranking, top20 cross-style window; no .92 image/.72 text hard cutoff',
              'semantic_rank':'.6 description cosine + .4 V4 trained image semantic cosine; heuristic fixed ranking weights',
              'same_pool_is_synonymy':False,'paid_api_calls':0}
    save(OUT/'protocol.json',protocol)
    for epoch in range(1,501):
        opt.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(model(xt[train]),target[train],weight=weights)
        loss.backward();opt.step()
        if epoch%10:continue
        with torch.no_grad():metric=classification_metrics(model(xt[val]).numpy(),labels[val])
        records.append({'epoch':epoch,'loss':loss.item(),'validation':metric})
        if metric['macro_f1']>best:
            best=metric['macro_f1'];state=copy.deepcopy(model.state_dict());selected_epoch=epoch;stale=0
        else:stale+=1
        if stale>=10:break
    model.load_state_dict(state);save_file(state,str(OUT/'broad-intent-head.safetensors'))
    save(OUT/'dataset.json',{'rows':rows,'source_sha256':digest(V4/'dataset.json'),'primary_group_counts':dict(Counter(r['intent_groups'][0] for r in rows))})
    report={'implementation':'xiaoyu-broad-intent-v5','selected_epoch':selected_epoch,'validation_macro_f1':best,
            'epochs':records,'protocol':protocol,'trainable_parameters':4104,'human_gold_evaluation':False,
            'retrieval_semantic_adapter':str(V4/'semantic-adapter.safetensors'),'paid_api_calls':0,'seconds':time.time()-start,
            'meaning':'The learned head predicts broad pools from images. Cached preview uses existing description pools, ranked by the trained V4 semantic adapter and BGE; this is not exact synonym classification.'}
    save(OUT/'training-report.json',report)
    save(OUT/'checkpoint-lock.json',{'checkpoint_sha256':digest(OUT/'broad-intent-head.safetensors'),'report_sha256':digest(OUT/'training-report.json'),'time':time.time()})
    # Test is read only after checkpoint selection and lock.
    with torch.no_grad():pred=model(xt).numpy()
    baseline=z@prototypes.T
    evaluation={'test':classification_metrics(pred[test],labels[test]),'baseline_test':classification_metrics(baseline[test],labels[test]),
                'validation':classification_metrics(pred[val],labels[val]),'test_ids':[ids[i] for i in test],
                'human_gold':False,'warning':'Broad-pool prediction is not contextual semantic equivalence.'}
    save(OUT/'evaluation-report.json',evaluation)
    for i,r in enumerate(rows):r['image_predicted_group']=cats[int(pred[i].argmax())]['id']
    save(OUT/'display-dataset.json',{'rows':rows})
    m=np.load(V4/'retrieval-matrices.npz');matrices={k:m[k] for k in ('v2','v4','meaning','style')}
    results={}
    for i,q in enumerate(rows):
        candidates=[{**r,'semantic_score':float(matrices['v4'][i,j]),'context_score':float(matrices['meaning'][i,j]),'style_distance':float(matrices['style'][i,j])} for j,r in enumerate(rows) if j!=i]
        value=broad_rank(q,candidates)
        results[q['id']]={'stage1_count':len(value['stage1']),
                          'stage2':[{k:r[k] for k in ('id','semantic_score','context_score','style_distance','recall_score','semantic_rank_before_style')} for r in value['stage2']],
                          'rejected_counts':dict(Counter(r['reason'] for r in value['rejected']))}
    save(OUT/'two-stage-results.json',{'policy':protocol['retrieval_policy'],'queries':results})
    print(json.dumps({'training':report['selected_epoch'],'evaluation':evaluation,'B0032':results['B0032'],'coverage':sum(bool(r['stage2']) for r in results.values())},ensure_ascii=False))


if __name__=='__main__':main()
