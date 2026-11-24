"""Transparent GSE-inspired reproduction helpers, not the author's implementation."""
import hashlib
import json
import math
from pathlib import Path

IMPLEMENTATION = 'xiaoyu-clip-sticker-reproduction-v1'
PAPER = 'https://arxiv.org/html/2511.04977v1'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def metrics(scores, labels, threshold):
    if not scores or len(scores) != len(labels) or set(labels) != {0,1}:
        raise ValueError('Evaluation needs both positive and negative human-labeled pairs')
    if not math.isfinite(threshold) or any(not math.isfinite(s) or not -1.00001 <= s <= 1.00001 for s in scores):
        raise ValueError('Invalid cosine scores or threshold')
    tp=sum(s>=threshold and y==1 for s,y in zip(scores,labels))
    fp=sum(s>=threshold and y==0 for s,y in zip(scores,labels))
    fn=sum(s<threshold and y==1 for s,y in zip(scores,labels))
    tn=len(labels)-tp-fp-fn
    precision=tp/(tp+fp) if tp+fp else 0
    recall=tp/(tp+fn) if tp+fn else 0
    ordered=sorted(zip(scores,labels));neg=0;wins=0.;i=0;positive_count=sum(labels)
    while i<len(ordered):
        j=i
        while j<len(ordered) and ordered[j][0]==ordered[i][0]:j+=1
        pos=sum(y for _,y in ordered[i:j]);n=j-i-pos
        wins+=pos*(neg+.5*n);neg+=n;i=j
    auc=wins/(positive_count*(len(labels)-positive_count))
    return {'n':len(labels),'accuracy':(tp+tn)/len(labels),'precision':precision,'recall':recall,
            'f1':2*precision*recall/(precision+recall) if precision+recall else 0,
            'roc_auc':auc,'tp':tp,'fp':fp,'fn':fn,'tn':tn,'threshold':threshold}

def calibrate(scores,labels):
    # This function must only receive validation data. A stricter tied threshold wins.
    metrics(scores,labels,0)  # Validate scores and both classes.
    ordered=sorted(zip(scores,labels),reverse=True);tp=fp=0;positive_count=sum(labels);i=0;best=None
    while i<len(ordered):
        threshold=ordered[i][0];j=i
        while j<len(ordered) and ordered[j][0]==threshold:
            tp+=ordered[j][1];fp+=1-ordered[j][1];j+=1
        precision=tp/(tp+fp);recall=tp/positive_count
        f1=2*precision*recall/(precision+recall) if precision+recall else 0
        candidate=(f1,precision,threshold)
        if best is None or candidate>best:best=candidate
        i=j
    return metrics(scores,labels,best[2])

def validate_dataset(root):
    root=Path(root)
    train=read_jsonl(root/'train.jsonl')
    val=read_jsonl(root/'validation.jsonl')
    test=read_jsonl(root/'test.jsonl')
    if len(train)<2 or len(val)<2 or len(test)<2: raise ValueError('Training and independent validation/test annotations are required')
    def resolve(name):
        p=(root/name).resolve()
        if not p.is_relative_to(root.resolve()) or not p.is_file(): raise ValueError('Image path outside dataset or missing')
        return p
    hashes={}
    for split,rows in [('train',train),('validation',val),('test',test)]:
        seen=set();pairs=set()
        for row in rows:
            if row.get('reviewed') is not True: raise ValueError('Every training caption and evaluation label must be human reviewed')
            names=[row['image']] if split=='train' else [row['left'],row['right']]
            if split=='train' and not row.get('text','').strip(): raise ValueError('Empty training caption')
            if split!='train' and (type(row.get('label')) is not int or row['label'] not in (0,1)): raise ValueError('Pair label must be 0 or 1')
            digests=[sha(resolve(name)) for name in names]
            if split!='train':
                pair=tuple(sorted(digests))
                if pair[0]==pair[1] or pair in pairs: raise ValueError('Self pairs and duplicate/reversed pairs are not independent observations')
                pairs.add(pair)
            seen.update(digests)
        hashes[split]=seen
        if split!='train' and {r['label'] for r in rows}!={0,1}: raise ValueError('Each evaluation split needs both labels')
    if any(hashes[a]&hashes[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('Image leakage between splits: assign by image/family before making pairs')
    return train,val,test,{'files':{name:sha(root/name) for name in ('train.jsonl','validation.jsonl','test.jsonl')},
                          'image_hashes':{s:sorted(h) for s,h in hashes.items()},
                          'note':'Exact-file disjointness checked; researcher must also keep near-duplicates and character families together.'}

def paper_pair_candidates(rounds, query_sets, text_cosines):
    """Paper §3.2 candidate construction; final manual verification still required.
    selected, not offered candidates, form the semantically accepted set.
    """
    from collections import defaultdict
    from itertools import combinations
    offered=defaultdict(set);accepted=defaultdict(set)
    judgments=set()
    for r in rounds:
        judgment=(str(r['annotator']),str(r['round']),str(r['anchor']))
        if judgment in judgments: raise ValueError('Duplicate annotation judgment')
        judgments.add(judgment)
        shown=set(r['candidates'])|{r['anchor']}
        selected=set(r['selected'])|{r['anchor']}
        if not selected<=shown: raise ValueError('Selected item was not shown')
        for pair in combinations(sorted(shown),2): offered[pair].add(judgment)
        for pair in combinations(sorted(selected),2): accepted[pair].add(judgment)
    result=[]
    for pair,seen in offered.items():
        pos=len(accepted[pair]);label=None
        if pos>=2: label=1
        elif len(seen)>=2 and pos==0:
            a,b=pair;score=text_cosines.get('|'.join(pair))
            if a in query_sets and b in query_sets and score is not None and math.isfinite(score):
                if len(set(query_sets[a])&set(query_sets[b]))<3 and score<.7: label=0
        if label is not None: result.append({'left':pair[0],'right':pair[1],'label':label,'reviewed':False,
                                             'offered_count':len(seen),'accepted_count':pos})
    return result
