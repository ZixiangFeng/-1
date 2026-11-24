"""Explicit AI-reviewed auxiliary image pairs. Never human gold labels."""
import torch
import torch.nn.functional as F

def validate_pairs(pairs, rows):
    by_id={r['id']:r for r in rows}
    seen=set()
    for p in pairs:
        if p.get('reviewer')!='codex_ai' or p.get('human_reviewed') is not False: raise ValueError('Incorrect label provenance')
        if p['label'] not in (0,1) or p['left']==p['right']: raise ValueError('Invalid pair')
        for aid in (p['left'],p['right']):
            if aid not in by_id or by_id[aid]['split']!='train' or by_id[aid]['quality']!='clear':raise ValueError('Pair crosses training/quality boundary')
        key=tuple(sorted((p['left'],p['right'])))
        if key in seen:raise ValueError('Repeated pair')
        seen.add(key)

def image_pair_loss(left,right,labels,negative_margin=.2):
    """Align explicit positives; penalize negatives above cosine margin."""
    cosine=(F.normalize(left,dim=-1)*F.normalize(right,dim=-1)).sum(-1)
    labels=labels.to(cosine.device)
    positive=labels==1;negative=labels==0;parts=[]
    if positive.any():parts.append((1-cosine[positive]).mean())
    if negative.any():parts.append(F.relu(cosine[negative]-negative_margin).mean())
    if not parts:raise ValueError('Empty pair batch')
    return torch.stack(parts).mean()
