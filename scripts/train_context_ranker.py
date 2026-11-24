"""Fit a meaning-only metric; keep the historical image partitions fixed."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import sticker_ranker as ranker
from scripts.review_sticker_ranker import CORPUS, DATA, MATRICES, review_set

OUT = ROOT / 'outputs/sticker-context-v6'
CAPTIONS = ROOT / 'outputs/sticker-divergence-v4-20261119/all-caption-features.npz'
PAIR_REVIEW = ROOT / 'outputs/sticker-codex-local-20261118/pair-review.json'
SEED = 261119


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,allow_nan=False))


def judgments(split, index):
    labels = read(DATA / (split + '-judgments.json'))
    groups = review_set(split)
    if labels.keys() != {g['query'] for g in groups}:
        raise ValueError('Incomplete split judgments')
    validated = []
    for g in groups:
        q = g['query']
        choices = g['choices']
        if not set(labels[q]).issubset(choices):
            raise ValueError('Unknown judged pair: ' + q)
        validated.append((index[q],np.array([index[i] for i in choices]),
                          np.array([i in labels[q] for i in choices], dtype=bool)))
    return validated


def score_pools(groups, scores, floor=-2):
    dcg, pairs, top1, hits = [], [], [], []
    tp = fp = fn = returned = 0
    for q, candidates, good in groups:
        s = scores[q,candidates]
        ranking = np.argsort(-s,kind='stable')
        ranking = ranking[s[ranking] >= floor]
        tp += int(good[ranking[:5]].sum())
        fp += len(ranking[:5]) - int(good[ranking[:5]].sum())
        fn += int(good.sum()) - int(good[ranking[:5]].sum())
        returned += bool(len(ranking))
        if not good.any():
            continue
        result = good[ranking[:5]]
        discounts = 1 / np.log2(np.arange(2,7))
        dcg.append(float(result @ discounts[:len(result)] / discounts[:min(5,good.sum())].sum()))
        top1.append(bool(len(result) and result[0]))
        hits.append(bool(result.any()))
        if (~good).any():
            pairs.append(float((s[good,None] > s[None,~good]).mean()))
    return {'queries':len(groups),'judged_pairs':sum(len(r[1]) for r in groups),
            'answerable':len(dcg),'positives':tp+fn,'ndcg5':float(np.mean(dcg)),
            'hit1':float(np.mean(top1)),'hit5':float(np.mean(hits)),
            'pair_concordance':float(np.mean(pairs)), 'returned_queries':returned,
            'returned5_precision':tp/max(1,tp+fp),'returned5_recall':tp/max(1,tp+fn)}


def training_pairs(rows, index, groups, reviewed, style_distance, gen):
    sets = {}
    pos = set()
    neg = set()
    compatible = {}
    next_i = len(rows)
    synthetic = set()
    for key, texts, opposites in groups:
        members = set(range(next_i,next_i+len(texts)))
        synthetic.update(members)
        next_i += len(texts)
        ids = [index[k] for k in reviewed[key]]
        assert all(rows[i]['split'] == 'train' and rows[i]['quality'] != 'uncertain' for i in ids)
        members.update(ids)
        sets[key] = sorted(members)
        compatible[key] = opposites
        pos.update(tuple(sorted((a,b))) for a in members for b in members if a != b)
    for name, unsafe in compatible.items():
        for other in unsafe:
            neg.update(tuple(sorted((a,b))) for a in sets[name] for b in sets[other] if a!=b)
    for r in read(PAIR_REVIEW)['accepted']:
        a,b = index[r['left']],index[r['right']]
        if rows[a]['split'] == rows[b]['split'] == 'train':
            (pos if r['label'] else neg).add(tuple(sorted((a,b))))
    neg -= pos  # Multi-use images can express more than one thing.
    neg_neighbors = {}
    for a,b in sorted(neg):
        neg_neighbors.setdefault(a,[]).append(b)
        neg_neighbors.setdefault(b,[]).append(a)
    triplets = []
    cross_positive, similar_negative = 0,0
    for pair in sorted(pos):
        for a,b in [pair, pair[::-1]]:
            pool = neg_neighbors.get(a,[])
            if not pool:
                continue
            cross = a not in synthetic and b not in synthetic and rows[a]['pack']!=rows[b]['pack'] and style_distance[a,b]>=.65
            close = [n for n in pool if a not in synthetic and n not in synthetic and style_distance[a,n] <= .55]
            for repeat in range(3 if cross else 1):
                use_close = bool(close) and repeat%2 == 0
                n = int(gen.choice(close if use_close else pool))
                triplets.append((a,b,n))
                cross_positive += int(cross)
                similar_negative += use_close
    return np.asarray(triplets), {'unique_positive_pairs':len(pos),'unique_negative_pairs':len(neg),
        'triplets':len(triplets),'cross_style_image_positive_triplets':cross_positive,
        'similar_style_image_negative_triplets':similar_negative,
        'reviewed_train_images':len({i for ids in reviewed.values() for i in ids}),
        'synthetic_training_utterances':len(synthetic)}


def features(rows):
    with np.load(CAPTIONS) as data:
        assert list(data['ids']) == [r['id'] for r in rows]
        return ranker.unit(data['meaning'])


def fit(phrase_path):
    import torch
    import torch.nn.functional as F
    if (OUT/'checkpoint-lock.json').exists():
        raise FileExistsError('This fit already has a locked checkpoint')
    OUT.mkdir(exist_ok=True)
    gen = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(3)
    rows = read(CORPUS)['rows']
    ix = {r['id']:i for i,r in enumerate(rows)}
    train = np.array([i for i,r in enumerate(rows) if r['split']=='train' and r['quality']!='uncertain'])
    val = judgments('validation',ix)
    x = features(rows)
    scenes = read(DATA/'dialogues.json')
    phrases = [p for _,texts,_ in scenes for p in texts]
    with np.load(phrase_path) as f:
        assert list(f['texts']) == phrases
        joint = torch.tensor(np.concatenate([x,ranker.unit(f['vectors'])]))
    with np.load(MATRICES) as m:
        assert list(m['ids']) == list(ix)
        triples, counts = training_pairs(rows, ix, scenes, read(DATA/'reviewed-training.json'),m['style'],gen)
    validation_indices = np.unique([i for q,cs,_ in val for i in [q,*cs]])
    center = joint[train].mean(dim=0)
    centered = joint-center
    layer = torch.nn.Linear(512,64,bias=False)
    with torch.no_grad():
        _,_,axes = torch.linalg.svd(centered[train], full_matrices=False)
        layer.weight.copy_(axes[:64])
    opt = torch.optim.AdamW(layer.parameters(),lr=.002,weight_decay=.02)
    model_path = OUT/'semantic-ranker-v6.npz'
    history = []
    best = (-1.,-1.)
    chosen = None
    # Validation and test never contribute gradient updates or centering.
    for epoch in range(1,49):
        costs = []
        for subset in np.array_split(gen.permutation(len(triples)),max(1,len(triples)//256)):
            t = torch.as_tensor(triples[subset])
            a,b,n = F.normalize(layer(centered[t]),dim=-1).unbind(1)
            pos = (a*b).sum(-1)
            neg = (a*n).sum(-1)
            ranking = F.softplus((neg-pos+.20)/.12).mean()
            # Anchor unlabelled training examples to the frozen teacher's geometry.
            ids = gen.choice(train,size=80,replace=False)
            teacher = F.normalize(centered[ids],dim=-1)
            pred = F.normalize(layer(centered[ids]),dim=-1)
            geometry = F.mse_loss(pred @ pred.T,teacher @ teacher.T)
            loss = ranking + .4*geometry
            opt.zero_grad()
            loss.backward()
            opt.step()
            costs.append(float(loss.detach()))
        if epoch % 2:
            continue
        with torch.no_grad():
            z = np.zeros((len(rows),64), dtype=np.float32)
            z[validation_indices] = F.normalize(layer(centered[validation_indices]),dim=-1).numpy()
        # Explicitly compute only validation comparisons before locking the fit.
        base = np.zeros((len(rows),len(rows)),dtype=np.float32)
        learned = np.zeros_like(base)
        for q,c,_ in val:
            base[q,c] = x[c] @ x[q]
            learned[q,c] = z[c] @ z[q]
        for mix in [.25,.5,.75]:
            scores = (1-mix)*base+mix*learned
            metric = score_pools(val,scores)
            objective = (metric['ndcg5'],metric['pair_concordance'])
            history.append({'epoch':epoch,'mix':mix,'loss':float(np.mean(costs)),
                            'validation_ndcg5':objective[0],'validation_pair_concordance':objective[1]})
            if objective > best:
                best = objective
                chosen = (epoch,mix,layer.weight.detach().T.numpy().copy(),scores.copy())
    epoch,mix,projection,scores = chosen
    # Emphasize recall when calibrating a first-pass cutoff. Context and form
    # are checked downstream. Candidate count is never a training target.
    candidates = []
    for floor in np.arange(.10,.86,.025):
        metrics = score_pools(val,scores,floor)
        precision,recall = metrics['returned5_precision'],metrics['returned5_recall']
        f2 = 5*precision*recall/max(1e-9,4*precision+recall)
        candidates.append((f2,recall,-float(floor)))
    floor = -max(candidates)[2]
    np.savez_compressed(model_path,version=ranker.VERSION,center=center.numpy(),projection=projection,
                        mix=np.float32(mix),floor=np.float32(floor))
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    notes = {'implementation':ranker.VERSION,'seed':SEED,'ranker_parameters':int(projection.size),
        'frozen_encoder':'BAAI/bge-small-zh-v1.5 / document embeddings',
        'objective':'supervised contrastive triplets plus frozen-teacher geometry on training descriptions',
        'split_counts':{s:sum(r['split']==s for r in rows) for s in ['train','validation','test']},
        'eligible_train_images':len(train),'train_pairs':counts,'selected_epoch':epoch,'epochs_run':48,
        'learned_metric_weight':mix,'score_cutoff':floor,'validation':score_pools(val,scores,floor),
        'checkpoint_sha256':digest,'label_source':'Codex judged public caption compatibility; synthetic paraphrases; no human gold',
        'historical_holdout':True,'validation_selection':'nDCG@5 followed by pair concordance; cutoff by F2',
        'paid_api_calls':0,'participant_data_used':False}
    write(OUT/'development.json',history)
    write(OUT/'training-report.json',notes)
    write(OUT/'checkpoint-lock.json',{'sha256':digest,'report_sha256':hashlib.sha256((OUT/'training-report.json').read_bytes()).hexdigest()})
    print(json.dumps(notes,ensure_ascii=False))


if __name__ == '__main__':
    if sys.argv[1:] == ['--cache-phrases']:
        OUT.mkdir(exist_ok=True)
        texts = [p for _,ts,_ in read(DATA/'dialogues.json') for p in ts]
        vectors = ranker.local_encoder().encode(texts)
        np.savez_compressed(OUT/'training-phrases.npz',texts=texts,vectors=vectors)
    else:
        fit(OUT/'training-phrases.npz')
