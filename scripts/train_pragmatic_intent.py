"""Fit eight broad language families. Select thresholds on development data only."""
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sticker_pragmatics as use
import sticker_ranker as ranker
from scripts.train_context_ranker import CORPUS, DATA, features, judgments, read, write

OUT = ROOT/'outputs/sticker-pragmatic-v7'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def examples(path):
    with path.open() as f:
        for row in csv.DictReader(f, delimiter='\t'):
            key = row['group']
            group = use.OUTSIDE if key == use.OUTSIDE else use.family(key)
            if not group:
                raise ValueError(key)
            for text in row['utterances'].split('|'):
                yield text, group


def performance(targets, predictions, labels):
    pred = np.asarray([g or use.OUTSIDE for g in predictions])
    truth = np.asarray(targets)
    f1 = []
    for g in labels:
        tp = int(((pred==g) & (truth==g)).sum())
        denom = int((pred==g).sum() + (truth==g).sum())
        f1.append(2*tp/max(1,denom))
    return dict(macro_f1=float(np.mean(f1)), correct=int((truth==pred).sum()),
                n=len(truth), returned=int((pred!=use.OUTSIDE).sum()))


def fit():
    OUT.mkdir(parents=True,exist_ok=True)
    path = use.DIRECTORY/'language-intent.npz'
    if (OUT/'training.json').exists() or (OUT/'blind-key.json').exists():
        raise RuntimeError('These weights have already been locked.')
    corpus = read(CORPUS)['rows']
    by_id = {r['id']:r for r in corpus}
    training = defaultdict(set)
    dev = list(examples(use.DIRECTORY/'development-dialogues.tsv'))
    excluded = {use.wording(t) for t,_ in dev}

    def add(text, group):
        if use.wording(text) not in excluded:
            training[text].add(group)

    for t,g in examples(use.DIRECTORY/'training-dialogues.tsv'):
        add(t,g)
    for g, phrases, _ in read(DATA/'dialogues.json'):
        for t in phrases:
            add(t,use.family(g))
    image_ids = set()
    for g, ids in read(DATA/'reviewed-training.json').items():
        for identifier in ids:
            r = by_id[identifier]
            if r['split'] != 'train' or r['quality']=='uncertain':
                raise ValueError('Unexpected non-training image')
            if use.wording(r['meaning']) not in excluded:
                add(r['meaning'], use.family(g))
                image_ids.add(identifier)
    texts = sorted(training)
    targets = [g for _,g in dev]
    all_text = texts + [t for t,_ in dev]
    proj = ranker.metric()
    vectors = proj.project(ranker.local_encoder().encode(all_text))
    stripped = [use.meaning_part(use.wording(t)) for t in all_text]
    n = len(texts)
    vocab, idf = use.make_vocab(stripped[:n])
    lex = use.lexical_features(stripped,{g:i for i,g in enumerate(vocab)},idf)
    kernel = use.language_kernel(vectors,lex,vectors[:n],lex[:n])
    labels = sorted(use.NAMES)+[use.OUTSIDE]
    y = np.zeros((n,len(labels)), dtype=np.float64)
    for i,t in enumerate(texts):
        for g in training[t]:
            y[i,labels.index(g)] = 1/len(training[t])
    best = None
    for regularizer in (.08,.25,.65):
        system = kernel[:n].astype('float64') + regularizer*np.eye(n)
        alpha = np.linalg.solve(system,y).astype(np.float32)
        val = kernel[n:] @ alpha
        for minimum in (.18,.28,.38,.48):
            for margin in (.03,.09,.15):
                guesses = use.classifier_decisions(all_text[n:],val,labels,minimum,margin)
                p = performance(targets,guesses,labels)
                key = (p['macro_f1'],p['correct'],minimum,margin)
                if best is None or key>best[0]:
                    best = (key,p,alpha,regularizer,minimum,margin)
    _, report, alpha, regularizer, minimum, margin = best
    # Score preexisting held-out caption-pair judgments before opening the new images.
    vec = proj.project(features(corpus))
    indices = {r['id']:i for i,r in enumerate(corpus)}
    groups = judgments('validation',indices)
    score = np.concatenate([vec[c] @ vec[q] for q,c,_ in groups])
    truth = np.concatenate([t for _,_,t in groups])
    trials = []
    for floor in (.38,.44,.50,.56,.60,.64,.68):
        accepted = score >= floor
        tp,fp,fn = int((truth & accepted).sum()),int((~truth & accepted).sum()),int((truth & ~accepted).sum())
        f = 1.5*tp/max(1,1.5*tp+fp+.5*fn)
        trials.append((f,floor,tp,fp,fn))
    _, pair_floor,tp,fp,fn=max(trials)
    np.savez_compressed(path,version=use.VERSION,base=proj.digest,labels=labels,anchor_z=vectors[:n],
        vocab=vocab,idf=idf,texts=stripped[:n],alpha=alpha,minimum=minimum,margin=margin,pair_floor=pair_floor)
    record=dict(implementation=use.VERSION,sha256=digest(path),base=proj.digest,n_families=len(use.NAMES),
        training_distinct_texts=n,training_image_ids=len(image_ids),class_counts=dict(Counter(g for s in training.values() for g in s)),
        development=report,regularization=regularizer,minimum=minimum,margin=margin,pair_floor=pair_floor,
        pair_calibration=dict(n=len(score),matches=tp,false_matches=fp,missed=fn),
        selection='Macro F1 including outside on development utterances; independent pair cutoff uses F-beta (beta squared = 0.5) on previous validation pairs.',
        inputs={p.name:digest(p) for p in [use.DIRECTORY/'training-dialogues.tsv',use.DIRECTORY/'development-dialogues.tsv',
                      DATA/'dialogues.json', DATA/'reviewed-training.json',DATA/'validation-judgments.json',ROOT/'sticker_pragmatics.py']},
        frozen_external_sample=digest(OUT/'holdout/manifest.json'))
    write(OUT/'training.json',record)
    print(json.dumps({k:record[k] for k in ('n_families','training_distinct_texts','training_image_ids','development','pair_floor')},ensure_ascii=False))


if __name__=='__main__':
    fit()
