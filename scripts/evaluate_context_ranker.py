"""Evaluate a locked checkpoint and export the scores actually used for ranking."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import sticker_ranker
from research.sticker_encoder.reviewed_gallery import meaning_vectors, reviewed_rows
from scripts.train_context_ranker import OUT, CORPUS, MATRICES, features, judgments, score_pools, read, write

OLD = ROOT/'outputs/sticker-broad-v5-20261119'
EXAMPLES = {
    'laugh_together':'朋友讲了一个笑话，我真心觉得太好笑了，和他一起大笑。',
    'awkward':'听到这句话我有点尴尬，只好跟着陪笑，其实挺不好意思的。',
    'gloat':'看到对方出了洋相，忍不住幸灾乐祸，笑话他。'
}


def export_reviewed_rankings():
    """Rebuild both directions of every similarity, retaining the historical evaluation."""
    import sticker_intent
    import sticker_pragmatics as use
    from research.sticker_encoder.disentangled import identity_reason
    model = sticker_ranker.metric()
    if model.digest != read(OUT/'checkpoint-lock.json')['sha256']:
        raise ValueError('Unexpected deployed semantic weights')
    rows = reviewed_rows(read(CORPUS)['rows'])
    z = model.project(meaning_vectors(rows, features(rows)))
    semantic = z @ z.T
    intent_model = use.model()
    inferred = intent_model.groups(z, [row['meaning'] for row in rows])
    labels = [(use.family(row.get('reviewed_intent')) or guess) if row['quality']!='uncertain' else ''
              for row,guess in zip(rows,inferred)]
    previous_labels = [(row.get('reviewed_intent') or guess) if row['quality']!='uncertain' else ''
                       for row,guess in zip(rows, sticker_intent.classify(z))]
    with np.load(MATRICES) as data:
        if list(data['ids']) != [row['id'] for row in rows]:
            raise ValueError('Style matrix order differs from catalog')
        style = data['style']
    with np.load(OUT/'demo-contexts.npz') as data:
        contexts = dict(zip(data['keys'], model.project(data['vectors'])))
    target = OUT/'web/rows'
    target.mkdir(exist_ok=True, parents=True)
    previously_assigned = sum(bool(g) for g in previous_labels)
    coverage = {'size':len(rows),'previous_assigned':previously_assigned,
                'assigned':sum(bool(g) for g in labels),'previous_with_winner':0,'with_winner':0,
                'family_counts':{g:labels.count(g) for g in use.NAMES}}
    membership = {}
    for i, row in enumerate(rows):
        def select(group, text, scores):
            return use.gallery_result(row,rows,labels,group,text,scores,style[i],identity_reason,intent_model.pair_floor)
        result = select(labels[i],row['meaning'],semantic[i])
        before = sticker_intent.gallery_result(row,rows,previous_labels,previous_labels[i],semantic[i],style[i],identity_reason)
        coverage['previous_with_winner'] += int(bool(before['maximum']))
        coverage['with_winner'] += int(bool(result['maximum']))
        membership[row['id']] = dict(family=labels[i],has_winner=bool(result['maximum']))
        # Full-gallery access for a researcher's explicit, per-context same-use set.
        # Numeric entries retain the actual stored precision for the maximum.
        result['style_lookup'] = [
            identity_reason(row,other) or sticker_intent.distance_gate(style[i,j]) or float(style[i,j])
            for j,other in enumerate(rows)]
        if row['id'] == 'B0032':
            result['contexts'] = {name:select(intent_model.groups(vec[None,:],[EXAMPLES[name]])[0],EXAMPLES[name],z @ vec)
                                  for name,vec in contexts.items()}
        result.update(query=row['id'], checkpoint_sha256=model.digest, intent_sha256=intent_model.digest)
        (target/(row['id']+'.json')).write_text(json.dumps(result, ensure_ascii=False,
                                                           separators=(',',':'), allow_nan=False))
    write(OUT.parent/'sticker-pragmatic-v7/catalog-coverage.json',coverage)
    return membership


def run():
    from research.sticker_encoder.contrast_retrieval import retrieve
    model = sticker_ranker.Metric(OUT/'semantic-ranker-v6.npz')
    assert model.digest == read(OUT/'checkpoint-lock.json')['sha256'] == sticker_ranker.metric().digest
    rows = read(CORPUS)['rows']
    index = {r['id']:i for i,r in enumerate(rows)}
    x = features(rows)
    z = model.project(x)
    scores = z @ z.T
    bge = x @ x.T
    matrices = np.load(MATRICES)
    assert list(matrices['ids']) == list(index)
    style = matrices['style']
    fixed = .6*matrices['meaning']+.4*matrices['v4']
    pools = judgments('test',index)
    scores_by_model = {k:score_pools(pools,m) for k,m in [('v5_fixed_score',fixed),('bge_no_training',bge),('v6',scores)]}
    calibrated = score_pools(pools,scores,model.floor)
    outcomes = []
    for q,choices,good in pools:
        sorted_ids = sorted(range(len(choices)),key=lambda k:(-scores[q,choices[k]],rows[choices[k]]['id']))
        keep = [j for j in sorted_ids if scores[q,choices[j]]>=model.floor]
        outcomes.append({'query':rows[q]['id'],'relevant':[rows[choices[i]]['id'] for i in np.where(good)[0]],
                         'ranked':[rows[choices[j]]['id'] for j in sorted_ids],
                         'passed':[rows[choices[j]]['id'] for j in keep],
                         'judged_pool_size':len(choices)})
    write(OUT/'test-rankings.json',outcomes)
    r = {'checkpoint_sha256':model.digest,'test_exposure':'historical pack-separated holdout; excluded from fitting/selection',
         'sample':'Within-test eligible cross-style choices, 10 per query. Pooled candidates drawn with frozen BGE, legacy score and ALADIN.',
         'ratings':'Codex reviewed shuffled generic captions with no ranks or model names. Context-compatible, same communicative role; no participant/human judgments.',
         'evaluated_stage':'meaning ranking on a fixed cross-style pool','ungated':scores_by_model,'validation_cutoff':model.floor,
         'after_cutoff':calibrated,'empty_truth_pools_returning':sum(bool(x['passed']) and not x['relevant'] for x in outcomes)}
    web = OUT/'web'
    (web/'rows').mkdir(parents=True,exist_ok=True)
    legacy = read(OLD/'two-stage-results.json')['queries']
    rng = np.random.default_rng(2620)
    sample = list(rng.choice([r['id'] for r in rows if r['split']=='test' and r['quality']!='uncertain'],size=14,replace=False))
    blind = {}
    coverage = {'test':[],'validation':[],'train':[]}
    for i,row in enumerate(rows):
        result = retrieve(row,rows,scores[i],style[i],model.floor)
        coverage[row['split']].append(len(result['divergent']))
        result['query'] = row['id']
        result['checkpoint_sha256'] = model.digest
        if row['id'] in sample:
            old = [v['id'] for v in legacy[row['id']]['stage2'][:5]]
            new = [v['id'] for v in result['divergent'][:5]]
            union = sorted(set(old+new))
            rng.shuffle(union)
            blind[row['id']] = {'choices':union,'v5':old,'v6':new}
    r['corpus_coverage'] = {k:{'queries':len(v),'queries_with_divergent_candidate':sum(n>0 for n in v),
            'median_candidates':float(np.median(v))} for k,v in coverage.items()}
    write(OUT/'full-gallery-review-key.json',blind)
    labeled_path = ROOT/'research/sticker_encoder/v6/gallery-judgments.json'
    if labeled_path.exists():
        labeled = read(labeled_path)
        assert set(labeled) == set(blind)
        assert all(set(labeled[q]) <= set(row['choices']) for q,row in blind.items())
        r['full_gallery'] = {'queries':len(blind),'gallery_size':len(rows),
            'review':'Random held-out queries, pooled top five from each native two-stage system, shuffled for Codex screenshot and caption review. No human ratings.',
            'systems':{}}
        for name in ('v5','v6'):
            returned = sum(len(row[name]) for row in blind.values())
            accepted = sum(sum(a in labeled[q] for a in row[name]) for q,row in blind.items())
            r['full_gallery']['systems'][name] = {'returned':returned,'accepted':accepted,
                'queries_with_accepted':sum(any(a in labeled[q] for a in row[name]) for q,row in blind.items()),
                'returned_precision':accepted/max(1,returned)}
    write(OUT/'evaluation-report.json',r)
    print(json.dumps(r,ensure_ascii=False))


if __name__ == '__main__':
    if sys.argv[1:] == ['--cache-contexts']:
        options = list(EXAMPLES)
        vectors = sticker_ranker.local_encoder().encode(list(EXAMPLES.values()))
        np.savez_compressed(OUT/'demo-contexts.npz',keys=options,vectors=vectors)
    else:
        run()
        export_reviewed_rankings()
