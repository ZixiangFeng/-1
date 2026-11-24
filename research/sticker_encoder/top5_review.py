"""Multi-relevant top-k judgments with explicit unknown bounds, never fake recall."""
import numpy as np

def summarize(rankings, ratings, query_ids=None, k=5):
    ids=list(query_ids) if query_ids is not None else list(rankings)
    models=sorted({m for q in ids for m in rankings[q]});result={}
    for model in models:
        values=[]
        for q in ids:
            keys=rankings[q][model]
            if len(keys)!=k or len(set(keys))!=k:raise ValueError('Exactly k distinct candidates required')
            labels=[ratings.get(key,{}).get('meaning') for key in keys]
            if any(x not in (None,'','yes','no','uncertain') for x in labels):raise ValueError('Unknown rating')
            yes=labels.count('yes');no=labels.count('no');unknown=k-yes-no
            values.append({'query':q,'yes':yes,'no':no,'unknown':unknown,'precision_lower':yes/k,'precision_upper':(yes+unknown)/k,'hit_lower':int(yes>0),'hit_upper':int(yes+unknown>0)})
        complete=[r for r in values if not r['unknown']]
        result[model]={'queries':len(values),'fully_judged_queries':len(complete),'judgment_coverage':sum(r['yes']+r['no'] for r in values)/(k*len(values)) if values else 0,
         'precision_at_5_all_query_bounds':[float(np.mean([r[key] for r in values])) for key in ['precision_lower','precision_upper']] if values else None,
         'hit_at_5_all_query_bounds':[float(np.mean([r[key] for r in values])) for key in ['hit_lower','hit_upper']] if values else None,
         'precision_at_5_complete_queries':float(np.mean([r['precision_lower'] for r in complete])) if complete else None,
         'hit_at_5_complete_queries':float(np.mean([r['hit_lower'] for r in complete])) if complete else None,
         'per_query':values,'recall_at_5':None,'recall_note':'Total relevant gallery items unknown; Recall@5 cannot be estimated from judged top5 only.'}
    return result
