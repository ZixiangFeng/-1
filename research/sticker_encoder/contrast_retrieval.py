"""Two-stage evaluation on a gallery with independently computed style vectors."""
import numpy as np
import sticker_ranker
from research.sticker_encoder.disentangled import identity_reason, style_eligibility


def retrieve(query, rows, scores, style_distances, floor, limit=sticker_ranker.SEMANTIC_WINDOW):
    ordered = sorted(range(len(rows)),key=lambda i:(-float(scores[i]),rows[i]['id']))
    semantic,eligible,blocked,explore = [],[],[],[]
    for i in ordered:
        r = rows[i]
        s = float(scores[i])
        if not np.isfinite(s):
            continue
        item = {'id':r['id'],'meaning_score':round(s,6),'style_distance':round(float(style_distances[i]),6)}
        identity = identity_reason(query,r)
        if identity:
            if r['id'] != query['id'] and s >= floor:
                blocked.append(dict(item,reason=identity))
            continue
        if s < floor:
            if len(explore) < limit and r['quality']!='uncertain':
                explore.append(item)
            if len(explore) >= limit:
                break
            continue
        semantic.append(item)
        reason = style_eligibility(query,r,style_distances[i],.65)
        if r['quality'] == 'uncertain':
            reason = 'ambiguous_caption'
        if reason:
            blocked.append(dict(item,reason=reason))
        elif len(eligible) < limit:
            eligible.append(dict(item,meaning_rank=len(semantic)))
    eligible.sort(key=lambda item:(-item['style_distance'],-item['meaning_score'],item['id']))
    return {'semantic':semantic,'divergent':eligible,'blocked':blocked,'explore':explore}
