"""Shared frozen semantic-group and full-precision maximum policy for arm D."""
import json
import math
from functools import lru_cache

import numpy as np

import sticker_ranker

POLICY = 'situated-intent-full-group-aladin-max-v2'
MIN_AFFINITY = .50
MIN_LEAD = .035
MIN_STYLE = .65
NAMES = dict(laugh='真心开心或共笑',grin='尴尬陪笑',mock='嘲笑',ack='收到或听懂',
             agree='赞同',confused='困惑求解',surprise='意外惊讶',comfort='安慰对方',seek_hug='寻求安慰',
             praise='称赞对方',proud='自我得意',cry='难过委屈',tired='累了',speechless='无语无奈',
             refuse='拒绝',attack='发火或威胁',fear='害怕',love='表达喜爱',thanks='感谢',sorry='道歉',
             cheer='鼓励',sleep='晚安休息',invite='邀请',leave='告别离开',doubt='质疑',money='要红包',hungry='想吃东西')


@lru_cache(maxsize=1)
def prototypes():
    folder = sticker_ranker.WEIGHTS.parent
    dialogues = json.loads((folder/'dialogues.json').read_text())
    keys = []
    with np.load(folder/'intent-phrases.npz', allow_pickle=False) as data:
        if list(data['texts']) != [t for _,texts,_ in dialogues for t in texts]:
            raise ValueError('Mismatched frozen language examples')
        z = sticker_ranker.metric().project(data['vectors'])
    for name, texts, _ in dialogues:
        key = 'laugh' if name == 'party' else name
        keys.extend([key]*len(texts))
    if set(keys) != set(NAMES):
        raise ValueError('Mismatched use groups')
    return z, np.array(keys)


def classify(projected):
    anchors, keys = prototypes()
    sims = np.asarray(projected) @ anchors.T
    groups = sorted(NAMES)
    affinity = np.column_stack([np.sort(sims[:, keys==g], axis=1)[:,-2:].mean(axis=1) for g in groups])
    order = np.argsort(-affinity, axis=1, kind='stable')
    result = []
    for row, pair in zip(affinity, order[:,:2]):
        first, second = pair
        guess = groups[first] if row[first]>=MIN_AFFINITY and row[first]-row[second]>=MIN_LEAD else ''
        result.append(guess)
    return result


def descending_style(entries, id_field='id'):
    # Never round a floating point distance for selection. ID breaks exact ties.
    if any(not math.isfinite(e['style_distance']) for e in entries):
        raise ValueError('Missing style vector in the eligible meaning group')
    return sorted(entries, key=lambda e:(-e['style_distance'], e[id_field]))


def distance_gate(distance):
    if not np.isfinite(distance):
        return 'missing_style_distance'
    if distance < MIN_STYLE:
        return 'insufficient_style_distance'
    return None


def gallery_result(query, rows, labels, group, scores, distances, identity):
    semantic, eligible, blocked = [], [], []
    unresolved = False
    for row, label, s, d in zip(rows, labels, scores, distances):
        if not group or label!=group or row['quality']=='uncertain':
            continue
        entry = dict(id=row['id'], meaning_score=float(s),style_distance=None if not np.isfinite(d) else float(d))
        excluded = identity(query,row)
        if excluded:
            if row['id']!=query['id']:
                blocked.append(dict(entry, reason=excluded))
            continue
        semantic.append(entry)
        reason=distance_gate(d)
        if reason:
            blocked.append(dict(entry,reason=reason))
            unresolved |= reason=='missing_style_distance'
        else:
            eligible.append(entry)
    semantic.sort(key=lambda x:(-x['meaning_score'],x['id']))
    ranks = {e['id']:i+1 for i,e in enumerate(semantic)}
    ranked = [] if unresolved else descending_style(eligible)
    for item in ranked:
        item['meaning_rank']=ranks[item['id']]
    return dict(semantic=semantic, divergent=ranked, maximum=ranked[0] if ranked else None, blocked=blocked,
                explore=[], group=group, group_name=NAMES.get(group,''), group_size=len(semantic),
                incomplete_style=unresolved, policy=POLICY)
