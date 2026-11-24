"""Meaning-only trained retrieval for the divergent sticker arm.

A frozen BGE encoder embeds a conversational utterance or generic meaning. A
learned residual metric scores those embeddings. Gallery caching is limited to
public material. Participant utterances are embedded only for the current call.
"""
import hashlib
from functools import lru_cache
from pathlib import Path
from threading import RLock

import numpy as np

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / 'research/sticker_encoder/v6/semantic-ranker-v6.npz'
VERSION = 'xiaoyu-context-metric-v6'
SEMANTIC_WINDOW = 18


def unit(matrix):
    x = np.asarray(matrix, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Invalid semantic embedding')
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms < 1e-7):
        raise ValueError('Invalid semantic embedding')
    return x / norms


class Metric:
    def __init__(self, path=WEIGHTS):
        with np.load(path, allow_pickle=False) as data:
            if str(data['version']) != VERSION:
                raise ValueError('Unexpected sticker ranker version')
            self.center = data['center']
            self.weight = data['projection']
            self.mix = float(data['mix'])
            self.floor = float(data['floor'])
        if (self.center.shape != (512,) or self.weight.shape[0] != 512
                or not np.isfinite(self.weight).all() or not np.isfinite(self.center).all()
                or not 0 < self.mix < 1 or not -1 <= self.floor <= 1):
            raise ValueError('Invalid sticker ranker weights')
        self.digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def project(self, vectors):
        x = unit(vectors)
        learned = unit((x - self.center) @ self.weight)
        return np.concatenate([np.sqrt(1 - self.mix) * x, np.sqrt(self.mix) * learned], axis=1).astype(np.float32)


@lru_cache(maxsize=1)
def metric():
    return Metric()


class TextEncoder:
    def __init__(self):
        from fastembed import TextEmbedding
        from rag import EMBED_MODEL, EMBED_REVISION
        cache = ROOT / 'models/embeddings'
        snapshot = cache / 'models--Qdrant--bge-small-zh-v1.5' / 'snapshots' / EMBED_REVISION
        self.embedder = TextEmbedding(EMBED_MODEL, cache_dir=str(cache), specific_model_path=str(snapshot),
                                      threads=2, local_files_only=True)

    def encode(self, texts):
        return list(self.embedder.embed(texts, batch_size=32))


@lru_cache(maxsize=1)
def local_encoder():
    return TextEncoder()


_public = {}
_lock = RLock()


def ranked_options(options, query, encoder=None, limit=SEMANTIC_WINDOW, same_intent=False):
    if same_intent:
        import sticker_intent
        import sticker_pragmatics as use
        # A broad review alone supplies no concrete wording for a pair comparison.
        options = [o for o in options if o.get('approved_groups') and
                   ((o.get('usages') or [{}])[0].get('meaning','').strip() or
                    any(g in sticker_intent.NAMES for g in o['approved_groups']))]
    else:
        options = [o for o in options if o.get('usages') and o['usages'][0].get('meaning','').strip()]
    if not options or not query.strip():
        return []
    model = metric()
    meanings = [(o.get('usages') or [{}])[0].get('meaning','').strip() for o in options]
    if same_intent:
        meanings = [m or '、'.join(sticker_intent.NAMES[g] for g in o['approved_groups'] if g in sticker_intent.NAMES) for m,o in zip(meanings,options)]
    with _lock:
        encoder = encoder if callable(getattr(encoder, 'encode', None)) else local_encoder()
        missing = list(dict.fromkeys(m for m in meanings if m not in _public))
        if len(_public) + len(missing) > 4096:
            _public.clear()
            missing = list(dict.fromkeys(meanings))
        # The BGE document encoding is symmetric on both sides, as in training.
        encoded = model.project(encoder.encode([query] + missing))
        _public.update(zip(missing, encoded[1:]))
        projected = np.stack([_public[m] for m in meanings])
        scores = projected @ encoded[0]
    if same_intent:
        language = use.model()
        group = language.groups(encoded[:1],[query])[0]
        if not group:
            return []
        members = []
        query_tags = use.signals(query)
        for i,option in enumerate(options):
            reviewed = option['approved_groups']
            if not any(use.family(g)==group for g in reviewed):
                continue
            text = meanings[i]
            tags = use.signals(text)
            if not use.compatible_use(group,query,text,query_tags,tags):
                continue
            # A narrowly confirmed legacy meaning can explicitly corroborate a
            # paraphrase. A broad checkbox always requires pair evidence.
            corroborated = bool(query_tags & tags & set(reviewed))
            if scores[i]>=language.pair_floor or corroborated:
                members.append(i)
        ranking = sorted(members, key=lambda i:(-scores[i], options[i]['asset_id']))
        return [dict(options[i], semantic_rank=n, semantic_score=float(scores[i]),
                     intent_group=group, semantic_ranker=VERSION, intent_classifier=language.digest) for n,i in enumerate(ranking,1)]
    ranking = sorted(range(len(options)), key=lambda i: (-scores[i], options[i]['asset_id']))
    return [dict(options[i], semantic_rank=rank, semantic_score=round(float(scores[i]), 5),
                 semantic_ranker=VERSION) for rank,i in enumerate(ranking[:limit],1)]


def situated_query(text, reply, reference, current_asset):
    # A new message can change the use of the same image. Previously inferred
    # meanings help on image-only turns; never use inventory or a visual caption.
    uses = reference.get('uses', [])
    latest = uses[0] if uses else {}
    utterance = text.strip()
    if current_asset and latest.get('interpretation'):
        when = latest.get('text','').strip()
        if not utterance or when == utterance:
            return '。'.join(dict.fromkeys(s for s in [utterance, latest['interpretation']] if s))[:700]
    if utterance or reply.strip():
        return '。'.join(s for s in [utterance, reply.strip()] if s)[:700]
    return (latest.get('interpretation') or latest.get('text') or
            '。'.join(x.get('text','') for x in latest.get('preceding',[])[-2:]))[:700]
