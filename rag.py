"""Persistent local Qdrant + Chinese dense embeddings. SQLite remains authoritative."""

import hashlib
import json
import threading
import uuid
from pathlib import Path
from qdrant_client import QdrantClient, models
from companion import CARDS, POLICY_VERSION

EMBED_MODEL = "BAAI/bge-small-zh-v1.5"
EMBED_REVISION = "46fbe35fd4374a00fee7de77dfddaeb6dd6a2c59"
COLLECTION = "expression_memory_bge_zh_v1"


class RagStore:
    def __init__(self, path, cache, embedder=None):
        if embedder is None:
            from fastembed import TextEmbedding

            model_path = (
                Path(cache)
                / "models--Qdrant--bge-small-zh-v1.5"
                / "snapshots"
                / EMBED_REVISION
            )
            if not model_path.exists():
                raise ValueError(
                    "请先运行 scripts/setup_embeddings.py 安装中文嵌入模型"
                )
            embedder = TextEmbedding(
                EMBED_MODEL,
                cache_dir=str(cache),
                threads=2,
                local_files_only=True,
                specific_model_path=str(model_path),
            )
        self.embedder = embedder
        self.lock = threading.RLock()
        self.client = QdrantClient(path=str(path), force_disable_check_same_thread=True)
        dimension = len(self.encode(["维度检查"])[0])
        if not self.client.collection_exists(COLLECTION):
            self.client.create_collection(
                COLLECTION,
                vectors_config=models.VectorParams(
                    size=dimension, distance=models.Distance.COSINE
                ),
            )
        self.upsert(
            [
                {
                    "id": "policy:" + x["id"],
                    "user": "__policy__",
                    "kind": "policy",
                    "text": x["text"],
                    "example": x["example"],
                    "version": POLICY_VERSION,
                }
                for x in CARDS
            ]
        )

    def encode(self, texts, query=False):
        method = self.embedder.query_embed if query else self.embedder.embed
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in method(texts)]

    @staticmethod
    def point_id(key):
        return str(uuid.uuid5(uuid.NAMESPACE_URL, "xiaoyu:" + key))

    def upsert(self, records):
        if not records:
            return
        ids = [self.point_id(r["id"]) for r in records]
        with self.lock:
            old = {
                p.id: p.payload.get("signature")
                for p in self.client.retrieve(COLLECTION, ids=ids)
            }
            pending = []
            for r, pid in zip(records, ids):
                signature = hashlib.sha256(
                    json.dumps(r, ensure_ascii=False, sort_keys=True).encode()
                ).hexdigest()
                if old.get(pid) != signature:
                    pending.append((pid, dict(r, signature=signature)))
            if pending:
                vectors = self.encode([r["text"] for _, r in pending])
                self.client.upsert(
                    COLLECTION,
                    points=[
                        models.PointStruct(id=p, vector=v, payload=r)
                        for (p, r), v in zip(pending, vectors)
                    ],
                )

    def search(self, query, user, kind, limit=8):
        with self.lock:
            vector = self.encode([query or "表情含义"], query=True)[0]
            hits = self.client.query_points(
                COLLECTION,
                query=vector,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="user", match=models.MatchValue(value=user)
                        ),
                        models.FieldCondition(
                            key="kind", match=models.MatchValue(value=kind)
                        ),
                    ]
                ),
                limit=limit,
            ).points
            return [dict(p.payload, score=round(p.score, 5)) for p in hits]

    def context(self, c, user, arm, text, asset, keys):
        description = ""
        if asset:
            row = c.execute(
                "SELECT description FROM assets WHERE id=? AND user=?", (asset, user)
            ).fetchone()
            if not row:
                raise ValueError("图片不属于当前参与者")
            description = row["description"]
        query = (text + "\n" + description)[:1200]
        policies = self.search(query, "__policy__", "policy", 2)
        selected = []
        if arm in ("memory", "M") and keys:
            rows = [
                dict(r)
                for r in c.execute(
                    "SELECT * FROM memories WHERE user=? AND status IN ('confirmed','inferred')",
                    (user,),
                )
            ]
            docs = []
            for r in rows:
                desc = ""
                if r["expression"].startswith("image:"):
                    a = c.execute(
                        "SELECT description FROM assets WHERE id=? AND user=?",
                        (r["expression"][6:], user),
                    ).fetchone()
                    desc = a["description"] if a else ""
                docs.append(
                    {
                        "id": r["id"],
                        "user": user,
                        "kind": "memory",
                        "expression": r["expression"],
                        "text": (
                            "表情："
                            + r["expression"]
                            + "\n图片："
                            + desc[:200]
                            + "\n情境："
                            + r["context"][:240]
                            + "\n含义："
                            + r["meaning"][:200]
                        ),
                    }
                )
            self.upsert(docs)
            # Remove stale vectors as well as checking the authoritative SQL rows below.
            with self.lock:
                old, _ = self.client.scroll(
                    COLLECTION,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="user", match=models.MatchValue(value=user)
                            ),
                            models.FieldCondition(
                                key="kind", match=models.MatchValue(value="memory")
                            ),
                        ]
                    ),
                    limit=10000,
                )
                live = {r["id"] for r in rows}
                stale = [p.id for p in old if p.payload["id"] not in live]
                if stale:
                    self.client.delete(COLLECTION, models.PointIdsList(points=stale))
            byid = {r["id"]: r for r in rows}
            # Pilot sizes permit exhaustive filtered ranking, not a global cross-user top-k.
            hits = self.search(query, user, "memory", max(8, len(rows))) if rows else []
            for h in hits:
                r = byid.get(h["id"])
                if not r:
                    continue
                exact = r["expression"] in keys
                similar = bool(
                    asset
                    and r["expression"].startswith("image:")
                    and h["score"] >= 0.65
                )
                if exact or similar:
                    selected.append(
                        dict(
                            r,
                            score=h["score"],
                            match=(
                                "same_expression"
                                if exact
                                else "similar_image_unconfirmed"
                            ),
                        )
                    )
            selected.sort(
                key=lambda r: (r["match"] == "same_expression", r["score"]),
                reverse=True,
            )
            selected = selected[:4]
        return {
            "memories": selected,
            "policies": policies,
            "embedding_model": EMBED_MODEL,
            "query": query,
            "policy_version": POLICY_VERSION,
        }

    def close(self):
        self.client.close()


_store = None
_init_lock = threading.Lock()


def get_store(data, root):
    global _store
    with _init_lock:
        if _store is None:
            _store = RagStore(Path(data) / "vectors", Path(root) / "models/embeddings")
    return _store
