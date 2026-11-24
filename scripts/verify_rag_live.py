"""Real BGE/Qdrant and companion inference evidence, using QA participants only."""

import json, pathlib, urllib.request, urllib.error

root = pathlib.Path(__file__).resolve().parents[1]
key = (root / "data/admin.key").read_text().strip()


def call(path, body=None, token=None):
    headers = {"Content-Type": "application/json", "X-Admin-Key": key}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(
        "http://127.0.0.1:8765" + path,
        json.dumps(body).encode() if body is not None else None,
        headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(e.read().decode())


results = []
for arm in ("memory", "control"):
    u = call("/api/admin/enroll", {"label": "QA-RAG-" + arm, "arm": arm})
    t = u["token"]
    call("/api/consent", {}, t)
    s = call("/api/session", {}, t)
    dialogue = []
    for text in ["你干嘛呢", "我今天好累", "导师又让我改论文了🙂"]:
        answer = call("/api/chat", {"session": s["id"], "text": text}, t)
        dialogue.append({"text": text, **answer})
        print(arm, text, answer, flush=True)
    call(
        "/api/feedback",
        {
            "message": answer["message_id"],
            "expression": "emoji:🙂",
            "meaning": "强颜欢笑，其实我很无奈",
            "context": "学业压力，导师反复要求修改论文",
        },
        t,
    )
    s = call("/api/session", {}, t)
    answer = call(
        "/api/chat", {"session": s["id"], "text": "这稿子又被打回来重写了🙂"}, t
    )
    detail = call("/api/admin/details?user=" + u["id"])
    audit = json.loads(
        [e for e in detail["events"] if e["kind"] == "generation"][-1]["payload"]
    )
    assert bool(audit["memory_ids"]) == (arm == "memory")
    assert audit["rag"]["embedding_model"] == "BAAI/bge-small-zh-v1.5"
    assert audit["rag"]["policies"]
    assert audit["companion_plan"]["need"]
    results.append(
        {"arm": arm, "dialogue": dialogue, "new_session": answer, "audit": audit}
    )
(root / "data/live-rag-smoke.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2)
)
print("RAG_LIVE_PASS", flush=True)
