"""Real local inference probe, distinct from stubbed unit tests."""

import json, pathlib, urllib.request, time

root = pathlib.Path(__file__).resolve().parents[1]
base = "http://127.0.0.1:8765"
key = (root / "data/admin.key").read_text().strip()


def call(path, body=None, token=None):
    headers = {"Content-Type": "application/json", "X-Admin-Key": key}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(
        base + path, json.dumps(body).encode() if body is not None else None, headers
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


record = []
for arm in ("memory", "control"):
    p = call("/api/admin/enroll", {"label": "QA-live-" + arm, "arm": arm})
    t = p["token"]
    call("/api/consent", {}, t)
    s = call("/api/session", {}, t)
    start = time.time()
    a = call("/api/chat", {"session": s["id"], "text": "论文终于改完了🙂"}, t)
    call(
        "/api/feedback",
        {
            "message": a["message_id"],
            "expression": "emoji:🙂",
            "meaning": "松了一口气，终于结束了，不是在反讽",
            "context": "完成论文修改之后",
        },
        t,
    )
    s = call("/api/session", {}, t)
    b = call("/api/chat", {"session": s["id"], "text": "这一轮修改结束了🙂"}, t)
    record.append(
        {
            "arm": arm,
            "first": a,
            "after_new_session": b,
            "seconds": round(time.time() - start, 2),
        }
    )
    print(json.dumps(record[-1], ensure_ascii=False), flush=True)
(root / "data/live-smoke.json").write_text(
    json.dumps(record, ensure_ascii=False, indent=2)
)
