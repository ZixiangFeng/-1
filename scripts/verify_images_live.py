"""Real image upload, correction, new-session recall and access check."""

import base64, json, pathlib, time, urllib.request, urllib.error

root = pathlib.Path(__file__).resolve().parents[1]
base = "http://127.0.0.1:8765"
key = (root / "data/admin.key").read_text().strip()


def call(path, body=None, token=None):
    h = {"Content-Type": "application/json", "X-Admin-Key": key}
    if token:
        h["Authorization"] = "Bearer " + token
    req = urllib.request.Request(
        base + path, json.dumps(body).encode() if body is not None else None, h
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(e.read().decode())


p = call("/api/admin/enroll", {"label": "QA-image-live", "arm": "memory"})
token = p["token"]
call("/api/consent", {}, token)
s = call("/api/session", {}, token)
start = time.time()
a = call(
    "/api/upload",
    {
        "image": base64.b64encode(
            (root / "tests/fixtures/smile-study.png").read_bytes()
        ).decode()
    },
    token,
)
print("Vision:", a, flush=True)
r = call(
    "/api/chat", {"session": s["id"], "text": "又要改论文了", "asset": a["id"]}, token
)
call(
    "/api/feedback",
    {
        "message": r["message_id"],
        "expression": "image:" + a["id"],
        "meaning": "无奈地自嘲，不是真的开心",
        "context": "反复修改论文时",
    },
    token,
)
s2 = call("/api/session", {}, token)
r2 = call(
    "/api/chat",
    {"session": s2["id"], "text": "导师让我再改一遍", "asset": a["id"]},
    token,
)
details = call("/api/admin/details?user=" + p["id"])
generations = [
    json.loads(e["payload"]) for e in details["events"] if e["kind"] == "generation"
]
assert generations[-1]["memory_ids"], "Confirmed image meaning not retrieved"
result = {
    "asset": a,
    "first": r,
    "after_correction_new_session": r2,
    "memory_ids": generations[-1]["memory_ids"],
    "elapsed_seconds": round(time.time() - start, 2),
}
# Same fixture, separate participant: no personal semantics may cross users or arms.
control = call("/api/admin/enroll", {"label": "QA-image-control", "arm": "control"})
ct = control["token"]
call("/api/consent", {}, ct)
cs = call("/api/session", {}, ct)
ca = call(
    "/api/upload",
    {
        "image": base64.b64encode(
            (root / "tests/fixtures/smile-study.png").read_bytes()
        ).decode()
    },
    ct,
)
assert ca["id"] != a["id"], "Image identifiers must be participant scoped"
cr = call(
    "/api/chat", {"session": cs["id"], "text": "又要改论文了", "asset": ca["id"]}, ct
)
call(
    "/api/feedback",
    {
        "message": cr["message_id"],
        "expression": "image:" + ca["id"],
        "meaning": "无奈地自嘲，不是真的开心",
        "context": "反复修改论文时",
    },
    ct,
)
cs2 = call("/api/session", {}, ct)
cr2 = call(
    "/api/chat",
    {"session": cs2["id"], "text": "导师让我再改一遍", "asset": ca["id"]},
    ct,
)
cd = call("/api/admin/details?user=" + control["id"])
cg = [json.loads(e["payload"]) for e in cd["events"] if e["kind"] == "generation"]
assert not cg[-1]["memory_ids"], "Control must not recall prior personal meaning"
result["control"] = {
    "asset": ca,
    "first": cr,
    "after_correction_new_session": cr2,
    "memory_ids": cg[-1]["memory_ids"],
}
result["elapsed_seconds"] = round(time.time() - start, 2)
(root / "data/live-image-smoke.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2)
)
print(json.dumps(result, ensure_ascii=False), flush=True)
