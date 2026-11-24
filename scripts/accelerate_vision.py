"""Resume verified-registry vision blob using bounded small range requests."""

import concurrent.futures, hashlib, json, os, pathlib, subprocess, threading, time, urllib.request

root = pathlib.Path(__file__).resolve().parents[1]
manifest = json.load(open(root / "data/vision-small-manifest.json"))
item = manifest["layers"][0]
digest = item["digest"]
size = item["size"]
base = root / "models/ollama"
old = base / (digest.replace(":", "-") + ".parts")
output = base / "vision.download"
state = base / "vision.download.json"
guard = threading.Lock()
if not state.exists():
    fd = os.open(output, os.O_CREAT | os.O_RDWR, 0o600)
    os.ftruncate(fd, size)
    jobs = []
    chunk = (size + 11) // 12
    for i in range(12):
        p = old / str(i)
        a = i * chunk
        end = min(size, (i + 1) * chunk)
        have = 0
        if p.exists():
            with p.open("rb") as f:
                while True:
                    block = f.read(1048576)
                    if not block:
                        break
                    os.pwrite(fd, block, a + have)
                    have += len(block)
        for start in range(a + have, end, 4 * 1048576):
            jobs.append([start, min(end, start + 4 * 1048576) - 1])
    os.close(fd)
    state.write_text(json.dumps(jobs))
jobs = json.loads(state.read_text())
pending = {tuple(x) for x in jobs}
fd = os.open(output, os.O_RDWR)
remote = "https://registry.ollama.ai/v2/library/qwen3-vl/blobs/" + digest
with urllib.request.urlopen(
    urllib.request.Request(remote, method="HEAD"), timeout=60
) as r:
    final = r.url


def get(job):
    a, b = job
    for attempt in range(8):
        try:
            req = urllib.request.Request(final, headers={"Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status != 206 or not r.headers.get("Content-Range", "").startswith(
                    f"bytes {a}-"
                ):
                    raise RuntimeError("Wrong range")
                data = r.read(b - a + 2)
            if len(data) != b - a + 1:
                raise RuntimeError("Wrong length")
            os.pwrite(fd, data, a)
            with guard:
                pending.remove(tuple(job))
                tmp = state.with_suffix(".tmp")
                tmp.write_text(json.dumps(list(pending)))
                tmp.replace(state)
            return
        except Exception as e:
            if attempt == 7:
                raise


with concurrent.futures.ThreadPoolExecutor(max_workers=36) as ex:
    fs = [ex.submit(get, j) for j in jobs]
    while not all(f.done() for f in fs):
        with guard:
            remaining = sum(b - a + 1 for a, b in pending)
        print(
            f"Vision {(size-remaining)/size:.1%} ({(size-remaining)//1048576}/{size//1048576} MB)",
            flush=True,
        )
        time.sleep(15)
    for f in fs:
        f.result()
os.close(fd)
h = hashlib.sha256()
with output.open("rb") as f:
    for block in iter(lambda: f.read(1048576), b""):
        h.update(block)
assert h.hexdigest() == digest.split(":")[1], "Checksum mismatch"
output.replace(base / "blobs" / digest.replace(":", "-"))
state.unlink()
print("Main vision blob verified", flush=True)
subprocess.run(
    [str(root / ".venv/bin/python"), str(root / "scripts/download_vision.py")],
    check=True,
)
