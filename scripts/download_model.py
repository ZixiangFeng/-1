import concurrent.futures, hashlib, os, pathlib, time, urllib.request

URL = "https://huggingface.co/mradermacher/rolePlay_Qwen2.5-7B-GGUF/resolve/ae9ac836e1cb6e1b58c83757bd6bf80bf529935a/rolePlay_Qwen2.5-7B.Q4_K_M.gguf"
out = (
    pathlib.Path(__file__).resolve().parents[1]
    / "models/roleplay-qwen2.5-7b-q4_k_m.gguf"
)
expected = "d9fafed4be1e5cb0e932662a087bfc876709786e0ae61016761329b315392c5a"
if out.exists() and out.stat().st_size == 4683073856:
    with out.open("rb") as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    if h.hexdigest() == expected:
        print("Already verified")
        raise SystemExit(0)
size = 4683073856
# Resolve remote size rather than trusting an estimate.
with urllib.request.urlopen(
    urllib.request.Request(URL, method="HEAD"), timeout=60
) as r:
    size = int(r.headers["Content-Length"])
    final = r.url
parts = out.parent / "parts"
parts.mkdir(exist_ok=True)
n = 16
chunk = (size + n - 1) // n
start = time.time()


def fetch(i):
    a = i * chunk
    b = min(size, (i + 1) * chunk) - 1
    p = parts / str(i)
    for attempt in range(8):
        have = p.stat().st_size if p.exists() else 0
        if have == b - a + 1:
            return
        try:
            req = urllib.request.Request(
                final, headers={"Range": f"bytes={a+have}-{b}"}
            )
            with urllib.request.urlopen(req, timeout=90) as r:
                if r.status != 206 or not r.headers.get("Content-Range", "").startswith(
                    f"bytes {a+have}-"
                ):
                    raise RuntimeError("Invalid range response")
                with p.open("ab") as f:
                    while True:
                        block = r.read(1024 * 256)
                        if not block:
                            break
                        f.write(block)
            if p.stat().st_size == b - a + 1:
                return
        except Exception as e:
            print("retry", i, str(e)[:100], flush=True)
    raise RuntimeError("download failed " + str(i))


with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
    futures = [ex.submit(fetch, i) for i in range(n)]
    while not all(f.done() for f in futures):
        downloaded = sum(p.stat().st_size for p in parts.iterdir())
        print(
            f"{downloaded/size:.1%} {downloaded/1024**2:.0f} MB / {size/1024**2:.0f} MB",
            flush=True,
        )
        time.sleep(15)
    for f in futures:
        f.result()
with out.open("wb") as f:
    for i in range(n):
        with (parts / str(i)).open("rb") as p:
            while True:
                block = p.read(1024**2)
                if not block:
                    break
                f.write(block)
assert out.stat().st_size == size
with out.open("rb") as f:
    h = hashlib.sha256()
    for block in iter(lambda: f.read(1048576), b""):
        h.update(block)
assert h.hexdigest() == expected, "Checksum mismatch"
print("Complete and verified", out, flush=True)
for p in parts.iterdir():
    p.unlink()
parts.rmdir()
