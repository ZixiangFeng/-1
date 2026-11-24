import concurrent.futures, hashlib, pathlib, sys, urllib.request

url, path, digest = sys.argv[1:]
p = pathlib.Path(path)
with urllib.request.urlopen(
    urllib.request.Request(url, method="HEAD"), timeout=60
) as r:
    size = int(r.headers["Content-Length"])
    final = r.url
have = p.stat().st_size if p.exists() else 0
if have < size:
    n = 4
    chunk = (size - have + n - 1) // n

    def get(i):
        a = have + i * chunk
        b = min(size, have + (i + 1) * chunk) - 1
        if a > b:
            return b""
        req = urllib.request.Request(final, headers={"Range": f"bytes={a}-{b}"})
        with urllib.request.urlopen(req, timeout=180) as r:
            if r.status != 206 or not r.headers.get("Content-Range", "").startswith(
                f"bytes {a}-"
            ):
                raise RuntimeError("Bad range")
            data = r.read()
            if len(data) != b - a + 1:
                raise RuntimeError("Wrong size")
            return data

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        parts = list(ex.map(get, range(n)))
    with p.open("ab") as f:
        for data in parts:
            f.write(data)
h = hashlib.sha256(p.read_bytes()).hexdigest()
print("SHA256", h, flush=True)
if digest != "-" and h != digest:
    raise RuntimeError("Checksum mismatch")
print("Complete", flush=True)
