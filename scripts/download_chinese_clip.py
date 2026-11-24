"""Download a pinned public Chinese-CLIP checkpoint without touching chat models."""
import hashlib
import json
from pathlib import Path
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import shutil
import os
import socket
import subprocess
import ipaddress

REPO = 'OFA-Sys/chinese-clip-vit-base-patch16'
REVISION = '36e679e65c2a2fead755ae21162091293ad37834'
WEIGHT_SHA256 = '7b7b583c210c867410bc6bdb8a55fe14eec62999e0a9ea31ff222dc501f9cfbe'
WEIGHT_SIZE = 753177983
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'models/research/chinese-clip-vit-base-patch16'

def optional_dns_override():
    """Process-local DNS fallback; HTTPS hostname verification stays enabled."""
    resolver = os.environ.get('DOWNLOAD_DNS_RESOLVER')
    if not resolver:
        return
    ipaddress.ip_address(resolver)
    original = socket.getaddrinfo
    hosts = {}
    for host in ('huggingface.co', 'us.aws.cdn.hf.co'):
        try:
            hosts[host] = list(dict.fromkeys(r[4][0] for r in original(host, 443, socket.AF_INET, socket.SOCK_STREAM)))
            if hosts[host]:
                continue
        except socket.gaierror:
            pass
        result = subprocess.run(['dig', '+time=3', '+tries=2', '+short', 'A', host, '@' + resolver],
                                check=True, capture_output=True, text=True)
        hosts[host] = []
        for line in result.stdout.splitlines():
            try:
                address = ipaddress.ip_address(line.strip())
                if address.version == 4:
                    hosts[host].append(str(address))
            except ValueError:
                pass
        if not hosts[host]:
            raise ValueError('DNS fallback returned no IPv4 addresses for ' + host)
    def resolve(host, port, family=0, type=0, proto=0, flags=0):
        if host in hosts:
            return [record for address in hosts[host] for record in original(address, port, socket.AF_INET, type, proto, flags)]
        return original(host, port, family, type, proto, flags)
    socket.getaddrinfo = resolve

def ranged_download(url, path):
    temp = path.with_suffix(path.suffix + '.partial')
    # Recover the complete prefix left by the interrupted streaming download.
    prefix = temp.stat().st_size if temp.exists() else 0
    total = WEIGHT_SIZE  # Pinned upstream X-Linked-Size; every range is verified below.
    if prefix > total:
        raise ValueError('Invalid existing prefix')
    pieces = OUT / 'download-parts'
    pieces.mkdir(exist_ok=True)
    ranges = [(start, min(total, start + 16 * 1024 * 1024) - 1) for start in range(prefix, total, 16 * 1024 * 1024)]
    def part(bounds):
        start, end = bounds
        p = pieces / f'{start}-{end}'
        if p.exists() and p.stat().st_size == end - start + 1:
            return p
        for attempt in range(10):
            try:
                progress = p.with_suffix('.tmp')
                offset = progress.stat().st_size if progress.exists() else 0
                if offset == end - start + 1:
                    progress.replace(p)
                    return p
                actual_start = start + offset
                with requests.get(url + f'?range_start={actual_start}', headers={'Range': f'bytes={actual_start}-{end}'},
                                  timeout=(25, 45), stream=True) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {actual_start}-{end}/{total}':
                        raise ValueError('Incorrect range returned: ' + str(response.headers.get('Content-Range')))
                    with progress.open('ab') as f:
                        for chunk in response.iter_content(1024 * 1024):
                            f.write(chunk)
                if p.with_suffix('.tmp').stat().st_size != end - start + 1:
                    raise ValueError('Incomplete range')
                p.with_suffix('.tmp').replace(p)
                print(json.dumps({'range_ready_mb': round(start / 1024**2), 'size_mb': round(p.stat().st_size / 1024**2)}), flush=True)
                return p
            except (requests.RequestException, ValueError) as e:
                print(json.dumps({'retry_range_mb': round(start/1024**2), 'attempt': attempt+1, 'error_type': type(e).__name__}), flush=True)
                if attempt == 9:
                    raise
                time.sleep(min(10, 2 * (attempt + 1)))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(part, ranges))
    combined = path.with_suffix('.complete')
    with combined.open('wb') as target:
        if prefix:
            with temp.open('rb') as f:
                shutil.copyfileobj(f, target)
        for start, end in ranges:
            with (pieces / f'{start}-{end}').open('rb') as f:
                shutil.copyfileobj(f, target)
    if combined.stat().st_size != total:
        raise ValueError('Combined file size mismatch')
    combined.replace(path)
    temp.unlink(missing_ok=True)
    shutil.rmtree(pieces)

def main():
    optional_dns_override()
    OUT.mkdir(parents=True, exist_ok=True)
    records = {}
    for name in ['config.json', 'preprocessor_config.json', 'vocab.txt', 'README.md', 'pytorch_model.bin']:
        path = OUT / name
        url = f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}'
        if not path.exists():
            if name == 'pytorch_model.bin':
                ranged_download(url, path)
                continue_record = False
            else:
                continue_record = True
        else:
            continue_record = False
        if continue_record:
            temp = path.with_suffix(path.suffix + '.partial')
            with requests.get(url, stream=True, timeout=(30, 120)) as r:
                r.raise_for_status()
                count = 0
                with temp.open('wb') as f:
                    for chunk in r.iter_content(4 * 1024 * 1024):
                        f.write(chunk)
                        count += len(chunk)
                        if count // (64 * 1024 * 1024) != (count - len(chunk)) // (64 * 1024 * 1024):
                            print(json.dumps({'file': name, 'downloaded_mb': round(count / 1024 ** 2)}), flush=True)
                expected = r.headers.get('Content-Length')
                if expected and count != int(expected):
                    raise ValueError('Incomplete download')
            temp.replace(path)
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
                h.update(chunk)
        records[name] = {'bytes': path.stat().st_size, 'sha256': h.hexdigest()}
        if name == 'pytorch_model.bin' and h.hexdigest() != WEIGHT_SHA256:
            raise ValueError('Weight checksum does not match the pinned upstream LFS object')
        print(json.dumps({'ready': name, **records[name]}), flush=True)
    (OUT / 'download.json').write_text(json.dumps({'repository': REPO, 'revision': REVISION, 'files': records}, indent=2))

if __name__ == '__main__':
    main()
