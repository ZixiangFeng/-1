"""Install the pinned author's ALADIN-ViT implementation and optional checkpoint."""
import argparse
import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "3ddd1124bf4a009e649621c680adb03d74a27953"
CODE = ROOT / "vendor/aladin/vit_networks.py"
WEIGHTS = ROOT / "models/style-lab/aladinVIT_bamfg_64.47.pt"
CODE_SHA = "9f6388000c1d40bfb5e5ce7e98f2f6e10e1c99faaa8f76b1e9636ed3a1ab92f1"
UPSTREAM_CODE_SHA = "ff63666fb13e69710d5f3f2bed5f038d386ddd383eb522ee22688ee6abff7ad1"
WEIGHTS_SHA = "310835f8b65349f8900ae7f70c95ee58858537939bdd2c7b86ce36a666b2a91d"


def check(path, expected):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"SHA-256 does not match the author release: {path.name}")


def install(path, expected, url, zipped=False):
    if path.exists():
        check(path, expected)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        download = Path(tmp) / "download"
        with urllib.request.urlopen(url, timeout=90) as src, download.open("wb") as out:
            shutil.copyfileobj(src, out)
        candidate = download
        if zipped:
            candidate = Path(tmp) / "weights"
            with zipfile.ZipFile(download) as archive:
                members = [n for n in archive.namelist() if Path(n).name == path.name]
                if len(members) != 1:
                    raise ValueError("Expected one ALADIN-ViT checkpoint in the archive")
                with archive.open(members[0]) as src, candidate.open("wb") as out:
                    shutil.copyfileobj(src, out)
        elif path == CODE:
            # The original research snapshot stored a final LF after the author's last line.
            check(candidate, UPSTREAM_CODE_SHA)
            with candidate.open("ab") as out:
                out.write(b"\n")
        check(candidate, expected)
        candidate.replace(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-only", action="store_true")
    args = parser.parse_args()
    install(CODE, CODE_SHA, f"https://raw.githubusercontent.com/DanRuta/ALADIN/{COMMIT}/vit_networks.py")
    if not args.code_only:
        install(WEIGHTS, WEIGHTS_SHA, "https://github.com/DanRuta/ALADIN/releases/download/v1.0.0/aladinVIT_bamfg_64.47.zip", zipped=True)
    print("Verified ALADIN source" + ("" if args.code_only else " and checkpoint"))
