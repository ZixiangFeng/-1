"""Build a lazy-loading public-gallery preview with numeric audit downloads."""
import gzip, json, shutil, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import numpy as np
from PIL import Image, ImageOps

out = ROOT / 'outputs/sticker-divergence-v4-20261119'
web = out / 'web'; (web / 'images').mkdir(parents=True, exist_ok=True); (web / 'rows').mkdir(exist_ok=True)
dataset = json.loads((out / 'dataset.json').read_text())
archive = np.load(out / 'retrieval-matrices.npz')
matrices = {key: archive[key] for key in archive.files}
rows = dataset['rows']; assert [r['id'] for r in rows] == list(matrices['ids'])
for i, r in enumerate(rows):
    src = ROOT / 'outputs/sticker-codex-local-20261118' / r['image']
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert('RGB'); im.thumbnail((220, 220)); im.save(web / 'images' / (r['id'] + '.jpg'), quality=80)
    obj = {key: np.round(matrices[key][i], 6).tolist() for key in ('v2', 'v4', 'style', 'meaning')}
    raw = json.dumps(obj, separators=(',', ':')).encode()
    p = web / 'rows' / (r['id'] + '.json'); p.write_bytes(raw); p.with_suffix('.json.gz').write_bytes(gzip.compress(raw, mtime=0))
payload = {'items': [{k: r[k] for k in ('id', 'meaning', 'split', 'quality', 'pack', 'families', 'medium', 'image_sha256')} for r in rows],
           'report': json.loads((out / 'training-report.json').read_text()),
           'evaluation': json.loads((out / 'evaluation-report.json').read_text()),
           'lock': json.loads((out / 'checkpoint-lock.json').read_text()), 'assets': '__ASSET_BASE__'}
html = (ROOT / 'static/semantic_v4.html').read_text().replace('__DATA__', json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c'))
(web / 'report-template.html').write_text(html)
(web / 'index.html').write_text(html.replace('__ASSET_BASE__', './'))
for name in ('evaluation-report.json', 'training-report.json', 'protocol.json', 'two-stage-results.json', 'medium-training-report.json'):
    shutil.copy2(out / name, web / name)
print(json.dumps({'web': str(web), 'images': len(rows), 'index_bytes': (web / 'index.html').stat().st_size}))
