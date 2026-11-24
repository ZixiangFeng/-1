"""Complete archived caption features only after the semantic checkpoint lock."""
import json, sys
from pathlib import Path
import numpy as np
from fastembed import TextEmbedding
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag import EMBED_MODEL
out = ROOT / 'outputs/sticker-divergence-v4-20261119'
if not (out / 'checkpoint-lock.json').exists():
    raise ValueError('Lock model first')
rows = json.loads((out / 'dataset.json').read_text())['rows']
old = np.load(ROOT / 'outputs/sticker-codex-local-20261118/teacher-features.npz')
vectors = {str(k): v for k, v in zip(old['ids'], old['meaning'])}
pending = [r for r in rows if r['id'] not in vectors]
if any(r['split'] != 'test' for r in pending):
    raise ValueError('Unexpected missing development description')
source = ROOT / 'outputs/sticker-benchmark-500'
emb = TextEmbedding(EMBED_MODEL, threads=2, local_files_only=True,
                    specific_model_path=str(source / 'models/bge'), cache_dir=str(source / 'models'))
for row, vector in zip(pending, emb.embed([r['meaning'] for r in pending], batch_size=32)):
    vectors[row['id']] = vector / np.linalg.norm(vector)
np.savez_compressed(out / 'all-caption-features.npz', ids=np.array([r['id'] for r in rows]),
                    meaning=np.array([vectors[r['id']] for r in rows]))
print(json.dumps({'encoded_after_lock': len(pending), 'total': len(rows), 'paid_api_calls': 0}))
