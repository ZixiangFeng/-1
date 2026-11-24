# -*- coding: utf-8 -*-
"""Local BGE extraction in the already installed ONNX environment."""
import json,sys
from pathlib import Path
import numpy as np
from fastembed import TextEmbedding
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rag import EMBED_MODEL
out=Path(sys.argv[1]);data=json.loads((out/'dataset.json').read_text());rows=data['rows'];source=ROOT/'outputs/sticker-benchmark-500'
emb=TextEmbedding(EMBED_MODEL,threads=2,local_files_only=True,specific_model_path=str(source/'models/bge'),cache_dir=str(source/'models'))
meaning=np.asarray(list(emb.embed([r['text'] for r in rows],batch_size=32)),dtype=np.float32)
meaning/=np.linalg.norm(meaning,axis=1,keepdims=True)
np.savez_compressed(out/'teacher-features.npz',ids=np.array([r['id'] for r in rows]),meaning=meaning)
print(json.dumps({'local_bge_rows':len(rows)}),flush=True)
