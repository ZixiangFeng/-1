"""Content-addressed ALADIN features for all eligible public images and this user's reference."""
import json
import os
import subprocess
from contextlib import closing
from pathlib import Path
from threading import Lock

import numpy as np

import sticker_ranker
import style_lab

_lock=Lock()


def distances(c, user, reference, options):
    row=c.execute('SELECT image FROM assets WHERE user=? AND id=?',(user,reference)).fetchone()
    public={r['id']:r['image'] for r in c.execute('SELECT id,image FROM generic_stickers WHERE enabled=1')}
    if not row or not row[0] or any(not public.get(o['asset_id']) for o in options):
        raise ValueError('Missing image in the semantic group')
    images=[row[0],*[public[o['asset_id']] for o in options]]
    keys=[style_lab.digest(im) for im in images]
    data=Path(os.environ.get('DEMO_DATA',sticker_ranker.ROOT/'data'))
    # Keep vectors in the existing style cache. No interaction text is persisted here.
    with _lock, closing(style_lab.cache(data)) as db:
        known={}
        for key in set(keys):
            found=db.execute('SELECT vector FROM features WHERE model=? AND hash=?',(style_lab.STYLE_ID,key)).fetchone()
            if found:
                known[key]=json.loads(found[0])
        missing={k:im for k,im in zip(keys,images) if k not in known}
        if missing:
            python=sticker_ranker.ROOT/'.venv-style/bin/python'
            result=subprocess.run([str(python),str(sticker_ranker.ROOT/'scripts/encode_live_contrast.py')],
                                  input=json.dumps(missing),text=True,capture_output=True,timeout=75,check=True)
            fresh=json.loads(result.stdout)
            if set(fresh)!=set(missing):
                raise ValueError('Incomplete style batch')
            for key,vector in fresh.items():
                x=np.asarray(vector,dtype=np.float64)
                if x.shape!=(256,) or not np.isfinite(x).all() or not .99<np.linalg.norm(x)<1.01:
                    raise ValueError('Invalid style embedding')
                db.execute('INSERT OR IGNORE INTO features VALUES(?,?,?,?)',
                           (key,style_lab.STYLE_ID,json.dumps(vector),style_lab.utc()))
            db.commit()
            known.update(fresh)
    values = np.asarray([known[key] for key in keys], dtype=np.float64)
    if values.shape != (len(images), 256) or not np.isfinite(values).all():
        raise ValueError('Invalid style features')
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError('Empty style feature')
    values /= norms
    distances = 1 - np.clip(values[1:] @ values[0], -1., 1.)
    return dict(zip((option['asset_id'] for option in options), distances.tolist()))
