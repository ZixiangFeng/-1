"""Bounded CPU subprocess for cache misses in the participant app's separate Torch runtime."""
import base64
from contextlib import redirect_stdout
import io
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PIL import Image
import torch

from research.sticker_encoder.aladin_features import StyleEncoder


if __name__=='__main__':
    torch.set_num_threads(2)
    with redirect_stdout(sys.stderr):
        model=StyleEncoder(ROOT,'cpu')
    records=json.load(sys.stdin)
    encoded={}
    keys=list(records)
    for start in range(0,len(keys),4):
        batch=keys[start:start+4]
        images=[]
        for key in batch:
            raw=base64.b64decode(records[key].split(',')[-1],validate=True)
            with Image.open(io.BytesIO(raw)) as im:
                if im.width*im.height>16_000_000:
                    raise ValueError('Oversized image')
                images.append(im.convert('RGBA'))
        encoded.update(zip(batch,model.encode(images).tolist()))
    print(json.dumps(encoded,allow_nan=False))
