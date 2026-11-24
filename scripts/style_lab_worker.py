"""Run the author's ALADIN-ViT network and preprocessing, with CPU cached output.

No random/untrained weights, no cloud inference and no fallback model.
"""
import argparse
import base64
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys


def run(args):
    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))
    import style_lab as lab
    from PIL import Image
    rows = json.loads(Path(args.snapshot).read_text())
    model_id = lab.SEMANTIC_ID if args.semantic else lab.STYLE_ID
    try:
        os.nice(10)
    except (AttributeError, OSError):
        pass
    with lab.cache(args.data) as c:
        existing = {r[0] for r in c.execute('SELECT hash FROM features WHERE model=?', (model_id,))}
    field = 'semantic_hash' if args.semantic else 'hash'
    pending = {r[field]: r for r in rows if r[field] and r[field] not in existing}
    manifest = {}
    if not pending:
        lab.save_status(args.data, {'state': 'running', 'stage': 'semantic' if args.semantic else 'style',
                                   'job': args.job, 'done': len(rows), 'total': len(rows), 'cached': True})
        return
    if args.semantic:
        from fastembed import TextEmbedding
        from rag import EMBED_MODEL, EMBED_REVISION
        model_path = root / 'models/embeddings/models--Qdrant--bge-small-zh-v1.5/snapshots' / EMBED_REVISION
        encoder = TextEmbedding(EMBED_MODEL, threads=1, local_files_only=True,
                                specific_model_path=str(model_path), cache_dir=str(root/'models/embeddings'))
        def encode(row):
            return next(iter(encoder.embed([row['meaning']]))).tolist()
    else:
        import torch
        from torchvision import transforms
        from vendor.aladin.vit_networks import VisionTransformer
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        class AViT(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.enc_style = VisionTransformer()

            def forward(self, image):
                return self.enc_style(image, just_style_code=True)[0]
        ckpt = root / 'models/style-lab/aladinVIT_bamfg_64.47.pt'
        manifest = json.loads((root / 'models/style-lab/manifest.json').read_text())
        if any(manifest[k] != lab.PROVENANCE['style'][k] for k in ('weights_sha256', 'code_sha256')):
            raise ValueError('Model manifest differs from pinned official version')
        actual = hashlib.sha256(ckpt.read_bytes()).hexdigest()
        if actual != manifest['weights_sha256']:
            raise ValueError('ALADIN checkpoint checksum mismatch')
        code = root / 'vendor/aladin/vit_networks.py'
        if hashlib.sha256(code.read_bytes()).hexdigest() != manifest['code_sha256']:
            raise ValueError('ALADIN source checksum mismatch')
        model = AViT()
        model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True), strict=True)
        model.eval()
        transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                                        transforms.Normalize((.5,.5,.5),(.5,.5,.5))])
        def encode(row):
            with Image.open(io.BytesIO(base64.b64decode(row['image']))) as im:
                # Author preprocessing composites RGBA on white; retain full frame.
                im.load()
                if len(im.split()) == 4:
                    rgb = Image.new('RGB', im.size, (255,255,255))
                    rgb.paste(im, mask=im.split()[3])
                else:
                    rgb = im.convert('RGB')
                with torch.inference_mode():
                    vector = model(transform(rgb).unsqueeze(0))
                    vector = vector / vector.norm(dim=1, keepdim=True)
                if vector.shape != (1,256) or not torch.isfinite(vector).all():
                    raise ValueError('Invalid author model output')
                return vector[0].tolist()
    done = 0
    for key, row in pending.items():
        vector = encode(row)
        with lab.cache(args.data) as c:
            c.execute('INSERT OR REPLACE INTO features VALUES(?,?,?,?)',
                      (key, model_id, json.dumps(vector), lab.utc()))
        done += 1
        lab.save_status(args.data, {'state': 'running', 'stage': 'semantic' if args.semantic else 'style',
                                   'job': args.job, 'done': done, 'total': len(pending), 'model': model_id,
                                   'weights_sha256': manifest.get('weights_sha256')})
    print(json.dumps({'model': model_id, 'computed': done, 'cached': len(existing)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'data', 'snapshot', 'job'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--semantic', action='store_true')
    run(parser.parse_args())
