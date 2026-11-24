"""Encode a new image with our trained visual model and search an indexed gallery."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from transformers import ChineseCLIPModel, ChineseCLIPProcessor
from research.sticker_encoder.weak_model import digest, rgb


def main(args):
    run = Path(args.run)
    report = json.loads((run / 'training-report.json').read_text())
    if report['state'] != 'trained_evaluated':
        raise ValueError('Training has not completed')
    checkpoint = run / 'checkpoint'
    for name, sha in report['checkpoint_files'].items():
        path = (checkpoint / name).resolve()
        if path.parent != checkpoint.resolve() or digest(path) != sha:
            raise ValueError('Checkpoint integrity failure')
    device = torch.device(args.device)
    model = ChineseCLIPModel.from_pretrained(checkpoint, local_files_only=True, use_safetensors=True).eval().to(device)
    processor = ChineseCLIPProcessor.from_pretrained(checkpoint, local_files_only=True)
    with torch.inference_mode():
        v = model.get_image_features(**processor(images=rgb(args.image), return_tensors='pt').to(device))
        v = torch.nn.functional.normalize(v, dim=-1)[0].cpu().numpy()
    features = np.load(run / 'features.npz', allow_pickle=False)
    data = json.loads((run / 'dataset.json').read_text())
    image_hash = digest(args.image)
    own = {r['id'] for r in data['rows'] if r['image_sha256'] == image_hash}
    scores = features['trained'] @ v
    ranked = sorted([{'id': str(id_), 'image_semantic_cosine': float(score)}
                     for id_, score in zip(features['ids'], scores) if str(id_) not in own],
                    key=lambda r: -r['image_semantic_cosine'])[:args.top_k]
    print(json.dumps({'model': report['implementation'], 'input': str(args.image), 'human_calibrated': False,
                      'ranking': ranked, 'note': 'Image-only inference. Neighbors are candidates, not proven synonyms.'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--top-k', type=int, default=10)
    main(p.parse_args())
