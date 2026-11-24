"""Prepare/train/evaluate an own GSE-inspired Chinese-CLIP experiment on public data.

No participant database access. No automatic promotion to the production model.
"""
import argparse
import base64
from collections import Counter
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import ChineseCLIPModel, ChineseCLIPProcessor
from research.sticker_encoder.weak_model import IMPLEMENTATION, contrastive_loss, digest, retrieval, rgb, save, split_groups


def prepare(source, out, seed):
    import sticker_identity
    manifest = json.loads((source / 'manifest.json').read_text())
    rows = []
    for r in manifest['items']:
        a = json.loads((source / 'annotations' / (r['id'] + '.json')).read_text())
        text = a['meaning'].strip()
        if not text:
            raise ValueError('Missing meaning; do not silently drop samples')
        rows.append({**r, 'text': text, 'label_source': 'doubao_visual_auto', 'human_reviewed': False,
                     'annotation_sha256': digest(source / 'annotations' / (r['id'] + '.json')),
                     'image_sha256': digest(source / r['image'])})
    fingerprints = [sticker_identity.fingerprint(base64.b64encode((source / r['image']).read_bytes()).decode()) for r in rows]
    coarse = np.array([list(f['coarse']) for f in fingerprints], dtype=np.float32)
    near = []
    for i, fp in enumerate(fingerprints):
        for j in np.flatnonzero(np.mean(np.abs(coarse - coarse[i]), axis=1) <= 8):
            if j > i and sticker_identity.same_image(fp, fingerprints[j]):
                near.append((rows[i]['id'], rows[j]['id']))
    splits = split_groups(rows, near, seed)
    assignments = {id_: split for split, ids in splits.items() for id_ in ids}
    for r in rows:
        r['split'] = assignments[r['id']]
    # Reuse automatic meaning embeddings only for masking likely false negatives.
    feat = np.load(source / 'features.npz', allow_pickle=False)
    by_id = {str(id_): i for i, id_ in enumerate(feat['ids'])}
    teacher = np.array([feat['semantic'][by_id[r['id']]] for r in rows], dtype=np.float32)
    teacher /= np.linalg.norm(teacher, axis=1, keepdims=True)
    np.savez_compressed(out / 'teacher-features.npz', meaning=teacher, ids=np.array([r['id'] for r in rows]))
    report = {'source': str(source), 'source_commit': manifest['commit'], 'source_manifest_sha256': digest(source / 'manifest.json'),
              'seed': seed, 'counts': {k: len(v) for k, v in splits.items()},
              'packs': {k: len({r['pack'] for r in rows if r['split'] == k}) for k in splits},
              'near_duplicate_links_kept_together': near, 'splits': splits,
              'label_source': 'Machine-generated descriptions; NOT human gold labels.',
              'split_note': 'Source pack + detected near-duplicate connected components; unknown cross-pack characters may remain.',
              'test_note': 'Held out from this training/selection; these images were previously explored in the BGE benchmark, not a new blind external test.',
              'rows': rows}
    save(out / 'dataset.json', report)
    return report


def run(args):
    source, base, out = Path(args.source).resolve(), Path(args.base).resolve(), Path(args.output).resolve()
    if out.exists():
        raise ValueError('Use a new run directory; do not overwrite previous checkpoints')
    if args.epochs < 1 or args.batch_size < 2 or not 1 <= args.train_layers <= 12 or args.lr <= 0 or args.projection_lr <= 0:
        raise ValueError('Invalid training settings')
    out.mkdir(parents=True)
    started = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(4)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    data = prepare(source, out, args.seed)
    rows = data['rows']
    print(json.dumps({'event': 'dataset_ready', 'counts': data['counts'], 'packs': data['packs']}), flush=True)
    device = torch.device(args.device)
    base_manifest = json.loads((base / 'download.json').read_text())
    for name, info in base_manifest['files'].items():
        if digest(base / name) != info['sha256']:
            raise ValueError('Base model integrity check failed')
    model = ChineseCLIPModel.from_pretrained(base, local_files_only=True, use_safetensors=False, weights_only=True).to(device)
    processor = ChineseCLIPProcessor.from_pretrained(base, local_files_only=True)
    for p in model.parameters():
        p.requires_grad_(False)
    layers = model.vision_model.encoder.layers
    for layer in layers[-args.train_layers:]:
        for p in layer.parameters():
            p.requires_grad_(True)
    for module in [model.visual_projection, model.vision_model.post_layernorm]:
        for p in module.parameters():
            p.requires_grad_(True)
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    print(json.dumps({'event': 'model_loaded', 'trainable_parameters': sum(p.numel() for p in trainable.values()),
                      'total_parameters': sum(p.numel() for p in model.parameters()), 'device': str(device)}), flush=True)
    # Cache pixels and frozen text embeddings. Captions never enter image inference.
    pixels = processor(images=[rgb(source / r['image']) for r in rows], return_tensors='pt')['pixel_values']
    model.eval()
    text_vectors = []
    with torch.inference_mode():
        for start in range(0, len(rows), 32):
            inputs = processor(text=[r['text'] for r in rows[start:start+32]], padding=True,
                               truncation=True, max_length=64, return_tensors='pt').to(device)
            text_vectors.append(F.normalize(model.get_text_features(**inputs), dim=-1).cpu())
    text_vectors = torch.cat(text_vectors).clone()
    teacher = torch.from_numpy(np.load(out / 'teacher-features.npz')['meaning'])
    teacher_sim = teacher @ teacher.T
    caption_map = {t: i for i, t in enumerate(sorted({r['text'] for r in rows}))}
    caption_ids = [caption_map[r['text']] for r in rows]
    split_idx = {s: [i for i, r in enumerate(rows) if r['split'] == s] for s in data['splits']}

    @torch.inference_mode()
    def encode(indices):
        model.eval()
        vectors = []
        for start in range(0, len(indices), args.batch_size):
            batch = indices[start:start+args.batch_size]
            v = F.normalize(model.get_image_features(pixel_values=pixels[batch].to(device)), dim=-1)
            if not torch.isfinite(v).all():
                raise ValueError('Nonfinite image features')
            vectors.append(v.cpu())
        return torch.cat(vectors).numpy()

    baseline_vectors = encode(list(range(len(rows))))
    val_idx = split_idx['validation']
    baseline_validation = retrieval(baseline_vectors[val_idx], text_vectors[val_idx].numpy())
    best_score = -1.
    best_epoch = 0
    best_state = None
    optimizer = torch.optim.AdamW([
        {'params': [p for name, p in trainable.items() if not name.startswith('visual_projection')], 'lr': args.lr},
        {'params': list(model.visual_projection.parameters()), 'lr': args.projection_lr}
    ], weight_decay=.01)
    records = []
    print(json.dumps({'event': 'baseline_validation', 'metrics': baseline_validation}), flush=True)
    for epoch in range(1, args.epochs + 1):
        indices = split_idx['train'].copy()
        random.shuffle(indices)
        model.eval()  # Keep frozen backbone/dropout deterministic, enable autograd below.
        for layer in layers[-args.train_layers:]:
            layer.train()
        total, seen = 0., 0
        for start in range(0, len(indices), args.batch_size):
            batch = indices[start:start+args.batch_size]
            if len(batch) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            image_v = F.normalize(model.get_image_features(pixel_values=pixels[batch].to(device)), dim=-1)
            loss = contrastive_loss(image_v, text_vectors[batch].to(device), [caption_ids[i] for i in batch],
                                    teacher_sim[batch][:, batch].to(device), model.logit_scale.detach().exp().clamp(max=100))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.0)
            if not torch.isfinite(grad_norm):
                raise ValueError('Nonfinite gradients')
            optimizer.step()
            total += float(loss.item()) * len(batch)
            seen += len(batch)
            print(json.dumps({'event': 'step', 'epoch': epoch, 'seen': seen, 'loss': float(loss.item()),
                              'seconds': round(time.time() - started, 1)}), flush=True)
        validation = retrieval(encode(val_idx), text_vectors[val_idx].numpy())
        score = validation['image_to_caption']['mrr']
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {name: p.detach().cpu().clone() for name, p in trainable.items()}
        records.append({'epoch': epoch, 'loss': total / seen, 'validation': validation})
        save(out / 'progress.json', {'state': 'training', 'epochs': records, 'best_epoch': best_epoch})
        print(json.dumps({'event': 'epoch', **records[-1], 'best_epoch': best_epoch}), flush=True)
    with torch.no_grad():
        for name, p in trainable.items():
            p.copy_(best_state[name].to(device))
    tuned_vectors = encode(list(range(len(rows))))
    test_idx = split_idx['test']
    # No test score is consulted until checkpoint selection is over.
    baseline_test = retrieval(baseline_vectors[test_idx], text_vectors[test_idx].numpy())
    tuned_test = retrieval(tuned_vectors[test_idx], text_vectors[test_idx].numpy())
    checkpoint = out / 'checkpoint'
    model.save_pretrained(checkpoint, safe_serialization=True)
    processor.save_pretrained(checkpoint)
    # Also save a small adapter-only state for reproducible deployment on the pinned base.
    from safetensors.torch import save_file
    save_file({name: p.detach().cpu().contiguous() for name, p in trainable.items()}, str(out / 'visual-finetune.safetensors'))
    np.savez_compressed(out / 'features.npz', ids=np.array([r['id'] for r in rows]),
                        baseline=baseline_vectors, trained=tuned_vectors, text=text_vectors.numpy())
    passed = tuned_test['image_to_caption']['mrr'] > baseline_test['image_to_caption']['mrr']
    report = {'implementation': IMPLEMENTATION, 'author_original': False, 'state': 'trained_evaluated',
              'supervision': 'weak_machine_captions', 'human_gold_evaluation': False,
              'base': base_manifest,
              'paper_inspiration': 'https://arxiv.org/html/2511.04977v1',
              'method': 'Frozen Chinese text tower; fine-tuned last vision blocks and projection; symmetric image-text contrastive loss.',
              'negative_mask': 'Within training batches only: automatic BGE meaning cosine >0.80 masks likely false negatives; identical captions are multi-positive.',
              'inference': 'Image pixels only; no Doubao or BGE required for this model at inference.',
              'settings': vars(args), 'trainable_parameter_count': sum(p.numel() for p in trainable.values()),
              'software': {'torch': torch.__version__, 'transformers': transformers.__version__, 'numpy': np.__version__,
                           'python': sys.version, 'device': str(device)},
              'code_sha256': {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__).resolve(), ROOT / 'research/sticker_encoder/weak_model.py']},
              'trainable_parameter_names': list(trainable), 'epochs': records, 'selected_epoch': best_epoch,
              'selection_metric': 'validation image-to-caption MRR', 'baseline_validation': baseline_validation,
              'baseline_test': baseline_test, 'trained_test': tuned_test,
              'improved_on_heldout_caption_mrr': passed,
              'deployment_status': 'research_preview_only; no production switch; human pair evaluation required',
              'data_counts': data['counts'], 'dataset_sha256': digest(out / 'dataset.json'),
              'seconds': round(time.time() - started, 1),
              'checkpoint_files': {p.name: digest(p) for p in checkpoint.iterdir() if p.is_file()},
              'limitations': ['Not author GSE and not an exact replication.',
                  'Only 500 public static images; machine descriptions may be wrong.',
                  'Caption retrieval is a proxy task, not a same-meaning accuracy estimate.',
                  'Source-pack splits reduce but cannot guarantee removal of all related characters.',
                  'Held-out images were previously explored in the BGE benchmark.',
                  'No human semantic threshold or contextual/person-specific validity established.']}
    save(out / 'training-report.json', report)
    save(out / 'progress.json', {'state': 'completed', 'selected_epoch': best_epoch, 'seconds': report['seconds']})
    print(json.dumps({'event': 'completed', 'output': str(out), 'baseline_test': baseline_test,
                      'trained_test': tuned_test, 'selected_epoch': best_epoch}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='outputs/sticker-benchmark-500')
    p.add_argument('--base', default='models/research/chinese-clip-vit-base-patch16')
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--train-layers', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--projection-lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=20261111)
    p.add_argument('--device', default='mps')
    run(p.parse_args())
