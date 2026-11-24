"""Continue v1 with expanded weak data; select on validation, evaluate tests once.

Three predefined relational-loss weights; all start from the identical v1
checkpoint. Epoch zero (the parent) is always an eligible incumbent.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import ChineseCLIPModel, ChineseCLIPProcessor
from research.sticker_encoder.weak_model import contrastive_loss, digest, retrieval, rgb, save
from research.sticker_encoder.continuation import relational_loss, validate_continuation, neighborhood_overlap, paired_bootstrap


def run(args):
    out, corpus, parent = Path(args.output).resolve(), Path(args.source).resolve(), Path(args.parent).resolve()
    if out.exists():
        raise ValueError('Use a new output directory')
    data = json.loads((corpus / 'dataset.json').read_text())
    source = Path(data['source']).resolve()
    old = json.loads((parent / 'dataset.json').read_text())
    prior_report = json.loads((parent / 'training-report.json').read_text())
    validate_continuation(data, old)
    for name, sha in prior_report['checkpoint_files'].items():
        if digest(parent / 'checkpoint' / name) != sha:
            raise ValueError('Parent checkpoint integrity failure')
    for r in data['rows']:
        if digest(source / r['image']) != r['image_sha256'] or digest(source / 'annotations' / (r['id'] + '.json')) != r['annotation_sha256']:
            raise ValueError('Dataset file changed')
    out.mkdir(parents=True)
    shutil.copy2(corpus / 'dataset.json', out / 'dataset.json')
    shutil.copy2(corpus / 'teacher-features.npz', out / 'teacher-features.npz')
    fresh_count = sum(r['split'] == 'test' and r['id'].startswith('E') for r in data['rows'])
    protocol = {'seed': 20261112, 'maximum_epochs_per_trial': 12, 'early_stopping_patience': 3,
                'weights': [0., .2, .5], 'backbone_lr': 3e-6, 'projection_lr': 3e-5,
                'selection': 'highest validation image-to-caption MRR, including parent epoch zero; ties keep earlier incumbent',
                'primary_test': f'{fresh_count} newly collected test images within held-out source packs' if fresh_count else 'Previously evaluated v1 test images; no fresh test images are available',
                'secondary_tests': ['old test images', 'combined test images'] if fresh_count else [],
                'test_access': 'Metrics computed once after all three validation trials and checkpoint selection',
                'supervision': 'Doubao automatic captions + BGE automatic caption neighborhoods; NOT human gold',
                'dataset_sha256': digest(corpus / 'dataset.json'), 'parent_report_sha256': digest(parent / 'training-report.json')}
    save(out / 'protocol.json', protocol)
    started = time.time(); device = torch.device(args.device)
    torch.set_num_threads(4); os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    rows = data['rows']; ids = [r['id'] for r in rows]
    teacher_file = np.load(out / 'teacher-features.npz', allow_pickle=False)
    if list(teacher_file['ids']) != ids:
        raise ValueError('Teacher ID order mismatch')
    teacher = torch.from_numpy(teacher_file['meaning']); teacher_sim = teacher @ teacher.T
    model = ChineseCLIPModel.from_pretrained(parent / 'checkpoint', local_files_only=True, use_safetensors=True).to(device)
    processor = ChineseCLIPProcessor.from_pretrained(parent / 'checkpoint', local_files_only=True)
    names = prior_report['trainable_parameter_names']
    for name, p in model.named_parameters():
        p.requires_grad_(name in names)
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    initial = {name: p.detach().cpu().clone() for name, p in trainable.items()}
    pixels = processor(images=[rgb(source / r['image']) for r in rows], return_tensors='pt')['pixel_values']
    model.eval(); texts = []
    with torch.inference_mode():
        for start in range(0, len(rows), 32):
            inp = processor(text=[r['text'] for r in rows[start:start+32]], padding=True, truncation=True, max_length=64, return_tensors='pt').to(device)
            texts.append(F.normalize(model.get_text_features(**inp), dim=-1).cpu())
    texts = torch.cat(texts).clone()
    captions = {t: i for i, t in enumerate(sorted({r['text'] for r in rows}))}
    caption_ids = [captions[r['text']] for r in rows]
    splits = {s: [i for i, r in enumerate(rows) if r['split'] == s] for s in data['splits']}

    @torch.inference_mode()
    def encode(indices):
        model.eval(); values = []
        for start in range(0, len(indices), 32):
            v = F.normalize(model.get_image_features(pixel_values=pixels[indices[start:start+32]].to(device)), dim=-1)
            if not torch.isfinite(v).all():
                raise ValueError('Nonfinite encoding')
            values.append(v.cpu())
        return torch.cat(values).numpy()

    def restore(state):
        with torch.no_grad():
            for name, p in trainable.items():
                p.copy_(state[name].to(device))

    all_idx = list(range(len(rows))); val = splits['validation']
    baseline = encode(all_idx)
    baseline_val = retrieval(baseline[val], texts[val].numpy())
    best = {'score': baseline_val['image_to_caption']['mrr'], 'epoch': 0, 'weight': None, 'trial': 0}
    best_state = copy.deepcopy(initial)
    trials = []; selected_records = []
    print(json.dumps({'event': 'baseline_validation', 'metrics': baseline_val, 'counts': data['counts']}), flush=True)
    for trial, weight in enumerate(protocol['weights'], 1):
        restore(initial)
        random.seed(protocol['seed']); np.random.seed(protocol['seed']); torch.manual_seed(protocol['seed'])
        optimizer = torch.optim.AdamW([
            {'params': [p for n, p in trainable.items() if not n.startswith('visual_projection')], 'lr': protocol['backbone_lr']},
            {'params': list(model.visual_projection.parameters()), 'lr': protocol['projection_lr']}], weight_decay=.01)
        trial_best, stale, records = baseline_val['image_to_caption']['mrr'], 0, []
        for epoch in range(1, protocol['maximum_epochs_per_trial'] + 1):
            indices = splits['train'].copy(); random.shuffle(indices)
            model.eval()
            for layer in model.vision_model.encoder.layers[-2:]:
                layer.train()
            totals = np.zeros(3); seen = 0
            for start in range(0, len(indices), 32):
                batch = indices[start:start+32]
                if len(batch) < 2:
                    continue
                optimizer.zero_grad(set_to_none=True)
                image = F.normalize(model.get_image_features(pixel_values=pixels[batch].to(device)), dim=-1)
                similarity = teacher_sim[batch][:, batch].to(device)
                clip = contrastive_loss(image, texts[batch].to(device), [caption_ids[i] for i in batch], similarity, model.logit_scale.detach().exp().clamp(max=100))
                relation = relational_loss(image, similarity)
                loss = clip + weight * relation
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.)
                if not torch.isfinite(norm):
                    raise ValueError('Nonfinite gradients')
                optimizer.step()
                totals += np.array([loss.item(), clip.item(), relation.item()]) * len(batch); seen += len(batch)
            v = encode(val); validation = retrieval(v, texts[val].numpy())
            score = validation['image_to_caption']['mrr']
            record = {'epoch': epoch, 'loss': float(totals[0] / seen), 'contrastive_loss': float(totals[1] / seen),
                      'relational_loss': float(totals[2] / seen), 'validation': validation,
                      'validation_weak_neighbor_overlap_at_5': neighborhood_overlap(v, teacher[val].numpy())}
            records.append(record)
            if score > best['score'] + 1e-8:
                best = {'score': score, 'epoch': epoch, 'weight': weight, 'trial': trial}
                best_state = {name: p.detach().cpu().clone() for name, p in trainable.items()}
                selected_records = copy.deepcopy(records)
            if score > trial_best + 1e-8:
                trial_best, stale = score, 0
            else:
                stale += 1
            save(out / 'progress.json', {'state': 'training', 'trial': trial, 'weight': weight, 'epoch': epoch, 'best': best, 'latest': record})
            print(json.dumps({'event': 'epoch', 'trial': trial, 'weight': weight, **record, 'best': best}), flush=True)
            if stale >= protocol['early_stopping_patience']:
                break
        trials.append({'trial': trial, 'weight': weight, 'records': records})
        save(out / 'validation-trials.json', trials)
    restore(best_state)
    tuned = encode(all_idx)
    if best['trial']:
        selected_records = trials[best['trial']-1]['records']
    else:
        selected_records = [{'epoch': 0, 'loss': 0., 'validation': baseline_val}]
    # The first test-score access in this program is below, after selection.
    tests = {}
    for name, ix in [('new_images', [i for i in splits['test'] if rows[i]['id'].startswith('E')]),
                     ('old_images', [i for i in splits['test'] if not rows[i]['id'].startswith('E')]),
                     ('combined', splits['test'])]:
        if not ix:
            continue
        tests[name] = {'ids': [ids[i] for i in ix], 'baseline_v1': retrieval(baseline[ix], texts[ix].numpy()),
                       'continued_v2': retrieval(tuned[ix], texts[ix].numpy()),
                       'weak_neighbor_overlap_at_5': {'baseline_v1': neighborhood_overlap(baseline[ix], teacher[ix].numpy()),
                                                     'continued_v2': neighborhood_overlap(tuned[ix], teacher[ix].numpy())},
                       'paired_uncertainty': paired_bootstrap(baseline[ix], tuned[ix], texts[ix].numpy())}
    model.save_pretrained(out / 'checkpoint', safe_serialization=True)
    processor.save_pretrained(out / 'checkpoint')
    from safetensors.torch import save_file
    save_file({n: p.detach().cpu().contiguous() for n, p in trainable.items()}, str(out / 'visual-finetune.safetensors'))
    np.savez_compressed(out / 'all-features.npz', ids=np.array(ids), baseline=baseline, trained=tuned, text=texts.numpy())
    # Keep the public preview on the original 500-image ALADIN-indexed gallery.
    gallery = [i for i, r in enumerate(rows) if r['id'] in {x['id'] for x in old['rows']}]
    np.savez_compressed(out / 'features.npz', ids=np.array([ids[i] for i in gallery]),
                        baseline=baseline[gallery], trained=tuned[gallery], text=texts[gallery].numpy())
    b, t = tests['combined']['baseline_v1'], tests['combined']['continued_v2']
    report = {**prior_report, 'implementation': 'xiaoyu-chinese-clip-weak-v2', 'base': prior_report['base'],
              'parent_run': str(parent), 'parent_report_sha256': digest(parent / 'training-report.json'),
              'method': 'Continue v1 on available annotated public data with lower learning rates; validation selects among contrastive-only and soft BGE-neighborhood distillation.',
              'negative_mask': prior_report['negative_mask'], 'neighborhood_teacher': 'Frozen BGE meaning descriptions; soft KL over off-diagonal within-batch similarities. Weak teacher, not gold pairs.',
              'settings': vars(args), 'protocol': protocol, 'data_counts': data['counts'],
              'baseline_label': '上一版 v1', 'trained_label': '继续训练 v2', 'gallery_count': len(gallery),
              'selected_epoch': best['epoch'], 'selected_trial': best['trial'], 'selected_relational_weight': best['weight'],
              'epochs': selected_records, 'validation_trials': trials, 'selection_metric': protocol['selection'],
              'baseline_validation': baseline_val, 'baseline_test': b, 'trained_test': t,
              'test_sets': tests, 'improved_on_heldout_caption_mrr': t['image_to_caption']['mrr'] > b['image_to_caption']['mrr'],
              'dataset_sha256': digest(out / 'dataset.json'), 'seconds': round(time.time()-started, 1),
              'code_sha256': {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__), ROOT/'research/sticker_encoder/continuation.py', ROOT/'research/sticker_encoder/weak_model.py', ROOT/'scripts/expand_sticker_training.py']},
              'software': {'torch': torch.__version__, 'transformers': transformers.__version__, 'numpy': np.__version__, 'python': sys.version, 'device': str(device)},
              'checkpoint_files': {p.name: digest(p) for p in (out/'checkpoint').iterdir() if p.is_file()},
              'limitations': ['Own model, not author GSE.', 'Automatic descriptions and BGE neighborhoods can be wrong.',
                  'Caption retrieval and teacher-neighborhood agreement do not measure human same-meaning accuracy.',
                  'Old test images were previously inspected. New test images share held-out source packs with old test.',
                  'Query bootstrap intervals are descriptive and do not account for source-pack clustering.',
                  'No independent human pair labels or contextual/person-specific validation; production chat model unchanged.']}
    save(out / 'training-report.json', report)
    save(out / 'progress.json', {'state': 'completed', 'selected': best, 'seconds': report['seconds']})
    print(json.dumps({'event': 'completed', 'best': best, 'seconds': report['seconds'], 'tests': tests}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='outputs/sticker-corpus-v2-20261112')
    p.add_argument('--parent', default='outputs/sticker-semantic-v1-20261111')
    p.add_argument('--output', default='outputs/sticker-semantic-v2-20261112')
    p.add_argument('--device', default='mps')
    run(p.parse_args())
