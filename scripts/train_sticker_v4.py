"""Train a local visual-to-meaning adapter; select by cross-style retrieval.

Frozen v3 visual backbone, newly learned 512x512 projection. No paid API.
Test labels/features are not used in fitting, checkpoint selection or calibration.
"""
import copy, json, random, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from research.sticker_encoder.disentangled import SemanticAdapter, training_objective, retrieval_metrics, normalize, style_eligibility
from research.sticker_encoder.weak_model import save, digest

OUT = ROOT / 'outputs/sticker-divergence-v4-20261119'
SOURCE = ROOT / 'outputs/sticker-semantic-v3-local-20261118'


def align(bundle, key, ids):
    ix = {str(a): i for i, a in enumerate(bundle['ids'])}
    return normalize(np.array([bundle[key][ix[a]] for a in ids]))


def calibrate(vectors, teacher):
    sim, meaning = vectors @ vectors.T, teacher @ teacher.T
    upper = np.triu(np.ones(sim.shape, dtype=bool), 1)
    use = upper & ((meaning >= .72) | (meaning <= .45))
    scores, positive = sim[use], meaning[use] >= .72
    records = []
    for t in np.linspace(.20, .99, 160):
        pred = scores >= t
        tp, fp = int((pred & positive).sum()), int((pred & ~positive).sum())
        fn = int((~pred & positive).sum())
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f05 = 1.25 * precision * recall / (.25 * precision + recall) if precision + recall else 0
        records.append({'threshold': float(t), 'precision': precision, 'recall': recall,
                        'f0_5': f05, 'tp': tp, 'fp': fp, 'fn': fn})
    best = max(records, key=lambda r: (r['f0_5'], r['precision'], r['threshold']))
    return {**best, 'pair_n': len(scores), 'policy': 'Validation-only F0.5 on AI-caption positive>=.72/negative<=.45; gray zone omitted', 'human_calibrated': False}


def main():
    if (OUT / 'checkpoint-lock.json').exists():
        raise ValueError('Locked run exists; do not overwrite')
    started = time.time()
    torch.set_num_threads(4)
    random.seed(20261119); np.random.seed(20261119); torch.manual_seed(20261119)
    data = json.loads((OUT / 'dataset.json').read_text())
    rows = [r for r in data['rows'] if r['split'] != 'test' and r['quality'] != 'uncertain']
    ids = [r['id'] for r in rows]
    feature_file = SOURCE / 'evaluation-features.npz'
    v = np.load(feature_file)
    teacher = np.load(ROOT / 'outputs/sticker-codex-local-20261118/teacher-features.npz')
    styles = np.load(SOURCE / 'all-style-features.npz')
    x, y, s = align(v, 'v3_image', ids), align(teacher, 'meaning', ids), align(styles, 'style', ids)
    train = np.array([i for i, r in enumerate(rows) if r['split'] == 'train'])
    val = np.array([i for i, r in enumerate(rows) if r['split'] == 'validation'])
    if set(rows[i]['pack'] for i in train) & set(rows[i]['pack'] for i in val):
        raise ValueError('Source-pack leakage')
    packs = {p: i for i, p in enumerate(sorted({r['pack'] for r in rows}))}
    pack_tensor = torch.tensor([packs[r['pack']] for r in rows])
    xt, yt, st = [torch.from_numpy(a) for a in (x, y, s)]
    protocol = {'version': 'v4', 'seed': 20261119, 'data_sha256': digest(OUT / 'dataset.json'),
                'backbone': 'frozen v3 Chinese-CLIP image tower', 'trainable': '512x512 projection+bias',
                'teacher': 'frozen local BGE Chinese meaning captions; machine weak supervision',
                'train_n': len(train), 'validation_n': len(val), 'test_n_used_in_training': 0,
                'epochs_max': 120, 'eval_every': 5, 'patience_evaluations': 5,
                'batch_size': 128, 'learning_rate': .0002, 'ridge_alpha': 1., 'pair_weights': [0., .35],
                'selection': 'Validation cross-style multi-positive nDCG@5; tie semantic P@5; earlier tie retained',
                'positive': 'caption cosine>=.78, ALADIN distance>=.65, different source series',
                'hard_negative': 'caption cosine<=.40 and ALADIN distance<=.45; same appearance alone is NEVER a negative',
                'test_policy': 'Historical fixed holdout, already examined in prior work, NOT a fresh blind test',
                'paid_api_calls': 0}
    save(OUT / 'protocol.json', protocol)
    valrows = [rows[i] for i in val]
    base = retrieval_metrics(x[val], y[val], valrows, s[val])
    oldv2 = align(np.load(SOURCE / 'v2-evaluation-features.npz'), 'v2_image', ids)
    old = retrieval_metrics(oldv2[val], y[val], valrows, s[val])
    # Fit initialization using train data only; no style or family enters inference.
    mx, my = x[train].mean(0), y[train].mean(0)
    xc, yc = x[train] - mx, y[train] - my
    w = np.linalg.solve(xc.T @ xc + np.eye(x.shape[1]), xc.T @ yc).astype('float32')
    init = SemanticAdapter()
    with torch.no_grad():
        init.projection.weight.copy_(torch.from_numpy(w.T))
        init.projection.bias.copy_(torch.from_numpy(my - mx @ w))
    initial = copy.deepcopy(init.state_dict())
    best, beststate, trials = None, None, []
    def measure(model):
        with torch.no_grad():
            z = model(xt[val]).numpy()
        m = retrieval_metrics(z, y[val], valrows, s[val])
        return m, (m['cross_style_ndcg_at_5'] or 0., m['semantic_p_at_5'])
    print(json.dumps({'event': 'start', 'protocol': protocol, 'v2_validation': old, 'v3_validation': base}), flush=True)
    for trial, weight in enumerate(protocol['pair_weights']):
        model = SemanticAdapter(); model.load_state_dict(initial)
        opt = torch.optim.AdamW(model.parameters(), lr=protocol['learning_rate'], weight_decay=.01)
        rng = np.random.default_rng(20261119)
        records, stale, local_best = [], 0, None
        for epoch in range(121):
            if epoch:
                order = rng.permutation(train)
                losses = []
                for start in range(0, len(order), 128):
                    batch = order[start:start+128]
                    if len(batch) < 2:
                        continue
                    opt.zero_grad(set_to_none=True)
                    z = model(xt[batch])
                    loss = training_objective(z, yt[batch], st[batch], pack_tensor[batch], weight)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite loss')
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
                    losses.append(loss.item())
            if epoch % 5:
                continue
            metric, key = measure(model)
            record = {'epoch': epoch, 'loss': float(np.mean(losses)) if epoch else None, 'validation': metric}
            records.append(record)
            if best is None or key > tuple(best['selection_key']):
                best = {'trial': trial, 'pair_weight': weight, 'epoch': epoch, 'selection_key': list(key), 'validation': metric}
                beststate = copy.deepcopy(model.state_dict())
            if local_best is None or key > local_best:
                local_best = key; stale = 0
            else:
                stale += 1
            print(json.dumps({'event': 'epoch', 'trial': trial, 'epoch': epoch, 'validation': metric, 'selected_epoch': best['epoch']}), flush=True)
            save(OUT / 'progress.json', {'trial': trial, 'latest': record, 'selected': best})
            if stale >= protocol['patience_evaluations']:
                break
        trials.append({'weight': weight, 'records': records})
    model = SemanticAdapter(); model.load_state_dict(beststate)
    with torch.no_grad():
        z = model(xt).numpy()
    calibration = calibrate(z[val], y[val])
    train_meaning, train_distance = y[train] @ y[train].T, 1 - s[train] @ s[train].T
    values = [train_distance[i, j] for i in range(len(train)) for j in range(i)
              if train_meaning[i, j] >= .72 and style_eligibility(rows[train[i]], rows[train[j]], train_distance[i, j], 0.) is None]
    minimum = max(.65, float(np.quantile(values, .25))) if values else .65
    calibration['style_minimum'] = minimum
    calibration['style_policy'] = 'max(0.65, train-only cross-medium same-meaning distance 25th percentile)'
    calibration['style_train_pair_n'] = len(values)
    from safetensors.torch import save_file
    save_file(beststate, str(OUT / 'semantic-adapter.safetensors'))
    np.savez_compressed(OUT / 'development-features.npz', ids=np.array(ids), trained=z, teacher=y)
    report = {'implementation': 'xiaoyu-semantic-adapter-v4', 'state': 'trained_locked',
              'protocol': protocol, 'selected': best, 'trials': trials, 'calibration': calibration,
              'v2_validation': old, 'v3_validation': base, 'seconds': time.time()-started,
              'trainable_parameters': sum(p.numel() for p in model.parameters()),
              'human_gold_evaluation': False, 'paid_api_calls': 0,
              'limitations': ['AI caption proxy labels, not human semantic truth.',
                             'Style labels are AI weak labels; ALADIN distance alone is not perceived divergence.',
                             'Historical holdout already evaluated before; not a new blind test.',
                             'Frozen image tower plus newly trained adapter, not full backbone retraining.',
                             'No private conversations used; context-specific equivalence not established.']}
    save(OUT / 'training-report.json', report)
    save(OUT / 'checkpoint-lock.json', {'checkpoint_sha256': digest(OUT / 'semantic-adapter.safetensors'),
         'report_sha256': digest(OUT / 'training-report.json'), 'data_sha256': digest(OUT / 'dataset.json'),
         'code_sha256': {str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__), ROOT / 'research/sticker_encoder/disentangled.py']},
         'time': time.time()})
    print(json.dumps({'event': 'locked', 'selected': best, 'calibration': calibration, 'seconds': report['seconds']}), flush=True)


if __name__ == '__main__':
    main()
