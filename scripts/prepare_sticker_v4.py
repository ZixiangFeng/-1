"""Offline preparation: existing public weak descriptions and local style inference."""
import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from research.sticker_encoder.weak_model import save, digest

OUT = ROOT / 'outputs/sticker-divergence-v4-20261119'
SOURCE = ROOT / 'outputs/sticker-semantic-v3-local-20261118'
PROMPTS = {
 'photo': ['真实人物的照片，摄影或影视截图。', '真实动物的摄影照片。', '真人照片配上文字的表情包。'],
 'line': ['黑白简笔线条小人的表情包，白色背景。', '黑白线稿漫画表情包。', '黑白熊猫头或蘑菇头的简笔表情包。'],
 'cartoon': ['彩色二维卡通人物动画表情包。', '彩色手绘动物的表情包。', '日式动漫人物截图。'],
 'three_d': ['三维立体卡通角色的表情包。', '毛绒玩偶或者黏土玩偶的照片。', '三维渲染的可爱动物。'],
 'typography': ['大段文字组成的表情包，只有文字没有人物。', '文字主导的标语表情包。', '没有人物，纯文字的截图。'],
}


def extract_families(text):
    groups = {'laugh_face_039': ['姚明'], 'panda_meme': ['熊猫头', '金馆长熊猫'],
              'mushroom_head': ['蘑菇头'], 'tom_jerry': ['猫和老鼠', '汤姆猫', '杰瑞'],
              'happy_duck': ['开心鸭'], 'yellow_funny_face': ['滑稽大佬'],
              'doge': ['Doge', 'doge'], 'peppa': ['小猪佩奇']}
    return [key for key, terms in groups.items() if any(t in text for t in terms)]


def explicit_medium(style):
    if not style:
        return None
    photo = bool(re.search('真人|实拍|照片|影视|电视截图|现场照|自拍', style))
    line = bool(re.search('简笔|线稿|黑白.*(小人|漫画|线条|熊猫|蘑菇)|黑白手绘', style))
    if photo and line:
        return 'unknown'
    if re.search('毛绒|玩偶|三维|3D|3d|立体卡通', style):
        return 'three_d'
    if line:
        return 'line'
    if photo:
        return 'photo'
    if re.search('卡通|动漫|动画|手绘|漫画', style):
        return 'cartoon'
    if re.search('纯文字|文字为主|网页截图|文字海报', style):
        return 'typography'
    return None


def main():
    OUT.mkdir(exist_ok=True)
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(20261119)
    data = json.loads((SOURCE / 'full-dataset.json').read_text())
    old = {r['id']: r for r in json.loads((ROOT / 'outputs/sticker-benchmark-500/results.json').read_text())['items']}
    features = np.load(SOURCE / 'all-style-features.npz')
    indices = {str(a): i for i, a in enumerate(features['ids'])}
    categories = list(PROMPTS)
    labels = [explicit_medium(old.get(r['id'], {}).get('style', '')) for r in data['rows']]
    train = [i for i, r in enumerate(data['rows']) if r['split'] == 'train' and labels[i] in categories]
    val = [i for i, r in enumerate(data['rows']) if r['split'] == 'validation' and labels[i] in categories]
    x = torch.tensor(np.array([features['style'][indices[r['id']]] for r in data['rows']]), dtype=torch.float32)
    y = torch.tensor([categories.index(t) if t in categories else -1 for t in labels])
    model = torch.nn.Linear(256, len(categories))
    opt = torch.optim.AdamW(model.parameters(), lr=.025, weight_decay=.05)
    counts = torch.bincount(y[train], minlength=len(categories)).clamp(min=1)
    weight = (len(train) / counts.float()).sqrt(); weight /= weight.mean()
    best, beststate, bestepoch = -1., None, None
    for epoch in range(1, 501):
        opt.zero_grad(); loss = torch.nn.functional.cross_entropy(model(x[train]), y[train], weight=weight)
        loss.backward(); opt.step()
        if epoch % 20 == 0:
            with torch.no_grad():
                pred = model(x[val]).argmax(1)
                score = np.mean([(pred[y[val] == c] == c).float().mean().item() for c in range(len(categories)) if (y[val] == c).any()])
            if score > best:
                best, bestepoch = score, epoch
                beststate = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(beststate)
    with torch.no_grad():
        probs = model(x).softmax(1).numpy()
    from safetensors.torch import save_file
    save_file(beststate, str(OUT / 'medium-classifier.safetensors'))
    save(OUT / 'medium-training-report.json', {'input': 'frozen ALADIN-ViT 256d', 'train_n': len(train), 'validation_n': len(val),
         'selected_epoch': bestepoch, 'validation_macro_recall_weak': best,
         'confidence_minimum': .60, 'categories': categories, 'human_gold': False,
         'training_ids': [data['rows'][i]['id'] for i in train], 'validation_ids': [data['rows'][i]['id'] for i in val]})
    rows = []
    for i, r in enumerate(data['rows']):
        style = old.get(r['id'], {}).get('style', '')
        p = probs[i]
        order = np.argsort(-p)
        margin = float(p[order[0]] - p[order[1]])
        automatic = categories[order[0]] if p[order[0]] >= .60 else 'unknown'
        explicit = explicit_medium(style)
        medium = explicit if explicit is not None else automatic
        rows.append({**r, 'meaning': r['text'], 'style_description': style,
                     'medium': medium, 'medium_evidence': 'archived_visual_caption' if explicit is not None else 'local_aladin_medium_classifier',
                     'medium_margin': margin, 'families': extract_families(r['path'] + ' ' + style),
                     'style_human_reviewed': False})
    save(OUT / 'dataset.json', {'rows': rows, 'source_sha256': digest(SOURCE / 'full-dataset.json'),
         'split_policy': 'Keep prior source-pack splits. Old test has been evaluated in earlier versions; not fresh blind test.',
         'style_policy': 'Archived AI visual descriptions when explicit; otherwise train-only ALADIN medium classifier, unknown confidence<0.60. All machine labels.'})
    # Only development descriptions are prepared before the new checkpoint lock.
    dev = [r for r in rows if r['split'] != 'test']
    save(OUT / 'development-dataset.json', {'rows': dev})
    print(json.dumps({'rows': len(rows), 'medium_counts': {k: sum(r['medium'] == k for r in rows) for k in [*PROMPTS, 'unknown']}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
