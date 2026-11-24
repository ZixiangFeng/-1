"""Admin-only, read-only sticker experiment. Never changes chat selection/memory."""
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import secrets
import sqlite3
import subprocess
import threading
from datetime import datetime, timezone
from PIL import Image
import sticker_identity

STYLE_ID = 'aladin-vit-bamfg-64.47-author-v1'
SEMANTIC_ID = 'bge-small-zh-v1.5-meaning-only-v1'
LOCK = threading.Lock()
PROVENANCE = {
    'style': {'name': 'ALADIN-ViT', 'id': STYLE_ID, 'dimensions': 256,
              'version': 'StyleBabel (ECCV 2022) 作者发布版本；不是 ICCV 2021 原始 CNN',
              'repository': 'https://github.com/DanRuta/ALADIN',
              'commit': '3ddd1124bf4a009e649621c680adb03d74a27953',
              'weights_sha256': '310835f8b65349f8900ae7f70c95ee58858537939bdd2c7b86ce36a666b2a91d',
              'code_sha256': '9f6388000c1d40bfb5e5ce7e98f2f6e10e1c99faaa8f76b1e9636ed3a1ab92f1',
              'weights': 'https://github.com/DanRuta/ALADIN/releases/tag/v1.0.0'},
    'gse': {'name': 'GSE / Triple-S', 'available': False,
            'reason': '论文链接返回 HTTP 410 repository_expired；未取得可验证的官方代码和训练权重，未运行 GSE。',
            'paper': 'https://arxiv.org/html/2511.04977v1',
            'repository': 'https://anonymous.4open.science/r/triple-s-6E65/'},
    'semantic_baseline': {'name': 'BGE 中文含义文本余弦相似度（辅助基线，非 GSE）', 'id': SEMANTIC_ID},
    'rule': '先按独立语义条件筛选，再在合格候选中最大化 1 − cos(ALADIN-ViT)。分数不是概率，也不是已验证的趋异效果。'
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def folder(data):
    p = Path(data) / 'style-lab'
    p.mkdir(exist_ok=True, mode=0o700)
    return p


def cache(data):
    c = sqlite3.connect(folder(data) / 'features.sqlite3', timeout=30)
    c.row_factory = sqlite3.Row
    c.execute('PRAGMA journal_mode=WAL')
    c.execute('CREATE TABLE IF NOT EXISTS features(hash TEXT,model TEXT,vector TEXT,created TEXT,PRIMARY KEY(hash,model))')
    return c


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def catalog(c, user=''):
    if user and not c.execute('SELECT 1 FROM users WHERE id=?', (user,)).fetchone():
        raise ValueError('被试不存在')
    rows = []
    for r in c.execute('SELECT id,image,semantics FROM generic_stickers WHERE enabled=1 ORDER BY id'):
        sem = json.loads(r['semantics'])
        meaning = sem.get('meaning', '')
        rows.append({'id': r['id'], 'scope': 'public', 'image': r['image'], 'meaning': meaning,
                     'meaning_source': '通用视觉分析推测，需人工核对', 'owner': ''})
    if user:
        for r in c.execute('SELECT id,image FROM assets WHERE user=? ORDER BY id', (user,)):
            evidence = c.execute("SELECT interpretation FROM messages WHERE user=? AND role='user' AND asset=? AND interpretation!='' ORDER BY rowid DESC LIMIT 1", (user, r['id'])).fetchone()
            rows.append({'id': r['id'], 'scope': 'personal', 'image': r['image'], 'owner': user,
                         'meaning': evidence[0] if evidence else '',
                         'meaning_source': '实际使用时的含义推测，需核对语境' if evidence else '仅导入，尚无实际使用含义'})
    for row in rows:
        row['key'] = row['scope'] + ':' + row['id']
        row['hash'] = digest(row['image'])
        row['semantic_hash'] = digest(row['meaning']) if row['meaning'] else ''
    return rows


def public_record(row):
    return {k: v for k, v in row.items() if k != 'image'}


def inventory(c, data, user=''):
    rows = catalog(c, user)
    with cache(data) as db:
        available = {r[0] for r in db.execute('SELECT hash FROM features WHERE model=?', (STYLE_ID,))}
    return {'items': [dict(public_record(r), indexed=r['hash'] in available) for r in rows],
            'provenance': PROVENANCE, 'status': status(data)}


def status(data):
    p = folder(data) / 'status.json'
    result = json.loads(p.read_text()) if p.exists() else {'state': 'idle', 'done': 0, 'total': 0}
    if result.get('state') in ('running', 'starting') and not LOCK.locked():
        # A process restart must not leave an apparently permanent job.
        result = dict(result, state='interrupted', error='服务已重启；可继续建立索引，已有特征保留。')
    return result


def save_status(data, value):
    p = folder(data) / 'status.json'
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(value, updated=utc()), ensure_ascii=False))
    os.replace(tmp, p)


def start(c, root, data, user=''):
    root, data = Path(root), Path(data)
    ckpt = root / 'models/style-lab/aladinVIT_bamfg_64.47.pt'
    python = root / '.venv-style/bin/python'
    if not ckpt.exists() or not python.exists():
        raise ValueError('ALADIN-ViT 权重或独立运行环境尚未安装，不能生成风格分数。')
    if not LOCK.acquire(blocking=False):
        return status(data)
    try:
        rows = catalog(c, user)
        if not rows:
            raise ValueError('图库为空，请先导入表情包。')
        if len(rows) > 5000:
            raise ValueError('本次索引超过 5000 张，请分批处理。')
        job = secrets.token_hex(8)
        snapshot = folder(data) / (job + '.json')
        snapshot.write_text(json.dumps(rows, ensure_ascii=False))
        snapshot.chmod(0o600)
        initial = {'state': 'starting', 'job': job, 'done': 0, 'total': len(rows), 'started': utc()}
        save_status(data, initial)
    except Exception:
        LOCK.release()
        raise

    def work():
        log = folder(data) / 'worker.log'
        try:
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
            with log.open('w') as out:
                # Separate CPU process; no model is loaded into the chat process.
                subprocess.run([str(python), str(root / 'scripts/style_lab_worker.py'), '--root', str(root),
                                '--data', str(data), '--snapshot', str(snapshot), '--job', job],
                               env=env, stdout=out, stderr=out, check=True, timeout=7200)
                # Existing local BGE model is an explicitly named optional baseline.
                result = subprocess.run([str(root / '.venv/bin/python'), str(root / 'scripts/style_lab_worker.py'),
                                         '--root', str(root), '--data', str(data), '--snapshot', str(snapshot),
                                         '--job', job, '--semantic'], env=env, stdout=out, stderr=out, timeout=1800)
            current = json.loads((folder(data) / 'status.json').read_text())
            save_status(data, dict(current, state='ready', semantic_ready=result.returncode == 0))
        except Exception as exc:
            save_status(data, dict(initial, state='failed', error='索引任务未完成，可重试；已有特征保留。', error_type=type(exc).__name__))
        finally:
            snapshot.unlink(missing_ok=True)
            LOCK.release()
    threading.Thread(target=work, daemon=True).start()
    return initial


def cosine(a, b):
    if len(a) != len(b) or not a or any(not math.isfinite(float(x)) for x in (*a, *b)):
        raise ValueError('特征维度或数值无效')
    norm = math.sqrt(sum(x*x for x in a)*sum(x*x for x in b))
    if norm == 0:
        raise ValueError('零向量不能计算相似度')
    return max(-1.0, min(1.0, sum(x*y for x, y in zip(a, b))/norm))


def rank(query_style, candidates, mode, threshold, approved):
    """Hard semantic gate FIRST. Never relax it to fill a result list."""
    if mode not in ('manual', 'bge', 'style_only', 'gse_reproduction'):
        raise ValueError('GSE 尚未取得官方权重；请选择已提供的测试模式。')
    if not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError('语义阈值必须在 -1 到 1 之间')
    result = []
    for item in candidates:
        semantic = item.get('semantic_score')
        if mode == 'manual' and item['key'] not in approved:
            continue
        if mode in ('bge', 'gse_reproduction') and (semantic is None or semantic < threshold):
            continue
        distance = 1 - cosine(query_style, item['style_vector'])
        result.append({**{k: v for k, v in item.items() if k != 'style_vector'}, 'style_distance': distance,
                       'style_similarity': 1-distance})
    return sorted(result, key=lambda x: (-x['style_distance'], x['key']))


def search(c, root, data, body, encode=None):
    user = str(body.get('user', ''))
    mode = str(body.get('mode', 'manual'))
    threshold = float(body.get('threshold', .75))
    approved = body.get('approved', [])
    if not isinstance(approved, list) or len(approved) > 5000 or any(not isinstance(x, str) for x in approved):
        raise ValueError('候选列表无效')
    rows = catalog(c, user)
    by_key = {r['key']: r for r in rows}
    query = by_key.get(body.get('query'))
    if not query:
        raise ValueError('请选择当前图库中的查询图片')
    with cache(data) as db:
        features = {(r['hash'], r['model']): json.loads(r['vector']) for r in db.execute('SELECT * FROM features')}
    style = features.get((query['hash'], STYLE_ID))
    if not style:
        raise ValueError('查询图片尚无 ALADIN-ViT 特征，请先建立索引。')
    query_text = str(body.get('meaning', '')).strip()[:1000] or query['meaning']
    sem_vec = None
    reproduction = None
    if mode == 'gse_reproduction':
        reproduction = reproduction_info(root)
        if not reproduction['available']:
            raise ValueError('自行复现程序已接入，尚需人工标注、训练和独立测试；当前没有已训练的语义权重。')
        threshold = reproduction['threshold']
        sem_vec = features.get((query['hash'], reproduction['id']))
        if sem_vec is None:
            raise ValueError('查询图片尚未建立自行复现语义模型的特征索引。')
    if mode == 'bge':
        if not query_text:
            raise ValueError('没有查询语义。请填写本次表达的意思，或选择人工同义候选模式。')
        sem_vec = features.get((digest(query_text), SEMANTIC_ID))
        if sem_vec is None and encode:
            sem_vec = encode([query_text])[0]
        if sem_vec is None:
            raise ValueError('辅助语义模型不可用；未自动改用风格排序。')
    excluded = sticker_identity.exclusions(c, user) if user else {}
    query_identity = sticker_identity.fingerprint(query['image'])
    candidates, counts = [], {'public_total': 0, 'excluded_owned_or_duplicate': 0, 'missing_style': 0, 'missing_semantics': 0}
    for row in rows:
        if row['scope'] != 'public':
            continue
        counts['public_total'] += 1
        if row['key'] == query['key'] or row['id'] in excluded or sticker_identity.same_image(query_identity, sticker_identity.fingerprint(row['image'])):
            counts['excluded_owned_or_duplicate'] += 1
            continue
        vector = features.get((row['hash'], STYLE_ID))
        if vector is None:
            if mode=='manual' and row['key'] in approved:
                raise ValueError('核准的同义组内有素材缺少风格向量，请完成索引再比较整组。')
            counts['missing_style'] += 1
            continue
        sem = features.get((row['hash'], reproduction['id'])) if reproduction else features.get((row['semantic_hash'], SEMANTIC_ID))
        if mode in ('bge', 'gse_reproduction') and sem is None:
            counts['missing_semantics'] += 1
        candidates.append(dict(public_record(row), style_vector=vector, semantic_score=cosine(sem_vec, sem) if sem_vec is not None and sem is not None else None))
    ranked = rank(style, candidates, mode, threshold, set(approved))
    result = {'query': public_record(query), 'query_meaning': query_text, 'mode': mode, 'threshold': threshold,
              'approved': approved if mode == 'manual' else [], 'counts': dict(counts, eligible=len(ranked)),
              'results': ranked[:100], 'maximum': ranked[0] if ranked else None,
              'provenance': dict(PROVENANCE, reproduction=reproduction), 'created': utc(),
              'note': '仅在当前启用且已索引的公共图库与当前筛选条件内取最大值；人工模式为研究者判定，BGE 模式为未校准辅助基线。'}
    # Audit stays separate from participant events, messages and personal memories.
    p = folder(data) / 'runs'
    p.mkdir(exist_ok=True, mode=0o700)
    (p / (secrets.token_hex(12)+'.json')).write_text(json.dumps(result, ensure_ascii=False))
    return result


def thumbnail(c, user, key):
    row = next((r for r in catalog(c, user) if r['key'] == key), None)
    if not row:
        raise ValueError('图片不存在或不在所选图库')
    with Image.open(io.BytesIO(base64.b64decode(row['image']))) as image:
        image = image.convert('RGBA')
        bg = Image.new('RGBA', image.size, 'white')
        bg.alpha_composite(image)
        image = bg.convert('RGB')
        image.thumbnail((240, 240))
        out = io.BytesIO()
        image.save(out, 'JPEG', quality=82)
        return out.getvalue()


def reproduction_info(root):
    from semantic_training import IMPLEMENTATION
    path=Path(root)/'models/sticker-encoder/reproduction.json'
    if not path.exists():
        return {'available':False,'state':'awaiting_reviewed_training_data','name':'小屿 CLIP 表情语义编码器（GSE 方法自行复现）'}
    report=json.loads(path.read_text())
    if report.get('implementation')!=IMPLEMENTATION or report.get('state')!='trained_evaluated':
        raise ValueError('自行复现模型尚未完成训练与独立测试')
    threshold=report['validation']['threshold']
    if not math.isfinite(threshold):raise ValueError('无效语义阈值')
    return {'available':True,'state':'trained_evaluated','name':'小屿 CLIP 表情语义编码器（GSE 方法自行复现）',
            'id':IMPLEMENTATION+'-'+digest(path.read_text())[:16], 'threshold':threshold,
            'validation':report['validation'],'test':report['test'],'author_original':False}


def training_material(c, root):
    """Only the researcher's public library. No participant images/chat records."""
    import zipfile
    out=io.BytesIO(); records=[]
    rows=catalog(c)
    if len(rows)>5000: raise ValueError('请分批导出公共素材')
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for index,row in enumerate(rows):
            filename='images/'+str(index+1).zfill(5)+'.png'
            with Image.open(io.BytesIO(base64.b64decode(row['image']))) as im:
                buf=io.BytesIO();im.convert('RGBA').save(buf,'PNG')
            z.writestr(filename,buf.getvalue())
            records.append({'id':row['id'],'image':filename,'text':'','machine_hint':row['meaning'],
                            'reviewed':False,'family':'','split':''})
        z.writestr('captions-to-review.jsonl','\n'.join(json.dumps(r,ensure_ascii=False) for r in records))
        z.writestr('pairs-to-review.example.jsonl',json.dumps({'left':'images/00001.png','right':'images/00002.png','label':None,'reviewed':False}))
        z.writestr('README.txt',(Path(root)/'research/sticker_encoder/README.txt').read_text())
    return out.getvalue()
