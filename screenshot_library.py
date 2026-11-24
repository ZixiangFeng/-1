"""Researcher screenshot intake; inventory evidence is separate from chat learning."""
import base64
import hashlib
import io
import json
import secrets
import statistics
from collections import Counter
from datetime import datetime, timezone, timedelta

from PIL import Image, ImageOps

VERSION = 'storage-only-v3'


def now():
    return datetime.now(timezone.utc).isoformat()


def init(c):
    # Inventory-analysis tables are retained only for historical audit, never read for decisions.
    c.executescript('''
    CREATE TABLE IF NOT EXISTS screenshot_drafts(
        id TEXT PRIMARY KEY,user TEXT NOT NULL REFERENCES users(id),image BLOB NOT NULL,
        boxes TEXT NOT NULL,created TEXT NOT NULL,result TEXT);
    CREATE TABLE IF NOT EXISTS sticker_inventory_features(
        user TEXT NOT NULL,asset TEXT NOT NULL REFERENCES assets(id),payload TEXT NOT NULL,
        model_metadata TEXT NOT NULL,created TEXT NOT NULL,PRIMARY KEY(user,asset));
    CREATE TABLE IF NOT EXISTS sticker_inventory_profiles(
        user TEXT PRIMARY KEY REFERENCES users(id),revision TEXT NOT NULL,payload TEXT NOT NULL,updated TEXT NOT NULL);
    ''')


def require_user(c, uid):
    row = c.execute('SELECT id,label,arm FROM users WHERE id=?', (uid,)).fetchone()
    if not row:
        raise ValueError('请先选择有效的被试账号')
    return dict(row)


def read_image(encoded):
    try:
        raw = base64.b64decode(encoded.split(',')[-1], validate=True)
        if len(raw) > 6_000_000:
            raise ValueError('截图请小于6MB')
        im = Image.open(io.BytesIO(raw))
        if im.format not in ('JPEG', 'PNG', 'WEBP') or getattr(im, 'n_frames', 1) != 1:
            raise ValueError('请使用PNG、JPEG或静态WebP截图')
        if im.width * im.height > 16_000_000:
            raise ValueError('截图像素过大')
        im = ImageOps.exif_transpose(im).convert('RGB')
        im.load()
        return im
    except (OSError, Image.DecompressionBombError, TypeError) as e:
        raise ValueError('无法读取截图') from e


def jpeg(im, edge=768):
    im = im.copy()
    im.thumbnail((edge, edge), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    im.save(out, 'JPEG', quality=85)
    return out.getvalue()


def box_checked(box, size):
    if not isinstance(box, list) or len(box) != 4 or any(type(v) is not int for v in box):
        raise ValueError('裁剪坐标须为四个整数')
    x, y, right, bottom = box
    if not (0 <= x < right <= size[0] and 0 <= y < bottom <= size[1]) or min(right-x, bottom-y) < 16:
        raise ValueError('裁剪范围超出截图或图片过小')
    return box


def bands(values, threshold, gap, minimum):
    spans = []
    for i, v in enumerate(values):
        if v > threshold:
            if spans and i - spans[-1][1] <= gap:
                spans[-1][1] = i + 1
            else:
                spans.append([i, i+1])
    return [s for s in spans if s[1]-s[0] >= minimum]


def detect(im, options=None):
    """Projection on a uniform grid background, with explicit manual-grid fallback."""
    options = options or {}
    region = box_checked(options.get('region', [0, 0, im.width, im.height]), im.size)
    area = im.crop(region)
    cols, rows = options.get('columns', 0), options.get('rows', 0)
    if cols or rows:
        if type(cols) is not int or type(rows) is not int or not (1 <= cols <= 12 and 1 <= rows <= 30):
            raise ValueError('手动网格请同时填写列数1–12和行数1–30')
        xs = [[round(i*area.width/cols), round((i+1)*area.width/cols)] for i in range(cols)]
        ys = [[round(i*area.height/rows), round((i+1)*area.height/rows)] for i in range(rows)]
        method = 'manual-grid'
    else:
        # Work on a bounded thumbnail; never rescale the stored screenshot before cutting.
        thumb = area.copy()
        thumb.thumbnail((1000, 1600))
        pixels = list(thumb.getdata())
        quant = Counter(tuple(v//8*8+4 for v in p) for p in pixels[::3])
        bg = quant.most_common(1)[0][0]
        mask = [max(abs(a-b) for a,b in zip(p,bg)) > 23 for p in pixels]
        xp = [sum(mask[y*thumb.width+x] for y in range(thumb.height))/thumb.height for x in range(thumb.width)]
        yp = [sum(mask[y*thumb.width:(y+1)*thumb.width])/thumb.width for y in range(thumb.height)]
        gap = max(3, round(min(thumb.size)*.009))
        xs = bands(xp, .045, gap, 22)
        ys = bands(yp, .045, gap, 22)
        if not (2 <= len(xs) <= 12 and 1 <= len(ys) <= 30):
            raise ValueError('未识别到清晰网格，请设置图片区域与行列数后重新裁剪')
        xs = [[round(a*area.width/thumb.width), round(b*area.width/thumb.width)] for a,b in xs]
        ys = [[round(a*area.height/thumb.height), round(b*area.height/thumb.height)] for a,b in ys]
        method = 'background-projection'
    typical = statistics.median([b-a for a,b in ys])
    boxes = []
    for row, (top,bottom) in enumerate(ys):
        for col, (left,right) in enumerate(xs):
            partial = (bottom-top < typical*.78 or (row == len(ys)-1 and bottom >= area.height-3))
            box = [left+region[0], top+region[1], right+region[0], bottom+region[1]]
            boxes.append({'index': len(boxes), 'row': row+1, 'column': col+1,
                          'box': box, 'partial': partial, 'selected': not partial})
    return method, boxes


def preview(c, uid, encoded, options=None):
    require_user(c, uid)
    im = read_image(encoded)
    method, boxes = detect(im, options)
    out = io.BytesIO()
    im.save(out, 'PNG')
    draft = secrets.token_hex(16)
    c.execute('DELETE FROM screenshot_drafts WHERE created<?', ((datetime.now(timezone.utc)-timedelta(days=1)).isoformat(),))
    c.execute('INSERT INTO screenshot_drafts VALUES(?,?,?,?,?,NULL)',
              (draft, uid, out.getvalue(), json.dumps(boxes), now()))
    return {'draft': draft, 'user': uid, 'width': im.width, 'height': im.height, 'method': method,
            'tiles': [{**b, 'image': base64.b64encode(jpeg(im.crop(b['box']), 240)).decode()} for b in boxes]}


def commit(c, uid, draft, chosen, event):
    require_user(c, uid)
    record = c.execute('SELECT * FROM screenshot_drafts WHERE id=? AND user=?', (draft,uid)).fetchone()
    if not record:
        raise ValueError('预览已过期或不属于该被试，请重新裁剪')
    if record['result']:
        return {**json.loads(record['result']), 'replayed': True}
    if not isinstance(chosen,list) or not 1 <= len(chosen) <= 360:
        raise ValueError('请选择要导入的图片')
    im = Image.open(io.BytesIO(record['image']))
    originals = {b['index']: b for b in json.loads(record['boxes'])}
    used = set()
    prepared = []
    for item in chosen:
        index = item.get('index')
        if type(index) is not int or index not in originals or index in used:
            raise ValueError('裁剪编号重复或无效')
        used.add(index)
        box = box_checked(item.get('box', originals[index]['box']), im.size)
        data = jpeg(im.crop(box))
        aid = hashlib.sha256(uid.encode()+data).hexdigest()[:24]
        prepared.append((aid,data,index,box))
    inserted, duplicates, ids = 0,0,[]
    for aid,data,index,box in prepared:
        ids.append(aid)
        if c.execute('SELECT 1 FROM assets WHERE id=? AND user=?',(aid,uid)).fetchone():
            duplicates += 1
            continue
        metadata = {'library_import': True, 'imported': now(), 'source': 'researcher_screenshot',
                    'draft':draft,'tile':index,'box':box,'screenshot_size':list(im.size)}
        c.execute('INSERT INTO assets VALUES(?,?,?,?,?)',(aid,uid,base64.b64encode(data).decode(),'',json.dumps(metadata)))
        inserted += 1
    result = {'user':uid,'imported':inserted,'duplicates':duplicates,'asset_ids':ids}
    c.execute('UPDATE screenshot_drafts SET result=? WHERE id=? AND user=?',(json.dumps(result),draft,uid))
    event(c,uid,'screenshot_library_import',dict(result,draft=draft))
    return result


def inventory(c, uid):
    """Imported images have no semantic status until real conversations exist."""
    import expression_usage
    import personal_divergence
    participant = require_user(c,uid)
    decisions = expression_usage.divergence_decisions(c,uid)
    items = []
    for r in c.execute("SELECT id,image FROM assets WHERE user=? ORDER BY rowid",(uid,)).fetchall():
        item = dict(r)
        count = c.execute("SELECT count(*) FROM messages WHERE user=? AND role='user' AND asset=?",(uid,item['id'])).fetchone()[0]
        item.update(used=bool(count),use_count=count,feature=None,
                    usages=expression_usage.examples(c,uid,item['id']),decisions=decisions.get(item['id'],[]))
        items.append(item)
    candidate_ids = {d['asset_id'] for ds in decisions.values() for d in ds}
    candidates = [dict(r) for r in c.execute('SELECT id,image,enabled FROM generic_stickers') if r['id'] in candidate_ids]
    return {'participant':participant,'items':items,'candidates':candidates,'profile':None,'pending':0,
            'divergence':personal_divergence.report(c,uid),
            'policy':expression_usage.POLICY,'used_assets':sum(x['used'] for x in items),
            'total_uses':sum(x['use_count'] for x in items)}


def analyze_step(c, uid, generate, event):
    # Old clients can still call this endpoint, but cannot trigger semantic inference.
    require_user(c,uid)
    return {'processed':0,'remaining':0,'profile':None,'policy':'storage-only',
            'message':'导入只保存图片；含义与趋异参照在实际聊天使用后结合上下文生成。'}


def context(c, uid, current_asset=None):
    # Legacy callers must never receive retired inventory classifications/mappings.
    import expression_usage
    return expression_usage.context(c,uid,current_asset)
