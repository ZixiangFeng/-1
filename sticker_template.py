"""Check reused sticker artwork independently of conversational meaning."""
import base64
import binascii
import io
import re

from PIL import Image, ImageDraw, ImageOps

# Text identifiers are evidence from the visible description or catalog, never
# derived from the participant's semantic interpretation.
ALIASES = (
    ('姚明', 'yaoming'), ('蘑菇头',), ('熊猫头','金馆长熊猫'),
    ('佩奇','peppa pig'), ('哆啦a梦','机器猫'),
    ('汤姆猫','tom and jerry','猫和老鼠'), ('开心鸭',), ('doge','狗头doge'),
    ('滑稽大佬',), ('蜡笔小新','野原新之助'), ('海绵宝宝','spongebob'),
)


def families(description):
    lower = description.casefold()
    return {names[0] for names in ALIASES if any(n in lower for n in names)}


def shared_family(reference, option):
    source = families(reference.get('visible_description',''))
    target = families(option.get('description','') + ' ' + option.get('style',''))
    return bool(source & target)


def medium(description):
    provenance = description.casefold()
    patterns = {'photo':r'真人照片|实拍|摄影|写实照片|影视截图|真人截[图屏]',
                'line':r'简笔|线稿|黑白线[条描]',
                'three_d':r'三维|3d渲染|立体玩偶|毛绒玩偶',
                'type':r'纯文字|文字主导画面'}
    hits = [key for key,pattern in patterns.items() if re.search(pattern,provenance)]
    if len(hits)==1:
        return hits[0]
    if not hits and re.search(r'二维卡通|彩色手绘|动漫截图',provenance):
        return 'cartoon'
    return None


def known_match(reference, option):
    if shared_family(reference, option):
        return True
    a = medium(reference.get('visible_description',''))
    b = medium(option.get('description','')+' '+option.get('style',''))
    return bool(a and a == b)


def comparison_sheet(c, user, references, options):
    items = []
    for ref in references:
        row = c.execute('SELECT image FROM assets WHERE id=? AND user=?',(ref['asset_id'],user)).fetchone()
        if row:
            items.append(('R',ref['asset_id'],row[0]))
    ids = [o['asset_id'] for o in options]
    if ids:
        keys = ','.join('?' for _ in ids)
        public = dict(c.execute(f'SELECT id,image FROM generic_stickers WHERE enabled=1 AND id IN ({keys})',ids))
        items.extend(('C',aid,public[aid]) for aid in ids if aid in public)
    tiles = []
    lookup = {}
    for prefix,aid,encoded in items:
        try:
            raw = base64.b64decode((encoded or '').split(',')[-1],validate=True)
            with Image.open(io.BytesIO(raw)) as source:
                if source.width * source.height > 16_000_000:
                    continue
                thumb = ImageOps.exif_transpose(source)
                thumb.thumbnail((144,130))
                thumb = thumb.convert('RGBA')
        except (binascii.Error, OSError, ValueError, Image.DecompressionBombError):
            continue
        key = f'{prefix}{len(tiles)+1}'
        tiles.append((key,thumb))
        lookup[key] = aid
    if not any(k.startswith('R') for k in lookup) or not any(k.startswith('C') for k in lookup):
        return None, {}
    cols = min(4,len(tiles))
    sheet = Image.new('RGB',(cols*152, ((len(tiles)+cols-1)//cols)*156),'white')
    draw = ImageDraw.Draw(sheet)
    for i,(k,im) in enumerate(tiles):
        x,y = (i%cols)*152,(i//cols)*156
        sheet.paste(im,(x+(152-im.width)//2,y+15+(135-im.height)//2),im)
        draw.text((x+3,y+1),k,fill='black')
    out = io.BytesIO()
    sheet.save(out,format='JPEG',quality=85)
    return base64.b64encode(out.getvalue()).decode(), lookup
