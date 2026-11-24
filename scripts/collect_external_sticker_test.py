"""Create pinned, deduplicated study-external holdout before computing any model output."""
import hashlib
import io
import json
import re
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.fft import dctn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/sticker-external-holdout-20261119'
CACHE = ROOT / 'tmp/holdout-nov19'
OLD = ROOT / 'outputs/sticker-broad-v5-20261119/display-dataset.json'
SOURCES = {
    'platform': ('Augenstern-O/Stickers', '91285989c50eaf45560dab02043df1d4b03aded1'),
    'meme': ('9Ashwin/sticker-ext', 'b655381bc075e2591cf32b2a64c034894d3621bf'),
}
PACKS = ['QQ/QQ黄脸', 'Feishu', 'DingTalk', 'RedNote', 'Coolapk/酷小牛', 'BiliBili/小黄脸']
AWKWARD = re.compile('无语|无奈|苦笑|尴尬|假笑|白眼|冷汗|一言难尽|汗颜|无言')
HAPPY = re.compile('开心|高兴|大笑|哈哈|爆笑|喜悦|欢笑|庆祝|太好笑|笑出')
SEED = 'xiaoyu-external-20261119-1'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def get(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent':'Xiaoyu-research-holdout'}), timeout=18) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise


def metadata(key, filename, endpoint):
    path = CACHE / filename
    if not path.exists():
        path.write_bytes(get(endpoint))
    return read(path)


def fingerprints(im):
    rgba = ImageOps.exif_transpose(im).convert('RGBA')
    base = Image.new('RGBA', rgba.size, 'white')
    base.alpha_composite(rgba)
    rgb = base.convert('RGB')
    def ph(image):
        low = dctn(np.asarray(image.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=float), type=2)[:8,:8].ravel()
        return int.from_bytes(np.packbits(low > np.median(low[1:])).tobytes(),'big')
    return rgb, sha(str(rgb.size).encode()+rgb.tobytes()), (ph(rgb),ph(ImageOps.mirror(rgb)))


def groups():
    repos = {}
    for key, filename in [('platform','source-tree.json'),('meme','source2.json')]:
        repo,rev = SOURCES[key]
        tree = metadata(key,filename,f'https://api.github.com/repos/{repo}/git/trees/{rev}?recursive=1')
        if tree['sha'] != rev or tree.get('truncated'):
            raise ValueError('Incomplete upstream tree')
        repos[key] = {e['path']:e for e in tree['tree'] if e['type']=='blob'}
    repo,rev = SOURCES['meme']
    meta = metadata('meme','manifest2.json',f'https://raw.githubusercontent.com/{repo}/{rev}/manifest.json')
    info = {r['filename']:r for r in meta['items']}
    bins = {}
    for key, files in repos.items():
        for name,e in files.items():
            if Path(name).suffix.lower() not in ('.jpg','.jpeg','.png','.webp') or not 200 <= e['size'] <= 1_000_000:
                continue
            pack = name.rsplit('/',1)[0]
            if key=='platform' and pack not in PACKS:
                continue
            if key=='meme' and name not in info:
                continue
            desc = info[name]['caption'] if key=='meme' else re.sub(r'^.*?_|^牛牛','',Path(name).stem)
            if not desc:
                continue
            cut = 'wry' if AWKWARD.search(desc) else 'joy' if HAPPY.search(desc) else 'general'
            group = (key,pack if key=='platform' else '',cut)
            bins.setdefault(group,[]).append(dict(source=key,path=name,git_blob=e['sha'],source_caption=desc,
                                                   source_pack=pack if key=='platform' else '',stratum=cut))
    for group,items in sorted(bins.items()):
        key,pack,cut = group
        quota = ({'wry':2,'joy':2,'general':4} if key=='platform' else {'wry':16,'joy':12,'general':20})[cut]
        items.sort(key=lambda r:sha((SEED+r['source']+r['path']).encode()))
        yield quota,items


def load_image(row):
    repo,rev = SOURCES[row['source']]
    url = 'https://raw.githubusercontent.com/'+repo+'/'+rev+'/'+urllib.parse.quote(row['path'])
    path = CACHE / ('blob-'+row['git_blob'])
    if not path.exists():
        path.write_bytes(get(url))
    raw=path.read_bytes()
    if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() != row['git_blob']:
        raise ValueError('Incorrect source blob')
    with Image.open(io.BytesIO(raw)) as im:
        if getattr(im,'n_frames',1)!=1 or im.width*im.height>12_000_000:
            return None
        rgb,pixels,ph = fingerprints(im)
    return rgb,dict(row,url=url,original_sha256=sha(raw),pixel_hash=pixels),ph


def main():
    CACHE.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    images=OUT/'images'; images.mkdir(exist_ok=True)
    old=read(OLD)['rows']
    seen={r.get(k) for r in old for k in ['file_sha256','pixel_hash','image_sha256','sha']}
    known=[]
    for row in old:
        with Image.open(ROOT/'outputs/sticker-corpus-v2-20261112'/row['image']) as im:
            known.append(fingerprints(im)[2][0])
    old_count=len(known)
    selected=[]; discarded=[]; seen_source_names=set()
    with ThreadPoolExecutor(10) as pool:
        for quota,items in groups():
            n=0; offset=0
            while n<quota and offset<len(items):
                batch=items[offset:offset+max(6,quota-n+4)]; offset+=len(batch)
                for item,result in zip(batch,pool.map(load_image,batch)):
                    if result is None:
                        discarded.append([item['path'],'animated_or_large']); continue
                    rgb,row,pair=result
                    uniq=(row['source_pack'],row['source_caption'])
                    if item['git_blob'] in seen or row['original_sha256'] in seen or row['pixel_hash'] in seen or uniq in seen_source_names:
                        discarded.append([item['path'],'exact_or_repeated_caption']); continue
                    distance=min(min(bin(p^q).count('1') for p in pair) for q in known)
                    if distance<=7:
                        discarded.append([item['path'],'perceptual_duplicate']); continue
                    row['id']='X%03d'%(1+len(selected))
                    row['nearest_phash_bits_at_intake']=distance
                    rgb.thumbnail((640,640)); target=images/(row['id']+'.png'); rgb.save(target)
                    row['image_sha256']=sha(target.read_bytes())
                    selected.append(row);seen.update([row['original_sha256'],row['pixel_hash'],row['git_blob']])
                    known.append(pair[0]);seen_source_names.add(uniq)
                    n+=1
                    if n==quota:break
    write(OUT/'manifest.json',{'seed':SEED,'sources':SOURCES,'checked_project_images':old_count,'rows':selected,
                               'excluded':discarded,'selection':'static only, source/caption-stratified; filtered exact hashes and 64-bit pHash (incl. mirror) <=7'})
    sheets=CACHE/'sheets';sheets.mkdir(exist_ok=True)
    for i in range(0,len(selected),20):
        sheet=Image.new('RGB',(800,4*178),'#edf1f0');draw=ImageDraw.Draw(sheet)
        for j,row in enumerate(selected[i:i+20]):
            with Image.open(images/(row['id']+'.png')) as im:
                im.thumbnail((148,144));x=j%5*160;y=j//5*178
                sheet.paste(im,(x+(160-im.width)//2,y+3+(144-im.height)//2))
                draw.text((x+58,y+150),row['id'],fill='black',stroke_width=0)
        sheet.save(sheets/('blind-%02d.jpg'%i),quality=92)
    print('Accepted',len(selected),'from',len(SOURCES),'sources',flush=True)


if __name__=='__main__': main()
