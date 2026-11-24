"""Collect a source-independent fixed sample for the frozen 8- vs 27-family comparison."""
import csv
import hashlib
import io
import json
import re
import sys
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageChops, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_external_sticker_test import fingerprints, read, write, sha, get

RUN = ROOT/'outputs/stickerqueries-100-20261119'
HOLDOUT = RUN/'holdout'
CACHE = ROOT/'tmp/stickerqueries-nov19'
TRAINING = ROOT/'outputs/sticker-pragmatic-v7/training.json'
CORPUS = ROOT/'outputs/sticker-broad-v5-20261119/display-dataset.json'
V7 = ROOT/'research/sticker_encoder/v7'
REV = 'c28a358226990ef183703bb3797c2b4c5b59e73c'
REPO = 'metchee/sticker-queries'
BASE = 'https://huggingface.co/datasets/'+REPO+'/resolve/'+REV+'/'
SEED = 'xiaoyu-externally-annotated-100-post-eight-family-20261119-round3'
N = 100


def norm(text):
    return ''.join(c for c in unicodedata.normalize('NFKC',text).lower() if c.isalnum())


def stored(path,data):
    if path.exists() and read(path)!=data:
        raise RuntimeError('Frozen file differs: '+path.name)
    if not path.exists():
        write(path,data)


def cached(url,path,expected_bytes=None):
    if not path.exists():
        data=get(url)
        if expected_bytes and sha(data)!=expected_bytes:
            raise ValueError('Source hash differs: '+path.name)
        path.write_bytes(data)
    data=path.read_bytes()
    if expected_bytes and sha(data)!=expected_bytes:
        raise ValueError('Cached hash differs: '+path.name)
    return data


def hashes(image):
    rgb, pixel, whole = fingerprints(image)
    # Remove near-white exterior gutter as a second screen for differently padded stickers.
    mask=ImageChops.difference(rgb,Image.new('RGB',rgb.size,'white')).convert('L').point(lambda v:255 if v>20 else 0)
    box=mask.getbbox()
    trimmed=fingerprints(rgb.crop(box))[2] if box and box!=(0,0,*rgb.size) else whole
    return rgb,pixel,set(whole+trimmed)


def old_examples():
    gallery=read(CORPUS)['rows']
    historical=ROOT/'outputs/sticker-validation-20261118/fresh-test-corpus'
    x=ROOT/'outputs/sticker-external-holdout-20261119'
    y=ROOT/'outputs/sticker-pragmatic-v7/holdout'
    corpora=[
        (ROOT/'outputs/sticker-corpus-v2-20261112',gallery,True),
        (historical,read(historical/'manifest.json')['items'],True),
        (x,read(x/'manifest.json')['rows'],False),(y,read(y/'manifest.json')['rows'],False)]
    exact, perceptual, texts=set(),set(),set()
    for directory, rows, historical_image in corpora:
        for r in rows:
            image=directory/(r['image'] if historical_image else 'images/'+r['id']+'.png')
            with Image.open(image) as im:
                _,pixel,ph=hashes(im)
            exact.update(v for k,v in r.items() if k in ('pixel_hash','image_sha256','original_sha256','file_sha256','sha','git_blob') and v)
            exact.add(pixel)
            perceptual.update(ph)
            texts.update(norm(r.get(k,'')) for k in ('text','meaning','source_caption'))
    for item in ('training-dialogues.tsv','development-dialogues.tsv'):
        with (V7/item).open() as stream:
            for r in csv.DictReader(stream,delimiter='\t'):
                texts.update(norm(t) for t in r['utterances'].split('|'))
    for _,phrases,_ in read(ROOT/'research/sticker_encoder/v6/dialogues.json'):
        texts.update(norm(t) for t in phrases)
    return exact,perceptual,texts,[len(rows) for _,rows,_ in corpora]


def candidate_image(item):
    info, row = item
    filename=info['rfilename']
    raw=cached(BASE+filename,CACHE/('img-'+Path(filename).name), info.get('lfs',{}).get('sha256'))
    if not info.get('lfs') and hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=info['blobId']:
        raise ValueError('Git blob differs: '+filename)
    with Image.open(io.BytesIO(raw)) as im:
        if im.format!='PNG' or getattr(im,'n_frames',1)!=1 or im.width*im.height>12_000_000:
            return row,None
        rgb,pix,ph=hashes(im)
    return row,(rgb,pix,ph,sha(raw),info)


def collect():
    for directory in [CACHE,RUN,HOLDOUT,HOLDOUT/'images']:
        directory.mkdir(exist_ok=True,parents=True)
    if (HOLDOUT/'manifest.json').exists():
        manifest=read(HOLDOUT/'manifest.json')
        if len(manifest['rows'])!=N or manifest['protocol_sha256']!=sha((RUN/'protocol.json').read_bytes()):
            raise ValueError('Unexpected sample.')
        print('Frozen images',N)
        return
    info=json.loads(cached('https://huggingface.co/api/datasets/'+REPO+'/revision/'+REV+'?blobs=true',CACHE/'upstream.json'))
    if info['sha']!=REV:
        raise ValueError('Upstream revision differs.')
    files={r['rfilename']:r for r in info['siblings']}
    csv_name='sticker_queries_zh_release.csv'
    raw=cached(BASE+csv_name,CACHE/csv_name)
    git_hash=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if git_hash!=files[csv_name]['blobId']:
        raise ValueError('Upstream CSV checksum mismatch')
    source_rows=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    choices=sorted(((files['stickers/'+r['sticker_id']],r) for r in source_rows
                    if r['sticker_id'].endswith('.png') and 'stickers/'+r['sticker_id'] in files),
                   key=lambda e:sha((SEED+'|'+e[1]['sticker_id']).encode()))
    lock=read(TRAINING)
    live=read(ROOT/'outputs/sticker-context-v6/deployment-result.json')
    frozen_files=[ROOT/'research/sticker_encoder/v6/semantic-ranker-v6.npz',V7/'language-intent.npz',
        ROOT/'sticker_intent.py',ROOT/'sticker_pragmatics.py',CORPUS,
        ROOT/'research/sticker_encoder/v6/catalog-reviews.json',ROOT/'research/sticker_encoder/v6/intent-phrases.npz',
        ROOT/'outputs/sticker-divergence-v4-20261119/all-caption-features.npz',
        ROOT/'outputs/sticker-semantic-v3-local-20261118/all-style-features.npz']
    protocol=dict(source=REPO,revision=REV,seed=SEED,requested=N,chinese_records=len(source_rows),png_records=len(choices),
        csv_sha256=sha(raw),model=lock['sha256'],base=lock['base'],pair_floor=lock['pair_floor'],style_floor=.65,
        deployed_release=live['assets'].split('/')[-1],frozen={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in frozen_files},
        selection='Seeded hash order of original filenames. One static, unique image and one distinct unmodified Chinese human query per case, first listed eligible query. Skip literal overlaps with known project labels/training/development texts.',
        deduplication='Raw/pixel SHA and minimum Hamming distance >7 across normal, mirror, and border-trimmed 64-bit perceptual hashes against gallery, historical 75, previous X/Y, and already accepted Z. No search based on model output.',
        annotation='Method-blind and rank-blind; original Chinese use with reference image and one candidate image. Meaning yes if it can express the same speech act, speaker role, target and stance; no for a contradiction or a different identifiable act; otherwise uncertain. An object-only query without an identifiable act is uncertain. Style yes for a clear change in rendering/medium/layout, not simply a different character. Mark an obvious reused template separately.',
        primary='Queries with a visually verified same-use and clearly different-style, non-template top divergent candidate divided by all 100. Also report among returned, coverage, clear mismatches, unclear intent and stage 1.',
        holdout_use='Test only. No training, threshold selection, or gallery edits with this sample.')
    stored(RUN/'protocol.json',protocol)
    exact,seen,used_text,counts=old_examples()
    rows=[]; discard=[]
    with ThreadPoolExecutor(8) as pool:
        for start in range(0,len(choices),12):
            for r,result in pool.map(candidate_image,choices[start:start+12]):
                if not result:
                    discard.append((r['sticker_id'],'not a static PNG'))
                    continue
                rgb,pix,ph,file_sha,f=result
                if {pix,file_sha,f['blobId']} & exact:
                    discard.append((r['sticker_id'],'exact image overlap'));continue
                nearest=min(bin(h^old).count('1') for h in ph for old in seen)
                if nearest<=7:
                    discard.append((r['sticker_id'],'near image overlap'));continue
                options=[t.strip() for t in re.split('[,，]',r['labeled_queries']) if t.strip()]
                chosen=next((t for t in options if re.search(r'[\u3400-\u9fff]',t) and norm(t) not in used_text),None)
                if chosen is None:
                    discard.append((r['sticker_id'],'no new Chinese query'));continue
                identifier='Z%03d'%(len(rows)+1)
                path=HOLDOUT/'images'/(identifier+'.png')
                rgb.thumbnail((640,640));rgb.save(path)
                rows.append(dict(id=identifier,source_id=r['sticker_id'],source_pack='',source_caption=chosen,
                            source_query_position=options.index(chosen)+1,original_sha256=file_sha,pixel_hash=pix,
                            image_sha256=sha(path.read_bytes()),nearest_phash_bits=nearest,url=BASE+f['rfilename']))
                exact.update((pix,file_sha,f['blobId']));seen.update(ph);used_text.add(norm(chosen))
                if len(rows)==N:break
            if len(rows)==N:break
            print('New images',len(rows),flush=True)
    if len(rows)!=N:
        raise RuntimeError(f'Found only {len(rows)} eligible queries.')
    stored(HOLDOUT/'manifest.json',dict(rows=rows,prior_image_records=counts,discarded=discard,
           protocol_sha256=sha((RUN/'protocol.json').read_bytes())))
    print('Frozen images',len(rows),'hash',sha((HOLDOUT/'manifest.json').read_bytes()),flush=True)


if __name__=='__main__':
    collect()
