"""Freeze a second, source-stratified holdout disjoint from the 1,250 gallery and prior 77."""
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import collect_external_sticker_test as source

OUT = ROOT / 'outputs/sticker-pragmatic-v7'
SEED = 'xiaoyu-77-followup-unseen-v7-nov19'
STRATA = {
    'emotion': re.compile('笑|哭|无语|无奈|伤心|委屈|安慰|惊|开心|紧张|生气|怕|尴尬|烦恼'),
    'exchange': re.compile('懂|收到|听话|谢谢|谢|赞|鼓励|吃|给|招呼|问候|早安|晚安|告别|同意|拒绝|支持|抱抱|加油'),
}


def main():
    output = OUT/'holdout'
    manifest = output/'manifest.json'
    if manifest.exists():
        print('Existing frozen sample:', len(source.read(manifest)['rows']))
        return
    (output/'images').mkdir(parents=True, exist_ok=True)
    gallery = source.read(source.OLD)['rows']
    old = source.read(source.OUT/'manifest.json')['rows']
    hashes = set()
    captions = {(r['source_pack'], r['source_caption']) for r in old}
    phashes = []
    for root, rows, path_of in [
        (ROOT/'outputs/sticker-corpus-v2-20261112', gallery, lambda r:r['image']),
        (source.OUT/'images', old, lambda r:r['id']+'.png')]:
        for r in rows:
            hashes.update(r.get(k) for k in ('sha','git_blob','original_sha256','image_sha256','pixel_hash'))
            with Image.open(root/path_of(r)) as im:
                phashes.append(source.fingerprints(im)[2][0])
    bins = {}
    for key, tree_name in [('meme','source2.json'), ('platform','source-tree.json')]:
        repo,rev = source.SOURCES[key]
        tree = source.read(source.CACHE/tree_name)
        assert tree['sha']==rev
        meta = {r['filename']:r['caption'] for r in source.read(source.CACHE/'manifest2.json')['items']}
        for entry in tree['tree']:
            path = entry['path']
            if (entry['type']!='blob' or Path(path).suffix.lower() not in {'.png','.jpg','.jpeg','.webp'}
                    or not 200 <= entry['size'] <= 1_000_000 or entry['sha'] in hashes):
                continue
            pack = path.rsplit('/',1)[0] if key=='platform' else ''
            if key=='platform' and pack.startswith('Images'):
                continue
            caption = re.sub(r'^.*?_', '', Path(path).stem) if key=='platform' else meta.get(path)
            if not caption or (pack,caption) in captions:
                continue
            stratum = next((name for name, pattern in STRATA.items() if pattern.search(caption)), 'general')
            bins.setdefault((key,stratum),[]).append(dict(source=key,path=path,git_blob=entry['sha'],
                 source_caption=caption,source_pack=pack,stratum=stratum))
    rows = []
    with ThreadPoolExecutor(8) as pool:
        for key, choices in sorted(bins.items()):
            choices.sort(key=lambda r:(r['source_pack'] in source.PACKS, source.sha((SEED+r['path']).encode())))
            count = 0
            for start in range(0,len(choices),10):
                if count==8:
                    break
                for result in pool.map(source.load_image, choices[start:start+10]):
                    if result is None:
                        continue
                    im,r,ph = result
                    nearest = min(min(bin(q^p).count('1') for q in ph) for p in phashes)
                    if (nearest<=7 or r['pixel_hash'] in hashes or r['original_sha256'] in hashes or
                            (r['source_pack'],r['source_caption']) in captions):
                        continue
                    identifier = 'Y%03d'%(1+len(rows))
                    path=output/'images'/(identifier+'.png')
                    im.thumbnail((640,640))
                    im.save(path)
                    rows.append(dict(r,id=identifier,image_sha256=source.sha(path.read_bytes()),nearest_phash_bits=nearest))
                    hashes.update((r['pixel_hash'],r['original_sha256'],r['git_blob']))
                    phashes.append(ph[0])
                    captions.add((r['source_pack'],r['source_caption']))
                    count+=1
                    if count==8:
                        break
            if count!=8:
                raise RuntimeError(f'Incomplete stratum {key}: {count}')
    source.write(manifest,dict(seed=SEED,sources=source.SOURCES,prior_gallery=len(gallery),previous_test=len(old),
              design='Two sources × three caption strata × eight, previously unseen platform packs first; not a prevalence sample.',
              rows=rows))
    print('Frozen',len(rows),source.sha(manifest.read_bytes()))


if __name__=='__main__':
    main()
