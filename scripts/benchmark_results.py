"""Evaluate all 500 queries using the production semantic gate and style rank."""
import argparse
import base64
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import style_lab as lab
import sticker_identity
from PIL import Image


def main(out):
    started=time.time()
    manifest=json.loads((out/'manifest.json').read_text())
    rows=json.loads((out/'snapshot.json').read_text())
    assert len(rows)==500 and len({r['id'] for r in rows})==500
    with lab.cache(out) as c:
        features={(r['hash'],r['model']):json.loads(r['vector']) for r in c.execute('SELECT * FROM features')}
    style=np.array([features[(r['hash'],lab.STYLE_ID)] for r in rows])
    semantic=np.array([features[(r['semantic_hash'],lab.SEMANTIC_ID)] for r in rows])
    assert style.shape==(500,256) and semantic.shape==(500,512)
    assert np.isfinite(style).all() and np.isfinite(semantic).all()
    semantic/=np.linalg.norm(semantic,axis=1,keepdims=True)
    ss=semantic@semantic.T
    fp=[sticker_identity.fingerprint(r['image']) for r in rows]
    coarse=np.array([list(f['coarse']) for f in fp],dtype=np.float32)
    duplicate_pairs=set()
    for i in range(500):
        close=np.flatnonzero(np.mean(np.abs(coarse-coarse[i]),axis=1)<=8)
        for j in close:
            if j>i and sticker_identity.same_image(fp[i],fp[j]):duplicate_pairs.add((i,int(j)))
    pair_count=0;queries={};maximums=[]
    for i,q in enumerate(rows):
        candidates=[]
        for j,r in enumerate(rows):
            if i==j or (min(i,j),max(i,j)) in duplicate_pairs:continue
            if ss[i,j]<.75:continue
            candidates.append({'id':r['id'],'key':r['key'],'semantic_score':float(ss[i,j]),'style_vector':style[j].tolist()})
        ranked=lab.rank(style[i].tolist(),candidates,'bge',.75,set())
        semantic_order=sorted(ranked,key=lambda r:(-r['semantic_score'],r['id']))
        assert {r['id'] for r in ranked}=={r['id'] for r in semantic_order}
        assert all(r['semantic_score']>=.75 for r in ranked)
        assert all(ranked[j]['style_distance']>=ranked[j+1]['style_distance'] for j in range(len(ranked)-1))
        queries[q['id']]={'stage1_ids':[r['id'] for r in semantic_order],
                         'stage2':[{k:r[k] for k in ('id','semantic_score','style_distance','style_similarity')} for r in ranked]}
        pair_count+=len(ranked)
        if ranked:maximums.append({'query':q['id'],**queries[q['id']]['stage2'][0]})
    np.savez_compressed(out/'features.npz',style=style,semantic=semantic,ids=np.array([r['id'] for r in rows]))
    (out/'thumbnails').mkdir(exist_ok=True)
    items=[]
    for m,r in zip(manifest['items'],rows):
        a=json.loads((out/'annotations'/(r['id']+'.json')).read_text())
        with Image.open(out/m['image']) as image:
            image=image.convert('RGB');image.thumbnail((240,240));buf=io.BytesIO();image.save(buf,'JPEG',quality=80)
            thumb=buf.getvalue();(out/'thumbnails'/(r['id']+'.jpg')).write_bytes(thumb)
        items.append({'id':r['id'],'meaning':r['meaning'],'ocr':a['ocr'],'situation':a['situation'],'ambiguity':a['ambiguity'],
                      'style':a['style'],'source':m['url'],'pack':m['pack'],
                      'thumbnail':'data:image/jpeg;base64,'+base64.b64encode(thumb).decode()})
    counts=[len(v['stage2']) for v in queries.values()]
    sensitivity=[]
    for threshold in [.75,.8,.85,.9]:
        ns=[sum(r['semantic_score']>=threshold for r in v['stage2']) for v in queries.values()]
        sensitivity.append({'threshold':threshold,'queries_with_candidates':sum(n>0 for n in ns),'directed_pairs':sum(ns),'unique_pairs':sum(ns)//2,'median_candidates':float(np.median(ns)),'max_candidates':max(ns)})
    # Each image batch shares one API request. Do not count usage five times.
    requests={}
    for m in manifest['items']:
        a=json.loads((out/'annotations'/(m['id']+'.json')).read_text());requests[a['stats']['request_id']]=a['stats']
    summary={'images':500,'source_series':len(set(r['pack'] for r in items)),'distinct_pixel_hashes':len(set(r['pixel_hash'] for r in manifest['items'])),
             'excluded_near_duplicate_pairs':len(duplicate_pairs),'threshold':.75,'queries_with_candidates':sum(n>0 for n in counts),
             'queries_without_candidates':sum(n==0 for n in counts),'directed_pairs':pair_count,'unique_pairs':pair_count//2,
             'median_candidates':float(np.median(counts)),'max_candidates':max(counts),'sensitivity':sensitivity,
             'generation_seconds':round(time.time()-started,2),'api_requests':len(requests),
             'api_total_tokens':sum(r['usage'].get('total_tokens',0) for r in requests.values()),
             'visual_models':sorted(set(r['model'] for r in requests.values()))}
    result={'summary':summary,'dataset':{k:manifest[k] for k in ('repository','commit','seed','sampling','scope','rights')},
            'provenance':lab.PROVENANCE,'limitations':['视觉语义描述为模型自动生成，并非人工标注的相同语义真值。','当前语义检索为 BGE 中文含义文本基线，未运行 GSE。','风格为作者 ALADIN-ViT 后续版本，非 ICCV2021 CNN。','静态图便利样本，不覆盖动图及全网分布；无用户实际聊天语境。','阈值为测试参数，未用人工验证集校准；模型距离不是人的感知差异或沟通趋异效果。'],
            'items':items,'queries':queries,'maximums':sorted(maximums,key=lambda r:-r['style_distance'])}
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False))
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (out/'stage1-semantic-candidates.json').write_text(json.dumps({q:{'meaning':rows[i]['meaning'],'candidates':[{'id':r['id'],'semantic_score':r['semantic_score']} for r in sorted(v['stage2'],key=lambda x:-x['semantic_score'])]} for i,(q,v) in enumerate(queries.items())},ensure_ascii=False,indent=2))
    (out/'stage2-style-ranking.json').write_text(json.dumps({q:v['stage2'] for q,v in queries.items()},ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/sticker-benchmark-500');a=p.parse_args();main(Path(a.output).resolve())
