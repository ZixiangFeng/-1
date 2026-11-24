"""Test-only paired validation for StickerQueries. No fit or parameter selection."""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import sticker_pragmatics as use
from scripts.collect_stickerqueries_holdout import CACHE, HOLDOUT, RUN, N, REV, read, sha, stored
from scripts.evaluate_pragmatic_retrieval import score_queries

FAMILY_REVIEW=RUN/'query-family-review.tsv'
PAIR_REVIEW=RUN/'pair-review.tsv'
WEB_GALLERY=ROOT/'outputs/sticker-context-v6/web/images'


def verify():
    manifest=read(HOLDOUT/'manifest.json')
    protocol=read(RUN/'protocol.json')
    if manifest['protocol_sha256']!=sha((RUN/'protocol.json').read_bytes()) or protocol['revision']!=REV:
        raise ValueError('Test sample or protocol differs.')
    for filename,checksum in protocol['frozen'].items():
        if sha((ROOT/filename).read_bytes())!=checksum:
            raise ValueError('Frozen input has changed: '+filename)
    rows=manifest['rows']
    if len(rows)!=N or len({r['source_id'] for r in rows})!=N or len({r['source_caption'] for r in rows})!=N:
        raise ValueError('Duplicate query, utterance, or wrong count.')
    for r in rows:
        if sha((HOLDOUT/'images'/(r['id']+'.png')).read_bytes())!=r['image_sha256'] or r['nearest_phash_bits']<=7:
            raise ValueError('Image overlap or change.')
    return rows


def thumbnail(sheet,path,x,y,w,h):
    with Image.open(path) as im:
        pic=im.convert('RGBA');pic.thumbnail((w,h),Image.Resampling.LANCZOS)
        sheet.paste(pic,(x+(w-pic.width)//2,y+(h-pic.height)//2),pic)


def sheets(rows,key=None):
    paired=key is not None
    out=CACHE/('blind-pairs' if paired else 'queries')
    out.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype('/System/Library/Fonts/Hiragino Sans GB.ttc',17)
    by_id={r['id']:r for r in rows}
    entries=key if paired else rows
    cols,lines=(3,4) if paired else (5,4)
    width,height=(360,196) if paired else (196,180)
    per_page=cols*lines
    for start in range(0,len(entries),per_page):
        im=Image.new('RGB',(width*cols,height*lines),'#f2f3ed')
        draw=ImageDraw.Draw(im)
        for j,entry in enumerate(entries[start:start+per_page]):
            r=by_id[entry['query']] if paired else entry
            x=j%cols*width; y=j//cols*height
            limit=width//2 if paired else width
            thumbnail(im,HOLDOUT/'images'/(r['id']+'.png'),x+2,y+1,limit-5,height-47)
            if paired:
                thumbnail(im,WEB_GALLERY/(entry['candidate']+'.jpg'),x+limit+2,y+1,limit-5,height-47)
            label=entry['blind_id'] if paired else r['id']
            text=label+' '+r['source_caption']
            # Wrap the occasional longer query in its own cell.
            line=''; buffer=[]
            for c in text:
                if line and draw.textlength(line+c,font=font)>width-10:
                    buffer.append(line);line=c
                else:line+=c
            buffer.append(line)
            draw.multiline_text((x+4,y+height-43),'\n'.join(buffer),fill='#101820',font=font,spacing=0)
            draw.line((x+width-1,y,x+width-1,y+height), fill='white',width=2)
        im.save(out/('page-%02d.jpg'%(1+start//per_page)),quality=96)
    print('Pages:',(len(entries)+per_page-1)//per_page,'records:',len(entries),flush=True)


def reviews(path, key, expected):
    with path.open() as f:
        rr=list(csv.DictReader(f,delimiter='\t'))
    labels={r[key]:r for r in rr}
    if sorted(labels)!=sorted(expected) or len(labels)!=len(rr):
        raise ValueError('Incomplete or duplicated judgments in '+path.name)
    return labels


def query_labels(rows):
    labels=reviews(FAMILY_REVIEW,'id',[r['id'] for r in rows])
    for r in labels.values():
        opts=set(r['family'].split('|'))
        if opts not in ({'outside'},{'uncertain'}) and (not opts or not opts<=set(use.NAMES)):
            raise ValueError('Invalid family code: '+str(r))
    return labels


def predict(rows):
    query_labels(rows)
    stored(RUN/'before-prediction.json',dict(manifest=sha((HOLDOUT/'manifest.json').read_bytes()),
           protocol=sha((RUN/'protocol.json').read_bytes()),family_judgments=sha(FAMILY_REVIEW.read_bytes()),
           rubric=sha((RUN/'review-protocol.json').read_bytes())))
    qids=[r['id'] for r in rows]
    results,pairs=score_queries(rows,HOLDOUT,[r['source_caption'] for r in rows])
    if list(results)!=qids:
        raise ValueError('Missing query output.')
    pairs=sorted(pairs)
    np.random.default_rng(19361037).shuffle(pairs)
    key=[dict(blind_id='K%03d'%i,query=q,candidate=c) for i,(q,c) in enumerate(pairs,1)]
    stored(RUN/'rankings.json',results)
    stored(RUN/'blind-key.json',key)
    stored(RUN/'prediction-lock.json',dict(rankings=sha((RUN/'rankings.json').read_bytes()),key=sha((RUN/'blind-key.json').read_bytes())))
    sheets(rows,key)


def adjudicate(rows):
    codes=query_labels(rows)
    locked=read(RUN/'before-prediction.json')
    pred_lock=read(RUN/'prediction-lock.json')
    if (locked['family_judgments']!=sha(FAMILY_REVIEW.read_bytes()) or
        locked['manifest']!=sha((HOLDOUT/'manifest.json').read_bytes()) or
        locked['rubric']!=sha((RUN/'review-protocol.json').read_bytes()) or
        pred_lock['rankings']!=sha((RUN/'rankings.json').read_bytes()) or
        pred_lock['key']!=sha((RUN/'blind-key.json').read_bytes())):
        raise ValueError('Predictions or judgments were altered.')
    key=read(RUN/'blind-key.json'); ranks=read(RUN/'rankings.json')
    labels=reviews(PAIR_REVIEW,'blind_id',[r['blind_id'] for r in key])
    for r in labels.values():
        if not all(r[k] in {'yes','no','uncertain'} for k in ('intent','style','same_template')):
            raise ValueError('Invalid pair judgment: '+str(r))
    pairs={(r['query'],r['candidate']):labels[r['blind_id']] for r in key}
    stats={};detail=[]; wins={m:set() for m in ['27','8']}
    for method in ('27','8'):
        out=dict(assigned=0,reference_clear=0,assigned_on_clear=0,correct_on_clear=0,unassigned=0,
                 no_semantic_match_after_classification=0,no_style_match_after_semantics=0,
                 semantic_yes_to_divergent_wrong=0,semantic_yes_to_divergent_unclear=0,semantic_yes_to_divergent_yes=0,
                 checked_maximum=0)
        for stage in ('semantic','divergent'):
            out[stage]=dict(returned=0,same=0,different=0,unclear=0,style_different=0,both=0)
        for q in rows:
            qid=q['id']; pred=ranks[qid][method]; gold=codes[qid]['family']
            family=use.family(pred['group'])
            out['assigned']+=bool(family)
            if gold not in ('uncertain','outside'):
                out['reference_clear']+=1
                out['assigned_on_clear']+=bool(family)
                out['correct_on_clear']+=family in gold.split('|')
            if not family:out['unassigned']+=1
            elif not pred['interchangeable']:out['no_semantic_match_after_classification']+=1
            elif not pred['style_eligible']:out['no_style_match_after_semantics']+=1
            entry=dict(query=qid,source_query=q['source_caption'],reference_family=gold,method=method,**pred)
            for stage in ('semantic','divergent'):
                rating=None
                if pred[stage]:
                    rating=pairs[qid,pred[stage]['id']]
                    s=out[stage];s['returned']+=1
                    s[{'yes':'same','no':'different','uncertain':'unclear'}[rating['intent']]]+=1
                    s['style_different']+=rating['style']=='yes'
                    hit=rating['intent']=='yes' and rating['style']=='yes' and rating['same_template']=='no'
                    s['both']+=hit
                    if stage=='divergent' and hit:wins[method].add(qid)
                entry[stage+'_rating']=rating
            if entry['semantic_rating'] and entry['semantic_rating']['intent']=='yes' and entry['divergent_rating']:
                k={'no':'wrong','yes':'yes','uncertain':'unclear'}[entry['divergent_rating']['intent']]
                out['semantic_yes_to_divergent_'+k]+=1
            out['checked_maximum']+=1
            detail.append(entry)
        if sum(out[k] for k in ('unassigned','no_semantic_match_after_classification','no_style_match_after_semantics'))+out['divergent']['returned']!=len(rows):
            raise ValueError('Invalid coverage denominators')
        stats[method]=out
    summary=dict(dataset='StickerQueries Chinese static holdout',n=len(rows),gallery=1250,pairs_reviewed=len(labels),
        sample_sha=sha((HOLDOUT/'manifest.json').read_bytes()),model=use.model().digest,
        assessments_sha=sha(PAIR_REVIEW.read_bytes()),query_coding_sha=sha(FAMILY_REVIEW.read_bytes()),
        paired_joint_wins=dict(both=len(wins['8']&wins['27']),new_only=len(wins['8']-wins['27']),
            old_only=len(wins['27']-wins['8']),neither=len(rows)-len(wins['8']|wins['27'])), methods=stats)
    stored(RUN/'evaluation.json',summary)
    stored(RUN/'evaluated-queries.json',detail)
    print(summary['methods'],summary['paired_joint_wins'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','predict','score'])
    command=parser.parse_args().command
    rows=verify()
    if command=='prepare':
        stored(RUN/'review-protocol.json',dict(taxonomy=use.NAMES,
          broad='Code acceptable broad family from the human Chinese query and source image before predictions. Outside means a recognizable purpose outside the eight or a purely visual/object tag. Uncertain means insufficient evidence. Neither is used as a positive class. A query may allow two plausible families.',
          pair='Paired images and original query only, shuffled and stripped of source, method, rank, scores and model captions. Primary win requires intent=yes, style=yes, repeated recognizable template=no. No credit for uncertainty. Coding by Codex, not by the source annotators.'))
        sheets(rows)
    elif command=='predict':predict(rows)
    else:adjudicate(rows)
