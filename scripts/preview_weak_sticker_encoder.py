# -*- coding: utf-8 -*-
"""Build an inspectable public-gallery preview from a completed weak training run."""
import argparse
import base64
import io
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.sticker_encoder.weak_model import save

HTML = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>小屿 · 自训练语义模型</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f4f5ef;color:#243e36;font:16px/1.6 system-ui,"PingFang SC",sans-serif}header,main,footer{max-width:1320px;margin:auto;padding:24px}h1{font-size:32px;line-height:1.3}h2{font-size:21px}a{color:#27694f}.panel{background:white;border:1px solid #d8e0d6;border-radius:16px;padding:22px;margin-bottom:20px}.muted{color:#637568;font-size:14px}.note{background:#f8efdf;border-left:4px solid #bc944d;padding:14px 18px}.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}button,input,select{font:inherit;padding:9px 12px;border:1px solid #c7d4ca;border-radius:8px;max-width:100%;background:white;color:inherit}button{cursor:pointer}input{flex:1}select{max-width:550px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #dbe4dc}.scroll{overflow:auto}.stages{display:grid;grid-template-columns:1fr 1fr;gap:20px}.cards{display:grid;grid-template-columns:1fr 1fr;gap:12px;max-height:740px;overflow:auto}.card{border:1px solid #dbe4dc;border-radius:10px;padding:12px;min-width:0}.card img{width:100%;height:130px;object-fit:contain;background:#f6f7f3;border-radius:6px}.card p{font-size:14px;margin:5px 0}.score{font-size:21px;font-weight:650;color:#27694f}.query{display:flex;gap:20px;margin-top:20px}.query img{width:150px;height:150px;object-fit:contain}.tag{display:inline-block;background:#edf3ec;font-size:12px;padding:2px 7px;border-radius:4px}details{margin-top:16px}summary{cursor:pointer}pre{white-space:pre-wrap;font-size:12px;word-break:break-word}@media(max-width:760px){header,main,footer{padding:16px}.stages{grid-template-columns:1fr}.panel{padding:16px}h1{font-size:26px}.query img{width:100px;height:100px}.cards{max-height:620px}.row select{width:100%}}</style>
<header><nav class="row"><a href="/xiaoyu/research">研究控制台</a><a href="/xiaoyu/research/benchmark-500">原 BGE 基线测试</a></nav><p class="muted">XIAOYU · OWN SEMANTIC ENCODER · WEAK SUPERVISION</p><h1>我们自己的表情包语义模型</h1><p>从公共表情包及自动含义描述中训练视觉编码器。检索时直接读取图片，再用 ALADIN-ViT 对候选进行风格差异排序。</p><div class="note"><b>这是已经实际训练的实验模型，非作者 GSE。</b>训练和评估描述均由机器生成。下方指标是“找回配对描述”的表现，不能解读成同义判断准确率，也不能证明适用于个人聊天语境。尚未替换正式被试的选图模型。</div></header>
<main><section class="panel"><h2>训练与留出评估</h2><p id="runInfo"></p><div class="scroll"><table><thead><tr><th>模型</th><th>验证集 R@1</th><th>验证集 MRR</th><th>测试集 R@1</th><th>测试集 R@5</th><th>测试集 MRR</th></tr></thead><tbody id="metrics"></tbody></table></div><p class="muted">R@1 / R@5：配对的自动描述排在前 1 / 5 位的比例。MRR：正确配对描述排名倒数的平均值。每个分区内独立检索；语义近似的其他描述仍可能被该指标算作错误。</p><p id="outcome"></p><details><summary>每轮训练记录</summary><div class="scroll"><table><thead><tr><th>轮次</th><th>训练损失</th><th>验证 R@1</th><th>验证 MRR</th></tr></thead><tbody id="epochs"></tbody></table></div></details></section>
<section class="panel"><div class="row"><input id="search" placeholder="按含义或编号搜索"><select id="query" aria-label="查询图"></select><button id="next">下一张</button></div><div class="row" style="margin-top:12px"><label>语义模型 <select id="model"><option value="trained">已训练的小屿模型</option><option value="baseline">训练前的 Chinese-CLIP</option></select></label><label>探索门槛 <select id="threshold"><option value="0.5">0.50</option><option value="0.6">0.60</option><option value="0.7" selected>0.70</option><option value="0.8">0.80</option><option value="0.9">0.90</option></select></label><button id="download">下载当前完整结果</button></div><p class="muted">门槛仅用于浏览，未经人工同义标签校准。训练前后分数分布可能不同，同一门槛下候选数量不代表模型优劣。</p><div id="queryCard" class="query"></div></section>
<div class="stages"><section class="panel"><h2>① 图片语义相近候选</h2><p id="count1" class="muted"></p><div id="stage1" class="cards"></div></section><section class="panel"><h2>② 同一候选中的风格差异</h2><p id="count2" class="muted"></p><div id="stage2" class="cards"></div></section></div>
<section class="panel"><h2>方法与使用边界</h2><p>基础模型为固定版本的 Chinese-CLIP ViT-B/16。中文文本编码器冻结，微调视觉编码器最后两层、输出归一化及投影层。使用对称图文对比损失；BGE 只在训练批次内屏蔽可能同义的负例，不作为推理输入。</p><p>按素材来源系列与检测到的近重复图片分组划分数据，以验证集选择训练轮次，选择结束后才计算测试指标。图片整体缩放并补白，保留图片上的文字。</p><p>这是 500 张公共静态图片的小规模弱监督试验。测试部分未参与本次梯度训练或模型选择，但此前已经在 BGE 测试中探索过，不能称为全新的盲测集。正式实验还需要独立人工标注的图片对、阈值校准与聊天语境核对。</p><p>风格得分来自现有作者 ALADIN-ViT 权重，风格距离 = 1 − 余弦相似度；视觉距离不等于人的趋异感知。</p><details><summary>查看完整训练元数据</summary><pre id="metadata"></pre></details></section></main><footer>仅公共素材算法研究预览 · 图片权利归原作者 · 不包含被试聊天数据</footer>
<script id="data" type="application/json">__DATA__</script><script>
'use strict';const D=JSON.parse(document.getElementById('data').textContent),R=D.report,$=id=>document.getElementById(id),byId=Object.fromEntries(D.items.map((r,i)=>[r.id,{...r,index:i}]));
if(D.remoteRows){D.similarities={baseline:{},trained:{}};D.style={}}
if(D.packed){const unpack=s=>{const b=Uint8Array.from(atob(s),c=>c.charCodeAt(0)),v=new DataView(b.buffer),n=D.items.length;return Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>v.getFloat32(4*(i*n+j),true)))};D.similarities={baseline:unpack(D.packed.baseline),trained:unpack(D.packed.trained)};D.style=unpack(D.packed.style);delete D.packed}
const el=(tag,text)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;return e},pct=n=>(100*n).toFixed(1)+'%',num=n=>n.toFixed(3);function img(r){const e=el('img');e.src=r.thumbnail;e.alt=r.id;e.loading='lazy';return e}
const selected=R.epochs.find(e=>e.epoch===R.selected_epoch).validation;
const total=Object.values(R.data_counts).reduce((a,b)=>a+b,0),before=R.baseline_label||'训练前',after=R.trained_label||'训练后';
$('runInfo').textContent=`${R.implementation} · ${total} 张图片；训练 ${R.data_counts.train} 张 / 验证 ${R.data_counts.validation} 张 / 测试 ${R.data_counts.test} 张。${R.validation_trials?'比较 '+R.validation_trials.length+' 种方案，选中方案 '+R.selected_trial+'，':''}验证集选中第 ${R.selected_epoch} 轮；可训练参数 ${(R.trainable_parameter_count/1e6).toFixed(2)} 百万。下方图库展示原有 ${D.items.length} 张图片。`;
$('model').options[0].textContent=after+' · 小屿模型';$('model').options[1].textContent=before+' · 对照模型';
for(const [name,v,t] of [[before,R.baseline_validation,R.baseline_test],[after,selected,R.trained_test]]){const tr=el('tr');for(const x of [name,pct(v.image_to_caption.recall_at_1),num(v.image_to_caption.mrr),pct(t.image_to_caption.recall_at_1),pct(t.image_to_caption.recall_at_5),num(t.image_to_caption.mrr)])tr.append(el('td',x));$('metrics').append(tr)}
$('outcome').textContent=R.improved_on_heldout_caption_mrr?'本次留出测试的描述检索 MRR 高于训练前。该结果仅支持这一代理任务，仍需人工检查表情包对的同义关系。':'本次留出测试的描述检索 MRR 未超过训练前，不应据此替换正式系统。训练结果完整保留，便于分析。';
for(const e of R.epochs){const tr=el('tr');for(const x of [e.epoch===R.selected_epoch?e.epoch+'（选中）':e.epoch,num(e.loss),pct(e.validation.image_to_caption.recall_at_1),num(e.validation.image_to_caption.mrr)])tr.append(el('td',x));$('epochs').append(tr)}
$('metadata').textContent=JSON.stringify(R,null,2);
if(R.test_sets){const section=el('section');section.className='panel';section.append(el('h2',R.test_sets.new_images?'新增图片与原有图片分别检验':'原有留出图片上的版本对比'));for(const [key,title] of [['new_images','新增留出图片'],['old_images','原有留出图片']]){const s=R.test_sets[key];if(!s)continue;const a=s.baseline_v1.image_to_caption,b=s.continued_v2.image_to_caption;section.append(el('h3',title+' · '+s.ids.length+' 张'),el('p',`R@1：${pct(a.recall_at_1)} → ${pct(b.recall_at_1)}；R@5：${pct(a.recall_at_5)} → ${pct(b.recall_at_5)}；MRR：${num(a.mrr)} → ${num(b.mrr)}`));}const note=el('p',R.test_sets.new_images?'上表使用合并测试集。这里保持两个测试子集各自的检索范围不变，比较上一版与新版。新增图片来自已留出的素材系列，并非独立外部数据集。所有描述均为自动标注。':'本轮继续使用原有 75 张留出测试图片，未加入新的测试图片。它们未参与本轮训练或选模，但此前已用于 v1 评估，因此这不是新盲测。所有描述均为自动标注。');note.className='muted';section.append(note);const reverseBefore=R.baseline_test.caption_to_image,reverseAfter=R.trained_test.caption_to_image;section.append(el('p',`反向检索（描述找图片）：R@1 ${pct(reverseBefore.recall_at_1)} → ${pct(reverseAfter.recall_at_1)}；R@5 ${pct(reverseBefore.recall_at_5)} → ${pct(reverseAfter.recall_at_5)}。两个方向可能存在取舍。`));const ci=R.test_sets.combined.paired_uncertainty.mrr.query_bootstrap_95ci;section.append(el('p',`图→描述 MRR 差值的查询级 bootstrap 95% 区间为 ${num(ci[0])} 至 ${num(ci[1])}，未校正素材系列的聚类。若区间跨 0，不能据此认定稳定提升。`));$('runInfo').closest('section').after(section);const methodSection=$('metadata').closest('section');const ps=methodSection.querySelectorAll(':scope > p');ps[0].textContent=`从上一版 v1 权重继续训练，中文文本编码器保持冻结，只更新视觉编码器最后两层、归一化及投影层。降低学习率，比较图文对比学习和加入 BGE 软语义邻域蒸馏的方案；验证集选中蒸馏权重 ${R.selected_relational_weight??'无（保留上一版）'}。推理仍然只读取图片。`;ps[2].textContent=`本轮训练使用 ${total} 张已标注公共静态图片，保持原有素材系列的训练、验证和测试分区。${R.test_sets.new_images?'新增测试图片未用于梯度训练和本轮选模。':'另行收集的 750 张新图片尚未完成标注，未进入本轮训练或测试。'}原有测试图片曾用于 v1 评估。弱监督指标仍不能替代人工同义判断。`;}
function filter(){const old=$('query').value,term=$('search').value.toLowerCase();$('query').replaceChildren();for(const r of D.items)if((r.id+' '+r.meaning+' '+r.ocr).toLowerCase().includes(term)){const o=el('option',r.id+' · '+r.meaning.slice(0,35));o.value=r.id;$('query').append(o)}if([...$('query').options].some(o=>o.value===old))$('query').value=old;else if(!old&&D.initialQuery&&[...$('query').options].some(o=>o.value===D.initialQuery))$('query').value=D.initialQuery;render()}
function candidates(){const q=byId[$('query').value];if(!q||!D.similarities[$('model').value][q.index])return null;const threshold=Number($('threshold').value),scores=D.similarities[$('model').value][q.index],style=D.style[q.index];const list=D.items.map((r,j)=>({id:r.id,semantic_score:scores[j],style_distance:style[j]})).filter((r,j)=>j!==q.index&&r.semantic_score>=threshold&&!D.excluded.includes([q.index,j].sort((a,b)=>a-b).join(':')));return {query:q.id,model:$('model').value,threshold,stage1:[...list].sort((a,b)=>b.semantic_score-a.semantic_score),stage2:[...list].sort((a,b)=>b.style_distance-a.style_distance)}}
function card(r,stage){const item=byId[r.id],box=el('article');box.className='card';const score=el('p',num(stage===1?r.semantic_score:r.style_distance));score.className='score';box.append(img(item),el('p',r.id+' · '+item.split),score,el('p',item.meaning));const p=el('p',stage===1?'图片语义余弦分数':'风格距离 · 语义 '+num(r.semantic_score));p.className='muted';box.append(p);return box}
async function render(){const requested=byId[$('query').value];if(D.remoteRows&&requested&&!D.style[requested.index]){for(const id of ['stage1','stage2'])$(id).replaceChildren(el('p','加载结果中…'));try{const response=await fetch(D.remoteRows+requested.id+'.json');if(!response.ok)throw new Error('load');const row=await response.json();D.similarities.baseline[requested.index]=row.baseline;D.similarities.trained[requested.index]=row.trained;D.style[requested.index]=row.style;}catch(e){for(const id of ['stage1','stage2'])$(id).replaceChildren(el('p','结果暂未加载，请重新选择这张图片。'));return}if($('query').value!==requested.id)return}const v=candidates();$('queryCard').replaceChildren();$('stage1').replaceChildren();$('stage2').replaceChildren();if(!v)return;const q=byId[v.query],text=el('div'),tag=el('span',q.id+' · '+q.split);tag.className='tag';text.append(tag,el('h2',q.meaning),el('p','文字：'+(q.ocr||'无')),el('p','机器标注可能的歧义：'+q.ambiguity));$('queryCard').append(img(q),text);$('count1').textContent=v.stage1.length+' 张候选（展示前 40 张；下载含全部）。尚未经人工判为同义。';$('count2').textContent='相同的 '+v.stage2.length+' 张候选按风格距离从大到小排序。';for(const [id,list,stage] of [['stage1',v.stage1,1],['stage2',v.stage2,2]]){if(!list.length)$(id).append(el('p','当前门槛下无候选，保留空结果。'));for(const r of list.slice(0,40))$(id).append(card(r,stage))}}
$('search').oninput=filter;$('query').onchange=render;$('model').onchange=render;$('threshold').onchange=render;$('next').onclick=()=>{const q=$('query');if(q.options.length)q.selectedIndex=(q.selectedIndex+1)%q.options.length;render()};$('download').onclick=()=>{const v=candidates();if(!v)return;const a=el('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(v,null,2)],{type:'application/json'}));a.download=v.query+'-'+v.model+'-two-stages.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};filter();
</script></html>'''


def run(run_dir, source):
    report = json.loads((run_dir / 'training-report.json').read_text())
    dataset = json.loads((run_dir / 'dataset.json').read_text())
    old = json.loads((source / 'results.json').read_text())
    f = np.load(run_dir / 'features.npz', allow_pickle=False)
    old_f = np.load(source / 'features.npz', allow_pickle=False)
    rows_by_id = {r['id']: r for r in dataset['rows']}
    gallery_rows = [rows_by_id[str(id_)] for id_ in f['ids']]
    idx = {str(id_): i for i, id_ in enumerate(old_f['ids'])}
    style = np.array([old_f['style'][idx[str(id_)]] for id_ in f['ids']])
    style /= np.linalg.norm(style, axis=1, keepdims=True)
    style_distance = 1 - np.clip(style @ style.T, -1, 1)
    similarity = {}
    for name in ('baseline', 'trained'):
        v = f[name]
        assert np.isfinite(v).all() and np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-5)
        similarity[name] = np.round(np.clip(v @ v.T, -1, 1), 6).tolist()
    old_items = {r['id']: r for r in old['items']}
    items = [{**old_items[r['id']], 'split': r['split']} for r in gallery_rows]
    new_idx = {str(id_): i for i, id_ in enumerate(f['ids'])}
    excluded = [':'.join(map(str, sorted([new_idx[a], new_idx[b]]))) for a, b in dataset['near_duplicate_links_kept_together'] if a in new_idx and b in new_idx]
    value = {'report': report, 'items': items, 'similarities': similarity,
             'style': np.round(style_distance, 6).tolist(), 'excluded': excluded}
    save(run_dir / 'preview-results.json', value)
    # Store exact float32 values in base64 instead of long decimal strings.
    # Compress only display thumbnails; training and feature files are untouched.
    display_items = []
    for item in items:
        with Image.open(io.BytesIO(base64.b64decode(item['thumbnail'].split(',', 1)[1]))) as image:
            image.thumbnail((160, 160))
            buf = io.BytesIO()
            image.convert('RGB').save(buf, 'JPEG', quality=72)
        display_items.append({**item, 'thumbnail': 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()})
    def pack(matrix):
        return base64.b64encode(np.asarray(matrix, dtype='<f4').tobytes()).decode()
    initial = next((row['id'] for i, row in enumerate(items) if any(j != i and similarity['trained'][i][j] >= .7 and ':'.join(map(str, sorted([i, j]))) not in excluded for j in range(len(items)))), items[0]['id'])
    web_value = {'report': report, 'items': display_items, 'excluded': excluded, 'initialQuery': initial,
                 'packed': {'baseline': pack(similarity['baseline']), 'trained': pack(similarity['trained']), 'style': pack(value['style'])}}
    (run_dir / '自训练模型_两阶段预览.html').write_text(HTML.replace('__DATA__', json.dumps(web_value, ensure_ascii=False).replace('<', '\\u003c')))
    stage1, stage2 = {}, {}
    for i, row in enumerate(items):
        candidates = [{'id': r['id'], 'semantic_score': similarity['trained'][i][j],
                       'style_distance': float(style_distance[i, j])}
                      for j, r in enumerate(items) if j != i and similarity['trained'][i][j] >= .7
                      and ':'.join(map(str, sorted([i, j]))) not in excluded]
        stage1[row['id']] = sorted(candidates, key=lambda r: -r['semantic_score'])
        stage2[row['id']] = sorted(candidates, key=lambda r: -r['style_distance'])
    for name, result in [('stage1-visual-semantic.json', stage1), ('stage2-style-ranking.json', stage2)]:
        save(run_dir / name, {'model': report['implementation'], 'exploratory_threshold': .7,
                             'human_calibrated': False, 'queries': result})
    print(json.dumps({'preview': str(run_dir / '自训练模型_两阶段预览.html'), 'queries': len(stage1),
                      'with_candidates_at_0_7': sum(bool(v) for v in stage1.values())}, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--source', default='outputs/sticker-benchmark-500')
    args = p.parse_args()
    run(Path(args.run), Path(args.source))
