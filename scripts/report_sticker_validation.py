# -*- coding: utf-8 -*-
"""Render frozen-model validation and export unlabelled semantic/style audit pairs."""
import argparse
import csv
import html
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from research.sticker_encoder.weak_model import save, digest


def table(headers, rows):
    esc = lambda v: html.escape(str(v))
    return '<div class="scroll"><table><thead><tr>' + ''.join('<th>'+esc(h)+'</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in rows) + '</tbody></table></div>'


def run(out):
    report = json.loads((out/'validation-report.json').read_text())
    audit = json.loads((out/'split-audit.json').read_text())
    rows = json.loads((out/'evaluated-items.json').read_text())
    data = np.load(out/'recomputed-features.npz',allow_pickle=False)
    public = np.load(ROOT/'outputs/sticker-benchmark-500/features.npz',allow_pickle=False)
    style_lookup = {str(a): v for a,v in zip(public['ids'],public['style'])}
    style = np.asarray([style_lookup[r['id']] for r in rows[:500]], dtype=np.float32)
    style /= np.linalg.norm(style,axis=1,keepdims=True)
    style_distance = 1-np.clip(style@style.T,-1,1)
    semantic = np.clip(data['v2_image'][:500]@data['v2_image'][:500].T,-1,1)
    blocked = {frozenset((p['a'],p['b'])) for p in audit['all_near_duplicate_pairs']}
    selections = []; pairs = {}
    for i,q in enumerate(rows[:500]):
        if q['eval_split']!='test': continue
        eligible = [j for j,r in enumerate(rows[:500]) if i!=j and frozenset((q['id'],r['id'])) not in blocked]
        top = max(eligible,key=lambda j:float(semantic[i,j]))
        gated = [j for j in eligible if semantic[i,j]>=.7]
        ranked = sorted(gated,key=lambda j:(-float(style_distance[i,j]),-float(semantic[i,j])))
        chosen = ranked[0] if ranked else None
        selections.append({'query':q['id'],'nearest_candidate':rows[top]['id'], 'semantic_candidates_at_0_7':len(gated),
            'max_style_candidate':rows[chosen]['id'] if chosen is not None else None,
            'max_style_distance':float(style_distance[i,chosen]) if chosen is not None else None,
            'all_gated_candidates':[{'id':rows[j]['id'],'semantic_score':float(semantic[i,j]),'style_distance':float(style_distance[i,j])} for j in ranked]})
        for j,stage in [(top,'nearest_semantic'),(chosen,'semantic_gate_then_max_style')]:
            if j is None: continue
            key=(q['id'],rows[j]['id'])
            if key not in pairs:
                pairs[key]={'query_id':q['id'],'candidate_id':rows[j]['id'],'query_image':q['export_image'],
                    'candidate_image':rows[j]['export_image'],'stages':[], 'semantic_score':float(semantic[i,j]),
                    'style_distance':float(style_distance[i,j]),'same_source_pack':q['pack']==rows[j]['pack']}
            pairs[key]['stages'].append(stage)
    review = list(pairs.values()); random.Random(20261118).shuffle(review)
    for i,p in enumerate(review,1): p['pair_id']=f'R{i:03d}'
    covered=sum(s['semantic_candidates_at_0_7']>0 for s in selections)
    pipeline={'queries':len(selections),'gallery_images':500,'exclude_self_and_detected_duplicates':True,
        'semantic_threshold':.7,'human_calibrated_threshold':False,'queries_with_candidates':covered,
        'coverage':covered/len(selections), 'style_rank_numeric_check':True,
        'style_features_sha256':digest(ROOT/'outputs/sticker-benchmark-500/features.npz'),
        'semantic_equivalence_accuracy':None,'perceived_style_divergence_accuracy':None,
        'reason':'No independent human pair annotations available. Coverage and sorted distances are not accuracy.', 'selections':selections}
    save(out/'two-stage-audit.json',pipeline)
    review_dir=out/'human-review';review_dir.mkdir(exist_ok=True)
    save(review_dir/'researcher-key.json',review)
    fields=['pair_id','rater_id','same_meaning_yes_no_uncertain','direction_matches_yes_no_uncertain','style_difference_1_to_5','notes']
    with (review_dir/'ratings-template.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows({'pair_id':p['pair_id']} for p in review)
    cards=[]
    for p in review:
        cards.append('<article><h2>'+p['pair_id']+'</h2><div class="pair"><div><p>查询图</p><img src="../'+html.escape(p['query_image'],quote=True)+'"></div><div><p>候选图</p><img src="../'+html.escape(p['candidate_image'],quote=True)+'"></div></div></article>')
    css='body{font:16px/1.7 system-ui,"PingFang SC",sans-serif;color:#263d36;background:#f4f5ef;margin:0}main{max-width:1120px;margin:auto;padding:24px}h1{font-size:28px}section,article{background:white;border:1px solid #d8e0d7;border-radius:12px;padding:20px;margin:18px 0}.scroll{overflow:auto}table{border-collapse:collapse;min-width:560px;width:100%}th,td{padding:12px;border-bottom:1px solid #e0e6df;text-align:left;white-space:nowrap}a{color:#27694f}.note{background:#fff3da;padding:16px;border-left:4px solid #b78c3d}.pair{display:flex;gap:24px}.pair>div{flex:1;min-width:0}.pair img{width:100%;height:200px;object-fit:contain}.small{color:#61746a;font-size:14px}code{word-break:break-all}@media(max-width:600px){main{padding:14px}section,article{padding:14px}.pair{gap:12px}.pair img{height:150px}}'
    instructions='''<h1>图片对人工核对</h1><p>本页未显示算法分数、选择阶段及机器描述。请先独立评价，再查看 researcher-key.json。</p><p>同义：是否表达相同交际意图，并保留说话方向？“向对方道谢”和“接受感谢”应区分；无法确定请选择 uncertain。表达方向单独记录。</p><p>风格差异：1＝几乎没有，2＝较小，3＝中等，4＝较大，5＝很大；比较绘制媒介、写实程度、线条、色彩和构图。视觉风格差异与语义差异分开评分。</p><p>未提供实际聊天情境，因此本表只能核对一般表达可能性，不能核对个人语义或对话回复是否适当。建议两位评价者独立填写，不预填结论。</p><p><a href="ratings-template.csv">下载评分模板</a> · <a href="../验证报告.html">返回验证报告</a></p>'''
    (review_dir/'图片对核对.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>图片对核对</title><style>'+css+'</style><main>'+instructions+''.join(cards)+'</main></html>',encoding='utf8')
    pct=lambda x:f'{100*x:.1f}%'
    splits=[['训练集',350,'67','用于梯度更新'],['验证集',75,'17','用于选训练轮次/方案'],['原有测试集',75,'18','未做梯度更新；已在此前评估过'],['新增留出图片',75,'12','标注未完成，未计算成绩' if not report['fresh_test_evaluated'] else '冻结模型检验']]
    comparison=[]
    for model,name in [('base','基础模型'),('v1','第一版 v1'),('v2','当前 v2')]:
        m=report['metrics'][model]['test']['within_split']['image_to_caption']
        comparison.append([name,f"{m['hits_at_1']}/75 · {pct(m['recall_at_1'])}",f"{m['hits_at_5']}/75 · {pct(m['recall_at_5'])}",pct(m['recall_at_10']),f"{m['mrr']:.3f}"])
    common=[]
    for split,name in [('train','训练集'),('validation','验证集'),('test','测试集')]:
        m=report['metrics']['v2'][split]['common_500_gallery']['image_to_caption']
        common.append([name,m['n_queries'],m['n_candidates'],pct(m['recall_at_1']),pct(m['recall_at_5']),f"{m['mrr']:.3f}"])
    ci=report['paired_model_differences']['test']['results']['image_to_caption']['mrr']['paired_pack_bootstrap_95ci']
    reverse=[]
    for model in ('v1','v2'):
        m=report['metrics'][model]['test']['within_split']['caption_to_image']
        reverse.append([model,pct(m['recall_at_1']),pct(m['recall_at_5']),f"{m['mrr']:.3f}"])
    body='''<h1>表情包算法：数据分区与冻结模型验证</h1><p class="small">2026 年 11 月 18 日 · 不重新训练、不调门槛 · 从已保存权重重新编码图片与描述</p><div class="note"><b>结论：分区隔离检查通过，旧结果可复现；泛化仍有明显不足。</b><br>当前测量的是找回图片配对的自动含义描述，不是“同义表情包判断正确率”。新增 75 张测试图片缺少标签，尚未得出新测试成绩。</div>'''
    body+='<section><h2>1. 数据已经分开</h2>'+table(['分区','图片数','来源系列数','用途'],splits)+'<p>训练、验证、测试之间：ID 交叉 0，文件相同 0，检测到的近重复交叉 0，来源系列交叉 0。检测到 1 对近重复图片，两张均在训练集。新旧测试图片允许共用已留出的来源系列。</p><p>保持原分区，不重新随机拆分已有图片，以免把模型已学过的图片当成测试。近重复检测是保守的像素/缩略图检查，不能保证识别所有裁切、改字或角色变体。</p><p><a href="splits/train/manifest.csv">训练集清单</a> · <a href="splits/validation/manifest.csv">验证集清单</a> · <a href="splits/test/manifest.csv">测试集清单</a> · <a href="split-audit.json">完整隔离检查</a></p></section>'
    body+='<section><h2>2. 复现此前测试：75 个候选描述</h2><p>每张测试图片从原有测试集的 75 条描述中，检索自己的配对自动描述。R@5 表示正确配对落在前五名，不是第一名正确。</p>'+table(['模型','R@1','R@5','R@10','MRR'],comparison)+f'<p>v2 比 v1 净多 1 个前五命中。按 18 个素材系列成组重采样的 MRR 差值 95% 区间为 [{ci[0]:.3f}, {ci[1]:.3f}]，跨 0；没有足够证据认定 v2 稳定优于 v1。</p><p>反向任务（描述找图片）：</p>'+table(['模型','R@1','R@5','MRR'],reverse)+'</section>'
    body+='<section><h2>3. 统一到 500 个候选，比较泛化</h2><p>训练、验证、测试查询都检索相同的 500 条描述。候选数量一致，便于判断模型对已训练图片和未训练图片的差别。这里的分数不能与上面的 75 候选分数混用。</p>'+table(['v2 查询分区','查询数','候选数','R@1','R@5','MRR'],common)+'<p>训练集与测试集差距较大，提示过拟合或来源分布差异。仅增加训练轮次不能据此保证改善。其他语义相近的描述仍会被“精确配对”指标算错，因此也不能反推真实同义判断正确率。</p></section>'
    body+='<section><h2>4. 语义筛选 → 最大风格差异</h2>'+f'<p>在 75 个测试查询中，使用原来的探索门槛 0.70，并排除自身和已检测近重复图后，{covered} 个查询有候选（覆盖率 {pct(covered/75)}）。随后按 ALADIN-ViT 距离 1 − cosine 从大到小排序，计算流程校验通过。</p><p><b>覆盖率和排序正确不等于语义正确或人感受到的趋异正确。</b>没有人工图片对标签，无法估计这两个准确率。已生成 {len(review)} 个去重后的查询—候选对，机器未代填人工评分。核对集包含各查询的最近邻，以及有合格候选时的最大风格差异选择；部分最近邻没有通过 0.70 门槛，仅供错误分析，不是实际可发送候选。</p><p><a href="human-review/图片对核对.html">查看盲化图片对</a> · <a href="human-review/ratings-template.csv">空白评分模板</a> · <a href="two-stage-audit.json">两阶段明细</a></p></section>'
    body+='<section><h2>5. 文件与重现</h2><p>已验证基础模型、v1 和 v2 权重哈希；重新编码结果与存档特征差异小于 0.00001，原有测试成绩一致。此次没有更新权重或正式聊天模型。</p><p>新增 75 张留出图片已经单独保存。方舟本次再次返回账户欠费，首批标注失败后即停止，未把未标注数据计入成绩。它们来自同一公共素材源的既有留出系列，并非独立外部测试集。</p><p><a href="validation-report.json">完整验证数据</a> · <a href="per-image-predictions.csv">逐图预测与排名</a> · <a href="protocol.json">冻结验证约定</a></p></section>'
    if report['fresh_test_evaluated']:
        body=body.replace('新增 75 张测试图片缺少标签，尚未得出新测试成绩。','新增 75 张测试图片已完成自动标注，并使用冻结权重评估。').replace('方舟本次再次返回账户欠费，首批标注失败后即停止，未把未标注数据计入成绩。','新增图片的逐模型检验结果保存在完整验证数据中。')
    (out/'验证报告.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>表情包算法验证</title><style>'+css+'</style><main>'+body+'</main></html>',encoding='utf8')
    text='''本目录用途：按原分区分开保存图片并独立复核当前模型。没有重新训练。
splits/train/：350 张训练图片、自动描述和清单。
splits/validation/：75 张选模用图片、自动描述和清单。
splits/test/：75 张原有测试图片、自动描述和清单。
fresh-test-corpus/：75 张新增留出图片，因 API 欠费缺少标签，尚未计入评估。
验证报告.html：中文报告（用浏览器打开）。
per-image-predictions.csv：三种模型逐图排名，含精确配对目标。
split-audit.json：分区、来源系列、重复图检查。
validation-report.json：全部数值与冻结模型依据。
human-review/：机器未填答案的图片对评价表，以及供研究者核对的算法选择依据。

重跑：
.venv-style/bin/python scripts/validate_sticker_encoder.py audit
.venv-style/bin/python scripts/validate_sticker_encoder.py evaluate
.venv-style/bin/python scripts/report_sticker_validation.py

余额恢复后，仅补充新测试标签再评估：
.venv-style/bin/python scripts/validate_sticker_encoder.py annotate-fresh
.venv-style/bin/python scripts/validate_sticker_encoder.py evaluate

不要重新随机拆分已有 500 张图片后声称是新测试。不要根据测试结果调门槛。
机器描述检索与人工同义判断是两种任务；本报告没有人工正确率。
'''
    if report['fresh_test_evaluated']:
        text=text.replace('因 API 欠费缺少标签，尚未计入评估。','已完成自动标注并计入新增图片的冻结检验。')
    (out/'目录说明.txt').write_text(text,encoding='utf8')
    print(json.dumps({'report':str(out/'验证报告.html'),'pipeline_queries_with_candidates':covered,'review_pairs':len(review)},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='outputs/sticker-validation-20261118')
    run(Path(p.parse_args().output).resolve())
