# -*- coding: utf-8 -*-
"""Broad recall preview; keeps V4 strict run and all numerical reports intact."""
import json, os, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/sticker-broad-v5-20261119';V4=ROOT/'outputs/sticker-divergence-v4-20261119'
web=OUT/'web';web.mkdir(exist_ok=True)
for folder in ('images','rows'):
    if not (web/folder).exists():shutil.copytree(V4/'web'/folder,web/folder,copy_function=os.link)
base=(ROOT/'static/semantic_v4.html').read_text();css=base[base.index('<style>'):base.index('</style>')+8]
rows=json.loads((OUT/'display-dataset.json').read_text())['rows']
data={'version':'broad-v5','items':[{k:r[k] for k in ('id','meaning','split','quality','pack','families','medium','image_sha256','intent_groups','image_predicted_group')} for r in rows],
      'assets':'__ASSET_BASE__','categories':json.loads((OUT/'categories.json').read_text()),
      'report':json.loads((OUT/'training-report.json').read_text()),'evaluation':json.loads((OUT/'evaluation-report.json').read_text())}
# Public metadata uses an artifact identifier, never the researcher's home path.
data['report']['retrieval_semantic_adapter']='xiaoyu-semantic-adapter-v4/semantic-adapter.safetensors'
html='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>小屿 · 宽语义与风格差异 v5</title>__CSS__</head><body>
<header><nav><a href="/xiaoyu/research">研究控制台</a> · <a href="/xiaoyu/research/style-lab">风格实验室</a></nav><p class="eyebrow">XIAOYU · BROAD MEANING / DISTINCT STYLE · V5</p><h1>同类表达，寻找不同画风</h1><p>“收到、明白、懂了、好的”放在同一个回应候选池。先扩大同类素材的召回，再根据这次想表达的意思排序。</p><div class="notice">本轮取消两个严格语义分数门槛；语义分数用于排序。趋异仍排除同人物改图、同来源系列和同类画风。同一大类只代表可供选择，不等于这些图片在任何聊天里都同义。</div></header>
<main><section class="panel"><h2>只保留 8 个宽泛表达大类</h2><div id="categories" class="audit"></div><p>一张表情可保留相近的两个大类，默认用主要大类召回；也可选择这次想表达的大类。讽刺／真诚、安慰对方／表达自己难过等具体用法，留给真实聊天语境判断，不再拆成大量固定子类。</p><details><summary>本轮实际训练与评估</summary><p id="trainingInfo"></p><div class="scroll"><table><thead><tr><th>模型</th><th>大类标签一致率</th><th>大类 Macro-F1</th><th>测试图片数</th></tr></thead><tbody id="metrics"></tbody></table></div><p class="muted">标签由本地 BGE 将既有机器含义描述映射到 8 个预设大类；不是人工金标准。这里评价的是大类识别，不能解读为同义准确率或真实沟通趋异效果。测试集曾被旧版本评估，未参与本轮拟合和选模。</p><p class="muted">当前缓存预览优先采用已有含义描述的大类；图像识别头的预测单独展示。语义排序 = 0.6 × 描述相似度 + 0.4 × 已训练 v4 图像语义分数。先通过形式排除，取语义排名前 20 张，再按 ALADIN 距离排序；这些排序权重是固定工程设定，未经人工效度校准。</p><a id="report">下载本轮评估报告</a></details></section>
<section class="panel"><div class="controls"><input id="search" placeholder="搜索含义或编号，如 B0032、收到、开心" aria-label="搜索图片"><select id="query" aria-label="查询图"></select><button id="next">下一张</button></div><div class="controls" style="margin-top:12px"><label>这次想表达 <select id="intent"><option value="">按当前图片的主要大类</option></select></label><label>语义排序模型 <select id="model"><option value="v4">新训练语义投影</option><option value="v2">旧版 v2 对照</option></select></label><button id="download" class="primary">下载结果与排除依据</button></div><div id="queryCard" class="query"></div></section>
<div class="stages"><section class="panel"><h2>① 同类表达候选</h2><p class="muted">宽泛大类召回；按照含义排序，不以“必须几乎同一句描述”筛选。</p><p id="count1"></p><div id="stage1" class="cards"></div></section><section class="panel"><h2>② 不同画风候选</h2><p class="muted">同人物变体和同画风不会进入此列表。以下为候选，实际发送仍须结合用户当时的聊天内容。</p><p id="count2"></p><div id="stage2" class="cards"></div></section></div>
<section class="panel"><h2>排除依据</h2><div id="audit" class="audit"></div><details><summary>查看被排除的同类图片</summary><div id="rejected" class="cards" style="margin-top:14px"></div></details></section>
<section class="panel"><p>这是公开素材研究预览，包含 1,250 张图片。本轮训练与检索使用本机计算，没有新增付费视觉 API 调用。正式被试图库和聊天选图模型尚未替换。</p></section></main><footer>保留 v2／v3／v4 训练记录 · 同类候选≠已证明同义 · 风格标签仍需核对</footer><script id="data" type="application/json">__DATA__</script><script>__JS__</script></body></html>'''
html=html.replace('__CSS__',css).replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('<','\\u003c')).replace('__JS__',(ROOT/'static/semantic_broad.js').read_text())
(web/'report-template.html').write_text(html);(web/'index.html').write_text(html.replace('__ASSET_BASE__','./'))
for name in ('training-report.json','evaluation-report.json','protocol.json','categories.json','two-stage-results.json'):
    if name=='training-report.json':
        (web/name).write_text(json.dumps(data['report'],ensure_ascii=False,indent=2))
    else:shutil.copy2(OUT/name,web/name)
print(json.dumps({'web':str(web),'items':len(rows)}))
