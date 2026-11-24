"""Export the frozen StickerQueries review as an offline reader and an audit bundle."""
import base64
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import sticker_intent
import sticker_pragmatics
from research.sticker_encoder.reviewed_gallery import reviewed_rows
from scripts.collect_stickerqueries_holdout import CORPUS, HOLDOUT, RUN, read

HTML = RUN / '新测试集逐图对照.html'
TABLE = RUN / '逐图结果.csv'
ARCHIVE = RUN / '100张独立测试集与结果.zip'
GALLERY = ROOT / 'outputs/sticker-context-v6/web/images'
STAGES = {'semantic': '第一阶段 · 用意最相近', 'divergent': '第二阶段 · 组内画风最远'}
VERDICTS = {'yes': '是', 'no': '否', 'uncertain': '待定'}


def thumbnail(path):
    with Image.open(path) as image:
        image = image.convert('RGB')
        image.thumbnail((240, 200), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format='WEBP', quality=84, method=4)
    return 'data:image/webp;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')


def empty_reason(row):
    if not row['group']:
        return '用意未能归组'
    if not row['interchangeable']:
        return '归组后没有通过语义筛选的图'
    return '组内没有达到风格门槛的图'


def outcome(stage, rating):
    if not rating:
        return '未出图'
    if rating['intent'] == 'no':
        return '用意不符'
    if rating['intent'] == 'uncertain':
        return '用意待定'
    if stage == 'semantic':
        return '用意相同'
    if rating['style'] == 'yes' and rating['same_template'] == 'no':
        return '同义且异风'
    return '画风或模板未通过'


def build_data():
    manifest = read(HOLDOUT / 'manifest.json')['rows']
    gallery = {r['id']: r for r in reviewed_rows(read(CORPUS)['rows'])}
    detailed = read(RUN / 'evaluated-queries.json')
    lookup = {(r['query'], r['method']): r for r in detailed}
    assert len(detailed) == len(lookup) == 2 * len(manifest)
    names = {'8': sticker_pragmatics.NAMES, '27': sticker_intent.NAMES}
    labels = dict(sticker_pragmatics.NAMES, outside='八类之外或仅描述外观', uncertain='用意待定')
    sources, candidates, table = [], {}, []
    for source in manifest:
        qid = source['id']
        query = {
            'id': qid, 'phrase': source['source_caption'],
            'image': thumbnail(HOLDOUT / 'images' / (qid + '.png')), 'methods': {},
        }
        for method, group_names in names.items():
            row = lookup[qid, method]
            query['reference'] = ' / '.join(labels[key] for key in row['reference_family'].split('|'))
            group_name = group_names.get(row['group'], '未归组')
            prediction = dict(group=group_name, family=row['family_candidates'],
                              semantic_count=row['interchangeable'], style_count=row['style_eligible'])
            for stage, stage_name in STAGES.items():
                top = row[stage]
                rating = row[stage + '_rating']
                status = outcome(stage, rating)
                if top:
                    cid = top['id']
                    if cid not in candidates:
                        candidates[cid] = {
                            'image': thumbnail(GALLERY / (cid + '.jpg')), 'caption': gallery[cid]['meaning'],
                        }
                    note = rating['observation']
                else:
                    cid = ''
                    note = empty_reason(row)
                prediction[stage] = dict(status=status, note=note, candidate=cid)
                if top:
                    prediction[stage].update(score=top['meaning_score'], distance=top['style_distance'],
                                             blind=rating['blind_id'])
                table.append({
                    '测试图': qid, '原始中文搜索词': query['phrase'], '原始样本ID': source['source_id'],
                    '源图片链接': source['url'], '看图预判大类': query['reference'], '版本': method + '类',
                    '算法分组': group_name, '大类候选数': row['family_candidates'],
                    '语义筛选通过数': row['interchangeable'], '风格筛选通过数': row['style_eligible'],
                    '阶段': stage_name, '首选图': cid, '图库释义': candidates[cid]['caption'] if cid else '',
                    '语义分': top['meaning_score'] if top else '', '风格距离': top['style_distance'] if top else '',
                    '盲评编号': rating['blind_id'] if rating else '', '用意相同': VERDICTS[rating['intent']] if rating else '',
                    '画风明显不同': VERDICTS[rating['style']] if rating else '',
                    '同一模板': VERDICTS[rating['same_template']] if rating else '', '结果': status, '逐对核对说明': note,
                })
            query['methods'][method] = prediction
        sources.append(query)
    return sources, candidates, table


PAGE = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>小屿 · 100 张新图的趋异检索测试</title>
<style>
:root{color-scheme:light;--ink:#192f30;--sub:#526865;--line:#cbd8d3;--green:#176453;--paper:#f1f5f2}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 Inter,"PingFang SC","Microsoft YaHei",sans-serif}
main{max-width:1130px;padding:40px 24px 88px;margin:auto}h1{font-size:clamp(27px,4vw,40px);margin:0 0 8px;letter-spacing:-.03em}h2{font-size:22px;margin:28px 0 12px}
p{margin:6px 0 14px}a{color:var(--green)}.sub,small{color:var(--sub)}.lead{font-size:19px;max-width:850px}
.panel,article,details.protocol{background:#fff;border:1px solid var(--line);border-radius:14px} .panel{padding:16px 20px;overflow:auto}
table{width:100%;border-collapse:collapse;white-space:nowrap;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:10px 8px;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}
tbody tr:last-child td{border:0}th{font-size:14px;color:var(--sub)}.method,summary{font-weight:650}
details.protocol{padding:12px 17px;margin:12px 0}summary{cursor:pointer}details p:last-child{margin-bottom:4px}
.controls,.pager{display:flex;align-items:end;gap:12px;flex-wrap:wrap}.controls label{display:grid;gap:4px;color:var(--sub);font-size:14px}
input,select,button{padding:8px 10px;border-radius:7px;border:1px solid #a9bbb2;background:white;color:var(--ink);font:inherit;min-height:40px}
button{cursor:pointer}button:disabled{opacity:.4;cursor:default}input:focus-visible,select:focus-visible,button:focus-visible,a:focus-visible,summary:focus-visible{outline:3px solid #368777;outline-offset:2px}
.pager{align-items:center;justify-content:space-between;margin:14px 0}.pager div{display:flex;gap:6px}
article{margin:10px 0 16px;padding:16px}.case-head{display:flex;align-items:baseline;gap:12px;border-bottom:1px solid var(--line);padding-bottom:8px}
.case-head strong{font-size:20px}.case-head span,.label{color:var(--sub);font-size:13px}
.grid{display:grid;grid-template-columns:.8fr 1fr 1fr;gap:18px;margin-top:14px}.cell{min-width:0}.label{display:block;margin-bottom:4px}
.imagebox{height:148px;background:#f5f6f3;border-radius:7px;display:flex;align-items:center;justify-content:center;margin:7px 0}
img{display:block;max-width:100%;max-height:100%;object-fit:contain}.note{font-size:15px;margin:4px 0}.judgment{font-weight:680}.good{color:#075e44}.bad{color:#962f27}.wait{color:#805210}
.cell details{color:var(--sub);font-size:13px;line-height:1.6;margin-top:5px}.cell summary{font-weight:500}#empty{padding:24px 0}footer{margin-top:25px;font-size:14px;color:var(--sub)}
@media(max-width:650px){main{padding:20px 12px 42px}.panel,article{padding:12px}.grid{grid-template-columns:1fr 1fr;gap:8px 12px}.source{grid-column:1/-1;display:grid;grid-template-columns:90px 1fr;column-gap:10px;align-items:center}.source .label{grid-column:2;grid-row:1}.source .imagebox{grid-column:1;grid-row:1/3;height:80px}.source small{grid-column:2;grid-row:2}.imagebox{height:124px}th,td{padding:7px}.controls{gap:7px}.controls input{max-width:160px}}
</style>
<main>
<h1>100 张新图的趋异检索测试</h1>
<p class="lead" id="lead"></p>
<p class="sub">从 StickerQueries 中文静态表情中固定抽样，每张带一个原始人工搜索词。两版使用同一批参考图和同一个 1,250 张候选图库。</p>
<h2>从理解用意到选出不同画风</h2>
<div class="panel"><table><thead><tr><th>检索版本</th><th>语义首选同义 / 有首选</th><th>趋异出图 / 测试图</th><th>趋异合格 / 出图</th><th>用意不符</th><th>用意待定</th></tr></thead><tbody id="counts"></tbody></table></div>
<p class="sub" id="stops"></p>
<details class="protocol"><summary>抽样与判定口径</summary>
<p>测试前固定模型、门槛和样本。用文件、像素哈希及包含镜像和裁边的感知哈希，对照原图库和三批历史测试记录。查询时把数据集的中文搜索词作为用意输入，把图片用于 ALADIN 风格计算。97 个搜索词不超过五个字符。</p>
<p>新版先归入 8 个大类，再逐对筛选可互换的图片；旧版按 27 类检索。两个版本都在符合各自语义条件、通过去重和风格门槛的完整组内取风格距离最大的图片，两版各核查了 100 次。结果差异反映两套完整规则的差别。</p>
<p>将两版在两个阶段给出的首选图对去重、打乱后，Codex 逐对评阅了 91 对图。评阅时只展示原图、原搜索词和候选图，隐藏版本、排名和模型分数。合格需要两张图表达同一用意、画风明显不同，并排除明显复用同一模板的情况。未出图与用意待定单独统计。</p>
<p>原作者提供的是人们会用来找这张图的词。候选图的可替换性由本次单一评阅者判定。样本没有参与训练或调参，也没有真实聊天上下文。<a href="https://huggingface.co/datasets/metchee/sticker-queries">原始数据集</a> · <a href="https://doi.org/10.1145/3746027.3758310">研究论文</a></p></details>
<h2 id="review">逐图对照</h2>
<p>可直接看 <a href="#Z001">Z001「委屈巴巴」</a> 和 <a href="#Z060">Z060「笑」</a>，切换阶段查看风格排序前后的候选。</p>
<div class="controls">
<label>检索阶段<select id="stage"><option value="divergent">第二阶段 · 组内画风最远</option><option value="semantic">第一阶段 · 用意最相近</option></select></label>
<label>按新版结果筛选<select id="filter"></select></label>
<label>查原搜索词或编号<input id="search" type="search" placeholder="如 拥抱 或 Z063"></label>
</div>
<div class="pager"><span id="shown" aria-live="polite"></span><div><button id="prev">上一页</button><button id="next">下一页</button></div></div>
<section id="cases"></section><footer>每个编号对应一张独立参考图。图库释义与本次逐对盲评分开显示。</footer>
</main>
<script type="application/json" id="payload">__PAYLOAD__</script>
<script>
const {sources, candidates, metrics} = JSON.parse(document.querySelector('#payload').textContent);
const $ = id => document.getElementById(id);
const safe = text => String(text).replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
const current = metrics.methods['8'], total = metrics.n, pageSize = 10;
let page = 0;
const pct = (n,d) => (100*n/d).toFixed(1)+'%';
$('lead').textContent = '新版在 '+total+' 张上返回 '+current.divergent.returned+' 个趋异首选。盲评中 '+current.divergent.both+' 个同时保留原意、换了明显不同的画风，占出图的 '+pct(current.divergent.both,current.divergent.returned)+'，占全部测试图的 '+pct(current.divergent.both,total)+'。';
for (const key of ['8','27']){
  const {semantic:s,divergent:d}=metrics.methods[key];
  $('counts').insertAdjacentHTML('beforeend','<tr><td class="method">'+(key==='8'?'新版 · 8 类':'旧版 · 27 类')+'</td><td>'+s.same+' / '+s.returned+'</td><td>'+d.returned+' / '+total+'</td><td>'+d.both+' / '+d.returned+'（'+pct(d.both,d.returned)+'）</td><td>'+d.different+'</td><td>'+d.unclear+'</td></tr>');
}
$('stops').textContent='新版有 '+(total-current.divergent.returned)+' 张没有出图。其中 '+current.unassigned+' 张未归组，'+current.no_semantic_match_after_classification+' 张归组后没有通过语义筛选的图，'+current.no_style_match_after_semantics+' 张卡在风格门槛。';
function filters(){
  const list=$('stage').value==='divergent'?['同义且异风','用意不符','用意待定','画风或模板未通过','未出图']:['用意相同','用意不符','用意待定','未出图'];
  $('filter').replaceChildren(...['全部',...list].map(x=>new Option(x,x)));
}
function cell(row,method,stage){
  const p=row.methods[method], t=p[stage], name=method==='8'?'新版 · 8 类':'旧版 · 27 类';
  const a=candidates[t.candidate], status=t.status;
  const color=['同义且异风','用意相同'].includes(status)?'good':status==='用意不符'?'bad':status==='用意待定'?'wait':'';
  const media=a?'<img alt="'+safe(t.candidate)+' 候选图" src="'+a.image+'">':'<small>未出图</small>';
  const tech=a?'<details><summary>'+safe(t.candidate)+' · 查看分数与图库释义</summary><div>语义分 '+t.score.toFixed(3)+' · 风格距离 '+t.distance.toFixed(3)+'</div><div>图库释义 · '+safe(a.caption)+'</div><div>盲评 '+t.blind+'</div></details>':'';
  return '<div class="cell"><span class="label">'+name+' · '+safe(p.group)+'</span><div class="imagebox">'+media+'</div><div class="judgment '+color+'">'+safe(status)+'</div><p class="note">'+safe(t.note)+'</p>'+tech+'</div>';
}
function render(){
  const stage=$('stage').value, status=$('filter').value, term=$('search').value.trim().toLowerCase();
  const rows=sources.filter(q=>(status==='全部'||q.methods['8'][stage].status===status)&&(!term||(q.id+' '+q.phrase).toLowerCase().includes(term)));
  const pages=Math.max(1,Math.ceil(rows.length/pageSize)); page=Math.min(page,pages-1);
  $('shown').textContent=rows.length+' 张 · 第 '+(page+1)+' / '+pages+' 页'; $('prev').disabled=page===0;$('next').disabled=page+1>=pages;
  $('cases').innerHTML=rows.slice(page*pageSize,(page+1)*pageSize).map(q=>'<article data-query="'+q.id+'"><div class="case-head"><span>'+q.id+'</span><strong>'+safe(q.phrase)+'</strong></div><div class="grid"><div class="cell source"><span class="label">原测试图 · 人工搜索词</span><div class="imagebox"><img alt="'+q.id+' 原图" src="'+q.image+'"></div><small>预先看图分组 · '+safe(q.reference)+'</small></div>'+cell(q,'8',stage)+cell(q,'27',stage)+'</div></article>').join('')||'<p id="empty">没有符合当前条件的测试图。</p>';
}
function jump(){
  const key=location.hash.slice(1).toUpperCase();
  if(!sources.some(q=>q.id===key))return;
  $('filter').value='全部'; $('search').value=key;page=0;render();$('review').scrollIntoView();
}
$('stage').onchange=()=>{filters();page=0;render()};
$('filter').onchange=$('search').oninput=()=>{page=0;render()};
$('prev').onclick=()=>{page--;render()};
$('next').onclick=()=>{page++;render()};
for(const link of document.querySelectorAll('a[href^="#Z"]')){
  link.addEventListener('click',()=>{if(location.hash===link.hash)jump()});
}
window.addEventListener('hashchange',jump);filters();render();jump();
</script></html>'''


def export():
    sources, candidates, table = build_data()
    metrics = read(RUN / 'evaluation.json')
    with TABLE.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    payload = json.dumps(dict(sources=sources, candidates=candidates, metrics=metrics),
                         ensure_ascii=False, separators=(',', ':')).replace('<', r'\u003c')
    HTML.write_text(PAGE.replace('__PAYLOAD__', payload), encoding='utf-8')
    records = ['protocol.json', 'review-protocol.json', 'holdout/manifest.json', 'before-prediction.json',
               'query-family-review.tsv', 'prediction-lock.json', 'blind-key.json', 'pair-review.tsv',
               'rankings.json', 'evaluation.json', 'evaluated-queries.json', TABLE.name, HTML.name]
    with zipfile.ZipFile(ARCHIVE, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
        for name in records:
            archive.write(RUN / name, name)
        for item in sources:
            name = 'holdout/images/' + item['id'] + '.png'
            archive.write(RUN / name, name)
    print(f'{len(sources)} queries, {len(table)} stage-level rows, {len(candidates)} candidate thumbnails.')


if __name__ == '__main__':
    export()
