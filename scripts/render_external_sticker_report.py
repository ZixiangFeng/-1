"""Build the private image-level holdout report and its public, image-free summary."""
import csv
import html
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sticker_intent import NAMES

OUT = ROOT / 'outputs/sticker-external-holdout-20261119'
WEB = ROOT / 'outputs/sticker-context-v6/web'
LABELS = ROOT / 'research/sticker_encoder/external'
PREVIEW = 'https://www.hustdesign.top/xiaoyu/research/semantic-trained'
CSS = """
*{box-sizing:border-box}html{background:#f2f5ef;color:#233b32;font:16px/1.65 system-ui,"PingFang SC",sans-serif}
body{max-width:1080px;margin:auto;padding:36px 24px 70px}h1{font-size:clamp(27px,4vw,42px);line-height:1.2}
h2{font-size:22px;line-height:1.35;margin:0 0 12px}h3{font-size:17px;margin:14px 0 6px}
section{background:white;padding:clamp(16px,3vw,28px);border:1px solid #dae3d9;border-radius:14px;margin:16px 0}
p{margin:8px 0}small,figcaption{color:#4a6156}a{color:#176449}td,th{border-bottom:1px solid #dbe4dd;padding:8px 10px;text-align:left;vertical-align:top}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}summary{cursor:pointer;font-weight:650}
.scroll{overflow:auto}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(205px,1fr));gap:10px;margin-top:14px}
figure{margin:0;border:1px solid #dce6df;border-radius:8px;padding:12px}img{width:80px;height:74px;object-fit:contain;background:#f4f6f2}
code{font-size:.88em;word-break:break-all}@media(max-width:640px){body{padding:18px 12px 42px}td,th{padding:6px;font-size:14px}}
"""


def load(name):
    return json.loads((OUT / (name + '.json')).read_text())


def esc(value):
    return html.escape(str(value))


def table(headers, rows):
    head = ''.join('<th scope="col">' + esc(x) + '</th>' for x in headers)
    body = ''.join('<tr>' + ''.join('<td>' + esc(x) + '</td>' for x in row) + '</tr>' for row in rows)
    return '<div class="scroll"><table><thead><tr>' + head + '</tr></thead><tbody>' + body + '</tbody></table></div>'


def section(title, body):
    return '<section><h2>' + esc(title) + '</h2>' + body + '</section>'


def photo(folder, identifier, suffix):
    return f'<img loading="lazy" alt="{esc(identifier)}" src="{folder}/{esc(identifier)}.{suffix}">'


def classify_pair(choice):
    if not choice:
        return '无候选'
    if choice['intent'] == 'yes' and choice['style'] == 'yes':
        return '同用意且画风有别'
    if 'no' in (choice['intent'], choice['style']):
        return '未满足要求'
    return '用意或画风难判断'


def main():
    result = load('results')
    known = load('known-group-test')
    bank = load('gallery-evaluation')
    manifest = load('manifest')
    automatic = bank['divergent']
    semantic = bank['semantic']
    total = result['images']
    winners = {r['query']: r for r in automatic['choices']}
    primaries = {r['query']: r for r in semantic['choices']}
    outcomes = [classify_pair(r) for r in winners.values()]
    n, w = known['cases'], known['with_winner']
    turns = table(
        ['给定条件', '观察到的结果'],
        [('先给定同一聊天用意', f"{known['unique_queries']} 张参照图构成 {n} 个「图片 × 用意」组合。{w} 个有合格候选，{n-w} 个留空。"),
         ('比较全部合格图', f'{w}/{w} 次选到了整组风格距离最大者。'),
         ('每组重排十次', f"{known['stable_checks']}/{known['stable_checks']} 次输出一致，含 {known['nonempty_stable_checks']} 次有首选图的比较。"),
         ('按独立画风标签复核', f"{known['visual_contrast']}/{w} 对的视觉媒介不同。视觉媒介由 Codex 编码。")])
    core = section('同义组已知时，能否稳定选出差别最大的一张？', turns)
    core += section('程序自己判定同义时，结果如何？', table(['检验步骤', '结果'], [
        ('新图查原有 1,250 张图', f"{total} 张新图分别检索一次，沿用公开描述模拟使用，没有真实对话上下文。"),
        ('第一步，自动语义组第一名', f"{semantic['returned']} 张有首选，其中 {semantic['meaning_accepted']} 对用意相合，{semantic['unclear_meaning']} 对难判断。"),
        ('第二步，最大风格差异图', f"{automatic['returned']} 张有首选，其中 {outcomes.count('同用意且画风有别')} 对同时满足两项，{outcomes.count('未满足要求')} 对不合格，{outcomes.count('用意或画风难判断')} 对难判断。其余 {total-automatic['returned']} 张留空。")])
        + '<p>部分错误来自否定语义。例如，开心大笑和配字「笑不出来」的图可以在视觉上相差很大，却很难在同一句话中互换。</p>')
    core += section('如何选定那一张图', '<p>先确定这轮要表达的用意，如收到、安慰或无语，再在这个用意的图库中逐张比较。参照图来自用户在聊天中实际使用过的表情。同一张公共图可以获准用于多个用意。</p>'
        '<p>风格距离取 <b>1 − 两个 ALADIN-ViT 风格向量的余弦相似度</b>。同一大类画风也会参与比较。排除用户自有图、重复图和能识别的同模板后，逐张比较原始距离，并列时以图片编号决定。最大距离低于 0.65、所需数据不完整或聊天语境核对未通过时，就只回文字。公开预览里可分别查看自动组和自行核准组。</p>'
        '<p>E0264 已按研究者反馈改为「无语、无奈地干笑」。两句表达无语的对话都允许这张图进入候选，「和朋友一起真心大笑」的对话将它排除了。该已知纠错样本单独复测。</p>')
    src = '、'.join(f'<a href="https://github.com/{esc(repo)}/tree/{esc(rev)}">{esc(repo)}</a>' for repo, rev in manifest['sources'].values())
    core += section('测试材料与判断方式', f'<p>使用 {result["sources"]["meme"]} 张社区表情与 {result["sources"]["platform"]} 张平台表情。固定来源版本后，逐图检查文件、解码像素及感知哈希，并与原有 {manifest["checked_project_images"]:,} 张图片去重。取样专门纳入了真笑、苦笑和无语，结果只描述本批样本。素材来源为 {src}。</p>'
        '<p>Codex 在看到模型分数之前为新图标注可表达的用意。另将第一步和第二步首选合并去重，打乱编号后看图核对。以上判断来自机器复核，真实被试对语境与画风的感受尚待预实验验证。首次运行排除了相同媒介，按整组取最大值的要求取消了这层排除，在同一批测试图上复算。沿用已核对的图对，只对新增图对做盲评。前后两次记录均保留，语义权重、标注和阈值均在初次测试前锁定。</p>')
    start = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
             '<title>小屿 · 77 张新图的趋异选图测试</title><style>' + CSS + '</style><main>'
             f'<nav><a href="{PREVIEW}?id=E0264">返回趋异选图测试</a></nav><h1>77 张新图的趋异选图测试</h1>'
             '<p>2026 年 11 月 19 日 · 公开素材 · 未用于训练</p>')
    (WEB / 'external-test.html').write_text(start + core + '</main></html>')

    comparison = OUT / 'comparison-images'
    comparison.mkdir(exist_ok=True)
    cards = []
    with (OUT / '逐图结果.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['新图编号', '独立看图释义', '自动识别用意', '语义首选编号', '最大风格差异编号', '距离', '两项核对', '核对备注'])
        for r in result['sample']:
            key = r['id']
            choice = winners.get(key, {})
            top = primaries.get(key, {}).get('id', '')
            winner = choice.get('id', '')
            for candidate in {top, winner} - {''}:
                shutil.copy2(WEB / 'images' / (candidate + '.jpg'), comparison / (candidate + '.jpg'))
            group = NAMES.get(r['predicted'], '用意未确定')
            status = classify_pair(choice)
            note = choice.get('note', '')
            writer.writerow([key, r['visual_read'], group, top, winner, choice.get('style_distance', ''), status, note])
            pair = photo('images', key, 'png')
            if winner:
                pair += photo('comparison-images', winner, 'jpg')
            cards.append(f'<figure><b>{key}</b> · {esc(group)}<div>{pair}</div><figcaption>{esc(r["visual_read"])}<br>'
                         f'语义首选 {esc(top or "—")} · 风格首选 {esc(winner or "—")}<br>{esc(status)}。{esc(note)}</figcaption></figure>')
    local = section('核对每张图', '<p>每组左侧为新图，右侧为从原有图库选出的最大风格差异图。「无候选」表示程序留空。</p>'
                    '<a href="逐图结果.csv">下载逐图结果 CSV</a> · <a href="known-group-test.json">查看已知同义组的全部 60 次首选与距离</a>'
                    '<div class="grid">' + ''.join(cards) + '</div>')
    (OUT / '测试结果.html').write_text(start + core + local + '</main></html>')
    for name in ['visual-judgments.tsv', 'gallery-pair-judgments.tsv', 'gallery-pair-judgments-full-group.tsv', 'context-checks.json']:
        shutil.copy2(LABELS / name, OUT / name)
    archive = OUT.parent / '77张表情包_独立测试与结果.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for item in sorted(OUT.rglob('*')):
            if item.is_file() and not item.name.startswith('.'):
                z.write(item, str(item.relative_to(OUT)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and '测试结果.html' in z.namelist()


if __name__ == '__main__':
    main()
