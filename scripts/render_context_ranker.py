"""Package the public corpus and model outputs; the browser does no reranking."""
import json
import os
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.evaluate_context_ranker import CORPUS, EXAMPLES, OLD, OUT, export_reviewed_rankings, read
from research.sticker_encoder.reviewed_gallery import reviewed_rows
import sticker_pragmatics

web=OUT/'web'
families = export_reviewed_rankings()
if not (web/'images').exists():
    shutil.copytree(OLD/'web/images',web/'images',copy_function=os.link)
for name in ['training-report.json','evaluation-report.json']:
    shutil.copy2(OUT/name,web/name)
release=Path(read(OLD/'deployment-result.json')['assets']).name
items = [{**{key:r[key] for key in ('id','meaning','medium','split','meaning_review') if key in r},
          **families[r['id']]} for r in reviewed_rows(read(CORPUS)['rows'])]
new_run = ROOT/'outputs/sticker-pragmatic-v7'
data = dict(items=items,families=sticker_pragmatics.NAMES,assets='__ASSET_BASE__',contexts=EXAMPLES,
            evaluation=read(OUT/'evaluation-report.json'),sha=read(OUT/'checkpoint-lock.json')['sha256'],
            intent_sha=sticker_pragmatics.model().digest,holdout=read(new_run/'evaluation.json'),
            coverage=read(new_run/'catalog-coverage.json'))
for source, target in [('evaluation.json','eight-family-test.json'),('training.json','eight-family-training.json'),
                       ('catalog-coverage.json','eight-family-coverage.json')]:
    shutil.copy2(new_run/source,web/target)
page=(ROOT/'static/semantic_context.html').read_text().replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('<','\\u003c'))
page=page.replace('__PREVIOUS__','/xiaoyu/research/semantic-trained/assets/'+release+'/report.html')
(web/'report-template.html').write_text(page)
(web/'index.html').write_text(page.replace('__ASSET_BASE__','./'))
