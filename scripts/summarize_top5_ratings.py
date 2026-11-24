"""Summarize one exported rater file; AI and human judgments remain separate."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.sticker_encoder.top5_review import summarize
from research.sticker_encoder.weak_model import digest,save
p=argparse.ArgumentParser();p.add_argument('ratings');p.add_argument('--run',default='outputs/sticker-semantic-v3-local-20261118');p.add_argument('--output',required=True);args=p.parse_args()
run=Path(args.run);ratings=json.loads(Path(args.ratings).read_text());key=json.loads((run/'top5-researcher-key.json').read_text())
if ratings.get('schema')!='xiaoyu-top5-review-v1' or ratings.get('revision')!=digest(run/'checkpoint-lock.json'):raise ValueError('Different run/schema')
if not ratings.get('rater_id') or ratings.get('rater_type') not in ('human','ai'):raise ValueError('Missing rater provenance')
if not set(ratings['ratings'])<=set(key['pairs']):raise ValueError('Unexpected pair IDs')
report={'rater_id':ratings['rater_id'],'rater_type':ratings['rater_type'],'rating_file_sha256':digest(Path(args.ratings)),'groups':{},'note':'One rater only. Unknown judgments retain bounds; complete-query estimates may have selection bias. AI ratings are not human validation.'}
for group,prefix in [('fresh_test','E'),('old_test','B')]:report['groups'][group]=summarize(key['rankings'],ratings['ratings'],[q for q in key['rankings'] if q.startswith(prefix)])
save(Path(args.output),report);print(json.dumps(report,ensure_ascii=False))
