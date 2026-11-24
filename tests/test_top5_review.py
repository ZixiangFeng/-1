import unittest
from research.sticker_encoder.top5_review import summarize
class ReviewTests(unittest.TestCase):
 def test_unknown_not_counted_as_negative(self):
  rankings={'q':{'m':['1','2','3','4','5']}};ratings={'1':{'meaning':'yes'},'2':{'meaning':'no'},'3':{'meaning':'no'}}
  r=summarize(rankings,ratings)['m'];self.assertEqual(r['precision_at_5_all_query_bounds'],[.2,.6]);self.assertEqual(r['hit_at_5_all_query_bounds'],[1.,1.]);self.assertIsNone(r['precision_at_5_complete_queries']);self.assertEqual(r['judgment_coverage'],.6)
 def test_multiple_correct_candidates_are_supported(self):
  r=summarize({'q':{'m':['1','2','3','4','5']}},{str(i):{'meaning':'yes' if i<=3 else 'no'} for i in range(1,6)})['m'];self.assertEqual(r['precision_at_5_complete_queries'],.6);self.assertIsNone(r['recall_at_5'])
 def test_empty_review_cannot_create_accuracy(self):
  r=summarize({'q':{'m':['1','2','3','4','5']}},{})['m'];self.assertEqual(r['precision_at_5_all_query_bounds'],[0.,1.]);self.assertEqual(r['fully_judged_queries'],0)
if __name__=='__main__':unittest.main()
