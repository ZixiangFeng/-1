import json, tempfile, unittest
from pathlib import Path
from semantic_training import metrics,calibrate,validate_dataset,paper_pair_candidates

class SemanticTrainingTests(unittest.TestCase):
    def test_known_auc_ties_and_validation_threshold(self):
        v=calibrate([.9,.8,.4,.2],[1,1,0,0]);self.assertEqual(v['threshold'],.8);self.assertEqual(v['f1'],1)
        test=metrics([.9,.7,.85,.2],[1,1,0,0],v['threshold']);self.assertEqual(test['f1'],.5)
        self.assertEqual(metrics([.5,.5],[1,0],.5)['roc_auc'],.5)
    def test_threshold_search_matches_brute_force(self):
        scores=[.1,.1,.3,.9,.4,.4];labels=[0,1,1,0,0,1]
        brute=max([metrics(scores,labels,t) for t in set(scores)],key=lambda m:(m['f1'],m['precision'],m['threshold']))
        self.assertEqual(calibrate(scores,labels),brute)
    def test_bad_evaluation_rejected(self):
        for scores,labels in [([1],[1]),([float('nan'),0],[1,0]),([.2],[1,0])]:
            with self.assertRaises(ValueError):calibrate(scores,labels)
    def test_paper_pair_requires_repeated_selection_and_negative_evidence(self):
        rounds=[{'anchor':'a','candidates':['b','c'],'selected':['b'],'round':n,'annotator':'r'+str(n)} for n in (1,2)]
        result=paper_pair_candidates(rounds,{'a':['yes'],'b':['ok'],'c':['no']},{'a|c':.2,'b|c':.2})
        self.assertEqual({(r['left'],r['right']):r['label'] for r in result},{('a','b'):1,('a','c'):0,('b','c'):0})
        self.assertTrue(all(not r['reviewed'] for r in result))
        self.assertEqual(len(paper_pair_candidates(rounds,{},{})),1)
        with self.assertRaises(ValueError):paper_pair_candidates(rounds+rounds,{}, {})
    def test_split_leakage_machine_labels_and_duplicates_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            for i in range(8):(root/(str(i)+'.png')).write_bytes(bytes([i]))
            train=[{'image':str(i)+'.png','text':'meaning '+str(i),'reviewed':True} for i in range(2)]
            val=[{'left':'2.png','right':str(i)+'.png','label':int(i==3),'reviewed':True} for i in (3,4)]
            test=[{'left':'5.png','right':str(i)+'.png','label':int(i==6),'reviewed':True} for i in (6,7)]
            def save():
                for file,rows in [('train',train),('validation',val),('test',test)]:
                    (root/(file+'.jsonl')).write_text('\n'.join(json.dumps(r) for r in rows))
            save();self.assertEqual(len(validate_dataset(root)[0]),2)
            train[0]['reviewed']=False;save()
            with self.assertRaises(ValueError):validate_dataset(root)
            train[0]['reviewed']=True;val[0]['left']='0.png';save()
            with self.assertRaises(ValueError):validate_dataset(root)
