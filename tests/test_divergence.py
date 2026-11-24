import copy
import json
import unittest

import divergence
import stickers
import test_screenshot_library as fixture


def profile(medium, subject, text, palette, description):
    # A short verbatim visible description grounds every tested field.
    return {'medium':medium,'subject':subject,'text_mode':text,'palette':palette,
            'evidence':{k:description for k in divergence.VALUES}}


def choice(aid, ref, desc_a, desc_b, a=None, b=None):
    return {'asset_id':aid,'reference_asset_id':ref,'reason':'同为确认收到，使用不同形式',
            'semantic_fit':True,'template_relation':'different',
            'reference_profile':a or profile('photo','animal','caption','color',desc_a),
            'candidate_profile':b or profile('line_drawing','human','none','monochrome',desc_b)}


class StrongD(unittest.TestCase):
    def test_reported_cartoon_animal_to_photo_animal_and_cat_to_duck_are_weak(self):
        for first in ('cartoon','photo'):
            a='彩色卡通猫，无文字' if first=='cartoon' else '彩色写实猫，无文字'
            b='彩色写实鸭，短配字爽翻了'
            result=divergence.evaluate(choice('d','u',a,b,
                profile(first,'animal','none','color',a),profile('photo','animal','caption','color',b)),
                {'asset_id':'u','visible_description':a},{'asset_id':'d','description':b})
            self.assertFalse(result['passed']);self.assertLess(result['score'],5)

    def test_strong_form_change_requires_semantics_grounding_and_two_axes(self):
        a='彩色写实猫，短配字收到';b='黑白简笔小人，无文字'
        selected=choice('d','u',a,b);ref={'asset_id':'u','visible_description':a};cand={'asset_id':'d','description':b}
        result=divergence.evaluate(selected,ref,cand)
        self.assertTrue(result['passed']);self.assertEqual(result['score'],8)
        bad=copy.deepcopy(selected);bad['semantic_fit']=False
        self.assertFalse(divergence.evaluate(bad,ref,cand)['passed'])
        for key in ('reference_profile','candidate_profile'):
            bad=copy.deepcopy(selected);bad[key]['evidence']['medium']='虚构的图片内容'
            self.assertFalse(divergence.evaluate(bad,ref,cand)['passed'])
        selected['reference_profile']['medium']='unknown'
        self.assertFalse(divergence.evaluate(selected,ref,cand)['passed'])

    def test_style_diversity_can_reach_beyond_first_twelve_semantic_hits(self):
        options=[{'asset_id':str(i),'style':'写实动物','usages':[]} for i in range(25)]
        options.append({'asset_id':'line','style':'黑白简笔人物','usages':[]})
        result=divergence.diverse_candidates(options)
        self.assertEqual(result[1]['asset_id'],'line');self.assertEqual(len(result),18)
        self.assertEqual(len({o['asset_id'] for o in result}),18)

    def test_ambiguous_legacy_text_metadata_cannot_add_a_point(self):
        a='彩色卡通猫，无文字';b='彩色写实猫，少量或无文字'
        selected=choice('d','u',a,b,profile('cartoon','animal','none','color',a),
                        profile('photo','animal','caption','color',b))
        result=divergence.evaluate(selected,{'asset_id':'u','visible_description':a},
                                   {'asset_id':'d','description':b})
        self.assertEqual(result['score'],3);self.assertFalse(result['passed'])

    def test_realistic_human_to_monochrome_line_human_is_strong(self):
        a='彩色人物摄影，无字';b='黑白简笔人物，无字'
        selected=choice('d','u',a,b,profile('photo','human','none','color',a),
                        profile('line_drawing','human','none','monochrome',b))
        result=divergence.evaluate(selected,{'asset_id':'u','visible_description':a},
                                   {'asset_id':'d','description':b})
        self.assertTrue(result['passed']);self.assertEqual(result['score'],5)

    def test_enumerated_literal_excerpts_and_independent_axes(self):
        a='手绘风格的蓝灰色卡通动物，没有显示可见文字。'
        b='线条简笔画小人，脸颊带着粉晕，两侧有粉红色爱心，上方红色文字“好感+∞”。'
        selected=choice('d','u',a,b,profile('cartoon','animal','none','color',a),
                        profile('line_drawing','human','caption','color',b))
        selected['candidate_profile']['evidence']['palette']='粉红色爱心、红色文字、粉晕'
        ref={'asset_id':'u','visible_description':a};cand={'asset_id':'d','description':b}
        result=divergence.evaluate(selected,ref,cand)
        self.assertTrue(result['passed']);self.assertEqual(result['ignored_evidence'],[])
        selected['candidate_profile']['evidence']['palette']='没有依据的色彩'
        result=divergence.evaluate(selected,ref,cand)
        self.assertTrue(result['passed']);self.assertEqual(result['score'],5)
        self.assertEqual(result['ignored_evidence'][0]['axis'],'palette')
        selected['candidate_profile']['evidence']['medium']='不存在的摄影风格'
        self.assertFalse(divergence.evaluate(selected,ref,cand)['passed'])
        self.assertFalse(divergence.grounded('粉红色爱心、凭空捏造',b))


class StrongDIntegration(unittest.TestCase):
    event=fixture.ScreenshotLibrary.event
    tearDown=fixture.ScreenshotLibrary.tearDown
    def setUp(self):
        fixture.ScreenshotLibrary.setUp(self)
        self.a='彩色写实猫，短配字收到';self.b='黑白简笔小人，无文字'
        self.c.execute('INSERT INTO assets VALUES(?,?,?,?,?)',('used','a','private',self.a,'{}'))
        self.c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?)',('source','a','s','user','收到','used'))
        self.c.execute('UPDATE generic_stickers SET semantics=? WHERE id=?',
                       (json.dumps(dict(description=self.b,meaning='收到',situations='回应提醒',avoid='危机')),'d_one'))
        self.c.execute("UPDATE sticker_library_metadata SET style='黑白简笔人物' WHERE asset='d_one'")
        self.c.execute("INSERT INTO sticker_intent_reviews VALUES('d_one','[\"ack\"]','fixture')")

    def select(self, data, current='used', options=None):
        return stickers.choose(self.c,'a','s','D','收到','好呀',current,
            lambda *a,**k:(json.dumps(data),{}),rng=lambda n:0,options=options)

    def test_delivery_and_audit_then_rejection_does_not_write_learning(self):
        selected=choice('d_one','used',self.a,self.b)
        aid,audit=self.select(selected)
        self.assertEqual(aid,'d_one');self.assertTrue(audit['contrast']['passed'])
        self.assertEqual(audit['contrast_policy'],divergence.POLICY)
        selected['semantic_fit']=False
        aid,audit=self.select(selected)
        self.assertIsNone(aid);self.assertIsNone(audit['asset_id']);self.assertEqual(audit['rejected_asset_id'],'d_one')
        self.assertEqual(self.c.execute('SELECT count(*) FROM memories').fetchone()[0],0)

    def test_cannot_use_an_old_reference_to_evade_current_image_and_text_turn_can(self):
        self.c.execute('INSERT INTO assets VALUES(?,?,?,?,?)',('now','a','new',self.b,'{}'))
        self.c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?)',('newsource','a','s','user','收到','now'))
        selected=choice('d_one','used',self.a,self.b)
        self.assertIsNone(self.select(selected,current='now')[0])
        self.assertIsNone(self.select(selected,current=None)[0])
        self.c.execute("UPDATE messages SET text='晚安，我先去睡了' WHERE id='newsource'")
        self.assertEqual(self.select(selected,current=None)[0],'d_one')

    def test_precomputed_candidates_cannot_bypass_strong_gate(self):
        selected=choice('d_one','used',self.a,self.b)
        selected['candidate_profile']=profile('photo','animal','caption','color',self.b)
        aid,audit=self.select(selected,options=stickers.candidates(self.c,'a','s','D'))
        self.assertIsNone(aid);self.assertEqual(audit['contrast']['score'],0)

    def test_empty_choice_does_not_require_unused_profiles_or_retry(self):
        for empty in ('',None):
            calls=[]
            def generate(*a,**kw):
                calls.append(1);return json.dumps({'asset_id':empty,'reason':'严肃话题不适合现有图片'}),{}
            aid,audit=stickers.choose(self.c,'a','s','D','收到','好呀','used',generate,rng=lambda n:0)
            self.assertIsNone(aid);self.assertEqual(len(calls),1);self.assertNotIn('failed',audit)
            self.assertTrue(audit['probability_passed'])

    def test_one_bounded_repair_keeps_same_random_draw_and_evidence_gate(self):
        valid=choice('d_one','used',self.a,self.b)
        bad=copy.deepcopy(valid);bad['candidate_profile']['evidence']['medium']='不存在的描述'
        outputs=iter([json.dumps(bad),json.dumps(valid)]);calls=[];draws=[]
        def generate(*a,**kw):calls.append(1);return next(outputs),{'attempt':len(calls)}
        aid,audit=stickers.choose(self.c,'a','s','D','收到','好呀','used',generate,
                                 rng=lambda n:(draws.append(n) or 0))
        self.assertEqual(aid,'d_one');self.assertEqual(len(calls),2);self.assertEqual(len(draws),1)
        self.assertEqual(len(audit['selection_attempts']),2);self.assertTrue(audit['contrast']['passed'])
        self.assertNotIn('rejected_asset_id',audit)

    def test_malformed_output_repair_and_failure_are_logged(self):
        calls=[]
        def bad(*a,**kw):calls.append(1);return '{broken',{}
        aid,audit=stickers.choose(self.c,'a','s','D','收到','好呀','used',bad,rng=lambda n:0)
        self.assertIsNone(aid);self.assertEqual(len(calls),2);self.assertTrue(audit['failed'])
        self.assertEqual(audit['error_stage'],'selector_output')
        self.assertEqual([x['asset_id'] for x in audit['candidates']],['d_one']);self.assertTrue(audit['probability_passed'])
        def unavailable(*a,**kw):raise TimeoutError('private upstream detail')
        aid,audit=stickers.choose(self.c,'a','s','D','收到','好呀','used',unavailable,rng=lambda n:0)
        self.assertEqual(audit['error_type'],'TimeoutError');self.assertNotIn('private upstream detail',str(audit))

    def test_weak_first_pick_can_be_corrected_once_but_semantic_rejection_is_final(self):
        valid=choice('d_one','used',self.a,self.b)
        weak=copy.deepcopy(valid);weak['candidate_profile']=profile('photo','animal','caption','color',self.b)
        outputs=iter([json.dumps(weak),json.dumps(valid)]);calls=[]
        def generate(*a,**kw):calls.append(1);return next(outputs),{}
        aid,audit=stickers.choose(self.c,'a','s','D','收到','好呀','used',generate,rng=lambda n:0)
        self.assertEqual(aid,'d_one');self.assertEqual(len(calls),2)
        self.assertEqual(audit['selection_attempts'][0]['contrast']['score'],0)
        invalid=copy.deepcopy(valid);invalid['semantic_fit']=False;calls=[]
        def inappropriate(*a,**kw):calls.append(1);return json.dumps(invalid),{}
        self.assertIsNone(stickers.choose(self.c,'a','s','D','收到','好呀','used',inappropriate,rng=lambda n:0)[0])
        self.assertEqual(len(calls),1)


if __name__=='__main__':unittest.main()
