import json
import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch
import temporal

class TemporalGrounding(unittest.TestCase):
    def test_clock_uses_one_timestamp_with_timezone(self):
        utc = datetime(2026,10,12,6,0,tzinfo=timezone.utc)
        _, local, offset = temporal.clock(-480,utc)
        self.assertEqual(local.hour,14)
        self.assertEqual(temporal.local_stamp('2026-10-12T05:55:00+00:00',offset),'2026-10-12 13:55 +0800')
        _, local, _ = temporal.clock(-480,datetime(2026,10,12,18,tzinfo=timezone.utc))
        self.assertEqual(local.day,13)

    def test_semantic_analysis_called_without_keyword_gate(self):
        model=Mock(return_value=(json.dumps({'events':[{'event':'见朋友','evidence':'再过两个钟头见面','event_time':'两小时后','status':'planned','relation_to_now':'future','may_ask_progress':True}], 'reply_constraints':'尚未见面'}),{}))
        result,_=temporal.analyze(model,[{'role':'user','content':'再过两个钟头见面'}],[])
        self.assertEqual(model.call_count,1)
        self.assertFalse(result['events'][0]['may_ask_progress'])

    def test_api_constraint_list_is_normalized(self):
        model=Mock(return_value=(json.dumps({'events': [], 'reply_constraints': ['尚未发生','不要预设进展']}), {}))
        result,_=temporal.analyze(model,[],[])
        self.assertTrue(result['analysis_available'])
        self.assertIn('尚未发生',result['reply_constraints'])

    def test_assistant_guess_is_not_evidence(self):
        event={'event':'吃饭','evidence':'你应该吃过了','event_time':'现在','status':'completed','relation_to_now':'past','may_ask_progress':True}
        model=Mock(return_value=(json.dumps({'events':[event],'reply_constraints':''}),{}))
        result,_=temporal.analyze(model,[{'role':'assistant','content':'你应该吃过了'},{'role':'user','content':'你好'}],[])
        self.assertEqual(result['events'],[])

    def test_failure_keeps_uncertainty(self):
        result,_=temporal.analyze(Mock(side_effect=TimeoutError),[],[])
        self.assertFalse(result['analysis_available'])
        self.assertEqual(result['events'],[])

    def test_future_progress_greeting_rejected(self):
        import server
        with patch('server.model', return_value=('{"valid":false}',{})):
            reply,stats=server.review_greeting_timing([{'role':'user','content':'14:00；13:55用户说晚上去吃饭'}],'吃上了吗？')
        self.assertFalse(stats['valid'])
        self.assertNotIn('吃上',reply)

    def test_malformed_review_is_conservative(self):
        import server
        with patch('server.model', return_value=('not json',{})):
            reply,stats=server.review_greeting_timing([{'role':'user','content':'时间未知'}],'已经结束了吧？')
        self.assertFalse(stats['valid'])
        self.assertNotIn('结束',reply)
