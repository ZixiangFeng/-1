import io, json, unittest, urllib.error
from cloud_api import generate, validate


class FakeResponse(io.BytesIO):
    pass


class Opener:
    def __init__(self, result):
        self.result = result
        self.request = None

    def open(self, req, timeout):
        self.request = req
        if isinstance(self.result, Exception):
            raise self.result
        return FakeResponse(json.dumps(self.result).encode())


class CloudAPI(unittest.TestCase):
    def setUp(self):
        self.config = {
            "api_key": "TEST-SECRET",
            "model": "TEST-MODEL",
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        }

    def test_private_header_and_json_format(self):
        op = Opener(
            {
                "choices": [
                    {
                        "message": {"content": '{"reply":"你好"}'},
                        "finish_reason": "stop",
                    }
                ],
                "model": "TEST-MODEL",
            }
        )
        result, stats = generate(
            self.config,
            [{"role": "user", "content": "你好"}],
            schema={"type": "object"},
            opener=op,
        )
        body = json.loads(op.request.data)
        self.assertNotIn("TEST-SECRET", op.request.data.decode())
        self.assertEqual(op.request.get_header("Authorization"), "Bearer TEST-SECRET")
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(stats["provider"], "volcengine_ark")
        self.assertNotIn("TEST-SECRET", json.dumps(stats))

    def test_errors_do_not_echo_upstream_secret(self):
        error = urllib.error.HTTPError(
            "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
            401,
            "error",
            {},
            FakeResponse(b"TEST-SECRET"),
        )
        with self.assertRaises(ValueError) as ctx:
            generate(self.config, [], opener=Opener(error))
        self.assertNotIn("TEST-SECRET", str(ctx.exception))
        self.assertIn("密钥", str(ctx.exception))

    def test_missing_model_and_wrong_host_blocked(self):
        for config in [
            dict(self.config, model=""),
            dict(self.config, base_url="https://example.org/api/v3"),
        ]:
            with self.assertRaises(ValueError):
                validate(config)

    def test_overdue_is_not_reported_as_model_permission_error(self):
        error = urllib.error.HTTPError('https://ark.cn-beijing.volces.com/api/v3/responses', 403, 'error', {},
            FakeResponse(json.dumps({'error': {'code': 'AccountOverdueError', 'message': 'TEST-SECRET'}}).encode()))
        with self.assertRaisesRegex(ValueError, '账户欠费') as ctx:
            generate(self.config, [], opener=Opener(error))
        self.assertNotIn('TEST-SECRET', str(ctx.exception))

    def test_empty_or_truncated_reply_rejected(self):
        for text, reason in [("", "stop"), ("half", "length")]:
            op = Opener(
                {"choices": [{"message": {"content": text}, "finish_reason": reason}]}
            )
            with self.assertRaises(ValueError):
                generate(self.config, [], opener=op)

    def test_responses_image_json_and_output(self):
        op=Opener({'status':'completed','model':'resolved-model','output':[
            {'type':'reasoning','summary':[]},
            {'type':'message','role':'assistant','content':[{'type':'output_text','text':'{"ok":true}'}]}
        ]})
        content,stats=generate(dict(self.config,api_type='responses',vision_model='vision'),
            [{'role':'user','content':'图片','images':['YWJj']}],vision=True,
            schema={'type':'object'},opener=op)
        body=json.loads(op.request.data)
        self.assertTrue(op.request.full_url.endswith('/responses'))
        self.assertFalse(body['store'])
        self.assertEqual(body['input'][-1]['content'][1],{'type':'input_image','image_url':'data:image/jpeg;base64,YWJj'})
        self.assertEqual(body['text']['format']['type'],'json_object')
        self.assertEqual(json.loads(content),{'ok':True})
        self.assertEqual(stats['model'],'resolved-model')

    def test_responses_incomplete_and_empty_rejected(self):
        for result in [{'status':'incomplete','output':[]},{'status':'completed','output':[]}]:
            with self.assertRaises(ValueError):
                generate(dict(self.config,api_type='responses'),[],opener=Opener(result))
