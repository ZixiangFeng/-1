import base64, io, json, threading, unittest, urllib.request, urllib.error
from unittest.mock import patch
from PIL import Image
import os, tempfile

os.environ.setdefault("DEMO_DATA", tempfile.mkdtemp())
import server


class API(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # API contracts use a stub generator and retrieval. Loading the real
        # ONNX model here makes an offline API suite depend on model hydration;
        # vector persistence/ranking are exercised separately in test_rag.py.
        cls.store_patch = patch('server.get_store', return_value=None)
        cls.store_patch.start()
        cls.time_analysis_patch = patch('server.temporal.analyze', return_value=({'events': [], 'analysis_available': True}, {}))
        cls.time_analysis_patch.start()
        cls.time_review_patch = patch('server.review_greeting_timing', side_effect=lambda prompt, greeting: (greeting, {'valid': True}))
        cls.time_review_patch.start()
        cls.real_review = staticmethod(server.review_reply)
        cls.review_patch = patch(
            "server.review_reply",
            side_effect=lambda prompt, raw: (
                server.clean_reply(raw),
                {"stats": {"model": "TEST-REVIEW"}},
            ),
        )
        cls.review_patch.start()
        cls.rag_patch = patch(
            "server.rag_context",
            side_effect=lambda c, u, a, t, im, k: {
                "memories": server.retrieve(c, u, a, k),
                "policies": [],
            },
        )
        cls.rag_patch.start()
        cls.plan_patch = patch(
            "server.companion_plan",
            return_value=(
                {
                    "emotion": "不确定",
                    "need": "倾听",
                    "strategy": "倾听",
                    "avoid": "臆测",
                    "expression_meaning": "无奈",
                    "expression_context": "论文",
                },
                {},
            ),
        )
        cls.plan_patch.start()
        server.init()
        cls.http = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.url = "http://127.0.0.1:" + str(cls.http.server_port)
        threading.Thread(target=cls.http.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.store_patch.stop()
        cls.time_analysis_patch.stop()
        cls.time_review_patch.stop()
        cls.review_patch.stop()
        cls.rag_patch.stop()
        cls.plan_patch.stop()
        cls.http.shutdown()
        cls.http.server_close()

    def call(self, path, b=None, token=None, admin=False):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = "Bearer " + token
        if admin:
            headers["X-Admin-Key"] = server.ADMIN
        req = urllib.request.Request(
            self.url + path, json.dumps(b).encode() if b is not None else None, headers
        )
        try:
            with urllib.request.urlopen(req) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise

    def test_style_lab_requires_admin_and_keeps_participant_data_private(self):
        user = self.call('/api/admin/enroll', {'label':'QA-STYLE-AUTH','arm':'M'}, admin=True)
        for path, body in [('/api/admin/style-lab/catalog', None),
                           ('/api/admin/style-lab/status', None),
                           ('/api/admin/style-lab/training-material', None),
                           ('/api/admin/style-lab/image?key=public:test', None),
                           ('/api/admin/style-lab/build', {}),
                           ('/api/admin/style-lab/search', {'query':'public:test'})]:
            with self.assertRaises(urllib.error.HTTPError) as error:
                self.call(path, body, token=user['token'])
            self.assertEqual(error.exception.code, 403)
        with patch('style_lab.inventory', return_value={'items': []}) as inventory:
            self.assertEqual(self.call('/api/admin/style-lab/catalog', admin=True)['items'], [])
            inventory.assert_called_once()

    def test_image_reuse_admin_and_participant_isolation(self):
        a = self.call('/api/admin/enroll', {'label':'QA-REUSE-A','arm':'M'}, admin=True)
        b = self.call('/api/admin/enroll', {'label':'QA-REUSE-B','arm':'M'}, admin=True)
        with server.connect() as c:
            uid = c.execute('SELECT id FROM users WHERE token=?',(a['token'],)).fetchone()[0]
            c.execute("UPDATE users SET consent=? WHERE id=?",(server.now(),uid))
            c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?)',
                      ('reuse-bot',uid,'session','assistant','回复','reuse-img','[]','',server.now()))
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.call('/api/admin/expression-reuse?user='+uid, token=a['token'])
        self.assertEqual(error.exception.code,403)
        self.call('/api/telemetry/image-seen',{'page':'reuse-p','message':'reuse-bot','asset':'reuse-img'},token=a['token'])
        with self.assertRaises(urllib.error.HTTPError):
            self.call('/api/telemetry/image-seen',{'page':'reuse-p','message':'reuse-bot','asset':'reuse-img'},token=b['token'])
        d = self.call('/api/admin/expression-reuse?user='+uid,admin=True)
        self.assertEqual(d['participant']['id'],uid)
        self.assertEqual(d['version'],'expression-reuse-28d-v2')
        self.assertEqual(d['period']['scope'],'whole_study')
        export = self.call('/api/admin/export',admin=True)
        self.assertIn('image_views',export['tables'])

    def test_researcher_screenshot_intake_is_private_and_not_a_chat_send(self):
        from tests.test_screenshot_library import screenshot
        a=self.call('/api/admin/enroll',{'label':'QA-INTAKE-A','arm':'D'},admin=True)
        b=self.call('/api/admin/enroll',{'label':'QA-INTAKE-B','arm':'M'},admin=True)
        with self.assertRaises(urllib.error.HTTPError) as denied:
            self.call('/api/admin/personal-library/preview',{'user':a['id'],'image':screenshot()},token=a['token'])
        self.assertEqual(denied.exception.code,403)
        d=self.call('/api/admin/personal-library/preview',{'user':a['id'],'image':screenshot()},admin=True)
        with self.assertRaises(urllib.error.HTTPError):
            self.call('/api/admin/personal-library/commit',{'user':b['id'],'draft':d['draft'],'tiles':[{'index':0}]},admin=True)
        result=self.call('/api/admin/personal-library/commit',{'user':a['id'],'draft':d['draft'],'tiles':[{'index':0}]},admin=True)
        self.assertEqual(result['imported'],1)
        self.call('/api/consent',{},token=a['token'])
        library=self.call('/api/library',token=a['token'])['items']
        self.assertEqual(library[0]['id'],result['asset_ids'][0]);self.assertFalse(library[0]['used'])
        with patch('server.model',side_effect=AssertionError('import must not call AI')):
            disabled=self.call('/api/admin/personal-library/analyze',{'user':a['id']},admin=True)
            self.assertEqual(disabled['policy'],'storage-only')
        own=self.call('/api/admin/personal-library?user='+a['id'],admin=True)
        self.assertIsNone(own['profile']);self.assertEqual(own['items'][0]['usages'],[])
        other=self.call('/api/admin/personal-library?user='+b['id'],admin=True)
        self.assertEqual(other['items'],[])
        with server.connect() as c:
            self.assertEqual(c.execute('SELECT count(*) FROM messages WHERE user=?',(a['id'],)).fetchone()[0],0)
            self.assertEqual(c.execute('SELECT count(*) FROM memories WHERE user=?',(a['id'],)).fetchone()[0],0)
        html=urllib.request.urlopen(self.url+'/research').read().decode()
        self.assertIn('截图导入与使用记录',html)
        self.assertNotIn('__SCREENSHOT_IMPORT__',html)

    def test_save_received_sticker_is_private_idempotent_and_not_learning(self):
        a=self.call('/api/admin/enroll',{'label':'QA-SAVE-A','arm':'G'},admin=True)
        b=self.call('/api/admin/enroll',{'label':'QA-SAVE-B','arm':'D'},admin=True)
        for person in (a,b):self.call('/api/consent',{},token=person['token'])
        buf=io.BytesIO();Image.new('RGB',(18,18),'purple').save(buf,format='JPEG')
        encoded=base64.b64encode(buf.getvalue()).decode()
        with server.connect() as c:
            c.execute('INSERT INTO generic_stickers VALUES(?,?,?,?,?)',('save-generic',encoded,'{}',1,server.now()))
            for person in (a,b):
                c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?)',
                    ('save-bot-'+person['id'],person['id'],'s','assistant','收到','save-generic','[]','',server.now()))
            c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?)',
                ('save-user-only',a['id'],'s','user','hi','save-generic','[]','',server.now()))
        with patch('server.model',side_effect=AssertionError('saving must not call the model')):
            with self.assertRaises(urllib.error.HTTPError):
                self.call('/api/library/save',{'message':'save-bot-'+a['id']},token=b['token'])
            for payload in ({'asset':'save-generic'},{'message':'save-user-only'},{'message':'missing'}):
                with self.assertRaises(urllib.error.HTTPError):self.call('/api/library/save',payload,token=a['token'])
            saved=self.call('/api/library/save',{'message':'save-bot-'+a['id']},token=a['token'])
            again=self.call('/api/library/save',{'message':'save-bot-'+a['id']},token=a['token'])
            self.assertFalse(saved['duplicate']);self.assertTrue(again['duplicate']);self.assertEqual(saved['id'],again['id'])
            self.assertNotEqual(saved['id'],'save-generic')
            self.assertEqual(self.call('/api/library',token=b['token'])['items'],[])
            other=self.call('/api/library/save',{'message':'save-bot-'+b['id']},token=b['token'])
            self.assertNotEqual(saved['id'],other['id'])
        library=self.call('/api/library',token=a['token'])['items']
        self.assertEqual(len(library),1);self.assertFalse(library[0]['used']);self.assertEqual(library[0]['description'],'')
        self.assertEqual(self.call('/api/asset/'+saved['id'],token=a['token'])['image'],encoded)
        with self.assertRaises(urllib.error.HTTPError):self.call('/api/asset/'+saved['id'],token=b['token'])
        with server.connect() as c:
            self.assertEqual(c.execute('SELECT count(*) FROM messages WHERE user=?',(a['id'],)).fetchone()[0],2)
            for table in ('memories','learning_evidence','message_sends'):
                self.assertEqual(c.execute('SELECT count(*) FROM '+table+' WHERE user=?',(a['id'],)).fetchone()[0],0)
            self.assertEqual(c.execute('SELECT count(*) FROM sticker_saves WHERE user=?',(a['id'],)).fetchone()[0],1)
            provenance=dict(c.execute('SELECT * FROM sticker_saves WHERE user=?',(a['id'],)).fetchone())
            self.assertEqual(provenance['source_asset'],'save-generic');self.assertEqual(provenance['asset'],saved['id'])
            c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?)',('save-own-bot',a['id'],'s','assistant','hi',saved['id'],'[]','',server.now()))
        own=self.call('/api/library/save',{'message':'save-own-bot'},token=a['token'])
        self.assertEqual(own,saved|{'duplicate':True})
        self.assertIn('sticker_saves',self.call('/api/admin/export',admin=True)['tables'])

    def test_public_demo_is_opt_in_and_isolated(self):
        with patch.dict(os.environ, {"PUBLIC_DEMO": "0"}):
            with self.assertRaises(urllib.error.HTTPError) as error:
                self.call("/api/demo/enroll", {})
            self.assertEqual(error.exception.code, 403)
        with patch.dict(os.environ, {"PUBLIC_DEMO": "1", "PUBLIC_DEMO_LIMIT": "200"}):
            first = self.call("/api/demo/enroll", {})
            second = self.call("/api/demo/enroll", {})
            self.assertNotEqual(first["token"], second["token"])
            with server.connect() as c:
                rows = c.execute("SELECT is_test,consent FROM users WHERE token IN (?,?)", (first["token"], second["token"])).fetchall()
                self.assertTrue(all(row["is_test"] == 1 and not row["consent"] for row in rows))
        with patch.dict(os.environ, {"PUBLIC_DEMO": "1", "PUBLIC_DEMO_LIMIT": "0"}):
            with self.assertRaises(urllib.error.HTTPError):
                self.call("/api/demo/enroll", {})

    def test_account_login_recovers_history_and_logout_revokes_session(self):
        a = self.call('/api/admin/enroll', {'label': 'QA-ACCOUNT', 'arm': 'memory'}, admin=True)
        b = self.call('/api/admin/enroll', {'label': 'QA-OTHER', 'arm': 'memory'}, admin=True)
        login = self.call('/api/login', {'username': a['username'], 'password': '66666666'})
        again = self.call('/api/login', {'username': a['username'].lower(), 'password': '66666666'})
        self.assertNotEqual(login['token'], again['token'])
        self.call('/api/consent', {}, login['token'])
        session = self.call('/api/session', {}, login['token'])
        with server.connect() as c:
            c.execute("INSERT INTO messages(id,user,session,role,text,asset,expressions,interpretation,created) VALUES(?,?,?,?,?,?,?,?,?)",
                      ('account-history', a['id'], session['id'], 'user', '跨设备恢复测试', None, '[]', '', server.now()))
        state = self.call('/api/state', token=again['token'])
        self.assertEqual(state['messages'][0]['text'], '跨设备恢复测试')
        other = self.call('/api/login', {'username': b['username'], 'password': '66666666'})
        self.assertEqual(self.call('/api/state', token=other['token'])['messages'], [])
        with self.assertRaises(urllib.error.HTTPError):
            self.call('/api/login', {'username': a['username'], 'password': 'wrong'})
        self.call('/api/logout', {}, login['token'])
        with self.assertRaises(urllib.error.HTTPError):
            self.call('/api/state', token=login['token'])
        self.assertEqual(self.call('/api/state', token=again['token'])['id'], a['id'])

    def test_admin_login_is_scoped_and_revocable(self):
        with server.connect() as c:
            before = c.execute('SELECT count(*) FROM users').fetchone()[0]
        login = self.call('/api/login', {'username': 'admin', 'password': server.ADMIN})
        self.assertEqual(login['role'], 'admin')
        token = login['token']
        self.assertEqual(self.call('/api/admin/session', token=token)['role'], 'admin')
        self.assertIsInstance(self.call('/api/admin/users', token=token), list)
        with server.connect() as c:
            self.assertEqual(c.execute('SELECT count(*) FROM users').fetchone()[0], before)
        with self.assertRaises(urllib.error.HTTPError):
            self.call('/api/state', token=token)
        participant = self.call('/api/admin/enroll', {'label': 'QA-ADMIN-SCOPE', 'arm': 'T'}, token=token)
        user = self.call('/api/login', {'username': participant['username'], 'password': '66666666'})
        self.assertEqual(user['role'], 'participant')
        for endpoint in ('/api/admin/session', '/api/admin/users', '/api/admin/export', '/api/admin/generic-library'):
            with self.assertRaises(urllib.error.HTTPError) as error:
                self.call(endpoint, token=user['token'])
            self.assertEqual(error.exception.code, 403)
        with self.assertRaises(urllib.error.HTTPError):
            self.call('/api/admin/enroll', {'label': 'SHOULD-NOT-CREATE'}, token=user['token'])
        self.call('/api/logout', {}, token=token)
        with self.assertRaises(urllib.error.HTTPError):
            self.call('/api/admin/session', token=token)

    def test_admin_session_expiry_and_wrong_password(self):
        with self.assertRaises(urllib.error.HTTPError):
            self.call('/api/login', {'username': 'ADMIN', 'password': '66666666'})
        login = self.call('/api/login', {'username': 'ADMIN', 'password': server.ADMIN})
        with server.connect() as c:
            c.execute('UPDATE admin_sessions SET expires=0')
        with self.assertRaises(urllib.error.HTTPError):
            self.call('/api/admin/session', token=login['token'])

    def test_frequency_settings_admin_only_and_persistent(self):
        defaults = self.call('/api/admin/sticker-frequency', admin=True)
        self.assertTrue(defaults['enabled'])
        new = {**defaults, 'increment': 0.05}
        try:
            with self.assertRaises(urllib.error.HTTPError):
                self.call('/api/admin/sticker-frequency', new)
            self.call('/api/admin/sticker-frequency', new, admin=True)
            self.assertEqual(self.call('/api/admin/sticker-frequency', admin=True), new)
            with self.assertRaises(urllib.error.HTTPError):
                self.call('/api/admin/sticker-frequency', {**new, 'cap': 1}, admin=True)
            self.assertEqual(self.call('/api/admin/sticker-frequency', admin=True), new)
        finally:
            self.call('/api/admin/sticker-frequency', defaults, admin=True)

    def test_behavior_access_telemetry_and_survey_schedule(self):
        p = self.call('/api/admin/enroll', {'label':'QA-BEHAVIOR','arm':'M'}, admin=True)
        self.call('/api/consent', {}, p['token'])
        self.call('/api/telemetry/activity', {'page':'qa-page','seq':1,'visible_seconds':15,'active_seconds':10}, p['token'])
        with self.assertRaises(urllib.error.HTTPError):
            self.call('/api/admin/behavior', token=p['token'])
        with self.assertRaises(urllib.error.HTTPError):
            self.call('/api/admin/transcript?user='+p['id'], token=p['token'])
        with self.assertRaises(urllib.error.HTTPError):
            self.call('/api/survey', {'wave':'day7','answers':[4,4,4]}, p['token'])
        with server.connect() as c:
            old = (server.datetime.now(server.timezone.utc)-server.timedelta(days=27)).isoformat()
            c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?)', ('behavior-user',p['id'],'s','user','第一天',None,'[]','',old))
        state=self.call('/api/state', token=p['token'])
        self.assertEqual(state['study']['current_day'],28)
        self.assertEqual([w['day'] for w in state['study']['waves']],[7,14,28])
        self.call('/api/survey', {'wave':'day28','answers':[4,4,4]}, p['token'])
        d=self.call('/api/admin/behavior?include_test=1&user='+p['id'],admin=True)
        self.assertEqual(d['summary']['user_sends'],1)
        self.assertEqual(d['summary']['active_seconds'],10)
        chat=self.call('/api/admin/transcript?user='+p['id'],admin=True)
        self.assertEqual(chat['total'],1)

    def create(self, arm):
        p = self.call("/api/admin/enroll", {"arm": arm, "label": "TEST"}, admin=True)
        self.call("/api/consent", {}, p["token"])
        s = self.call("/api/session", {}, p["token"])
        return p, s

    @patch(
        "server.model",
        return_value=(
            "听起来有点无奈。",
            {"model": "TEST-STUB"},
        ),
    )
    def test_full_memory_loop_and_control(self, mock):
        for arm in ("memory", "control"):
            p, s = self.create(arm)
            t = p["token"]
            m = self.call("/api/chat", {"session": s["id"], "text": "改论文🙂"}, t)
            self.call(
                "/api/feedback",
                {
                    "message": m["message_id"],
                    "expression": "emoji:🙂",
                    "meaning": "实际是开心",
                    "context": "论文",
                },
                t,
            )
            s2 = self.call("/api/session", {}, t)
            self.call("/api/chat", {"session": s2["id"], "text": "改论文🙂"}, t)
            with server.connect() as c:
                logs = c.execute(
                    'SELECT payload FROM events WHERE user=? AND kind="generation" ORDER BY id DESC',
                    (p["id"],),
                ).fetchall()
                last = json.loads(logs[0][0])
                self.assertEqual(bool(last["memory_ids"]), arm == "memory")
            state = self.call("/api/state", token=t)
            self.assertNotIn("arm", state)
            self.call("/api/survey", {"wave": "baseline", "answers": [4, 5, 6]}, t)
            with self.assertRaises(urllib.error.HTTPError):
                self.call("/api/survey", {"wave": "baseline", "answers": [4, 5, 6]}, t)

    @patch("server.model", return_value=("一张猫猫图片", {"model": "TEST-STUB"}))
    def test_upload_and_ownership(self, mock):
        p, s = self.create("memory")
        q, _ = self.create("control")
        image = Image.new("RGB", (32, 32), "green")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        a = self.call(
            "/api/upload",
            {"image": base64.b64encode(buf.getvalue()).decode()},
            p["token"],
        )
        self.assertTrue(self.call("/api/asset/" + a["id"], token=p["token"])["image"])
        with self.assertRaises(urllib.error.HTTPError):
            self.call("/api/asset/" + a["id"], token=q["token"])

    @patch("server.model", return_value=("一张绿色表情图片", {"model": "TEST-STUB"}))
    def test_library_import_does_not_learn_until_sent(self, mock):
        p, s = self.create("memory")
        buf = io.BytesIO()
        Image.new("RGB", (32, 32), "green").save(buf, format="PNG")
        a = self.call(
            "/api/library/import",
            {"image": base64.b64encode(buf.getvalue()).decode()},
            p["token"],
        )
        mock.assert_not_called()
        library = self.call("/api/library", token=p["token"])
        item = next(x for x in library["items"] if x["id"] == a["id"])
        self.assertEqual(item["used"], 0)
        with server.connect() as c:
            row = c.execute(
                "SELECT description FROM assets WHERE id=?", (a["id"],)
            ).fetchone()
            self.assertEqual(row[0], "")
            self.assertEqual(
                c.execute(
                    "SELECT count(*) FROM memories WHERE user=? AND expression=?",
                    (p["id"], "image:" + a["id"]),
                ).fetchone()[0],
                0,
            )
        self.call(
            "/api/chat",
            {"session": s["id"], "text": "", "asset": a["id"]},
            p["token"],
        )
        self.assertTrue(mock.called)
        with server.connect() as c:
            self.assertTrue(
                c.execute(
                    "SELECT description FROM assets WHERE id=?", (a["id"],)
                ).fetchone()[0]
            )

    def test_summary_context_reaches_four_weeks(self):
        from datetime import datetime, timezone, timedelta
        participant, _ = self.create("M")
        with server.connect() as c:
            for days in (1, 20, 27, 31):
                sid='long-summary-'+str(days)
                created=(datetime.now(timezone.utc)-timedelta(days=days)).isoformat()
                c.execute('INSERT INTO sessions VALUES(?,?,?)',(sid,participant['id'],created))
                c.execute('INSERT INTO conversation_summaries VALUES(?,?,?,?,?,?)',
                          (sid,participant['id'],sid,'第'+str(days)+'天前的事情',created,created))
            rows=server.conversation_summary_context(c,participant['id'],'current')
            self.assertEqual(len(rows),3)
            self.assertTrue(any('27' in r['summary'] for r in rows))
            self.assertFalse(any('31' in r['summary'] for r in rows))

    @patch("server.model", return_value=("用户正在修改论文，担心返工，明天准备继续处理。", {"model": "TEST-SUMMARY"}))
    def test_cross_day_summary_is_user_isolated(self, mock):
        p, first = self.create("memory")
        q, other = self.create("memory")
        with server.connect() as c:
            c.execute(
                "INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?)",
                ("summary-source-p", p["id"], first["id"], "user", "我的论文又要返工，明天继续改", None, "[]", "", server.now()),
            )
            c.execute(
                "INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?)",
                ("summary-source-q", q["id"], other["id"], "user", "我准备去旅行", None, "[]", "", server.now()),
            )
            c.commit()
        current = self.call("/api/session", {}, p["token"])
        with server.connect() as c:
            server.ensure_prior_conversation_summaries(c, p["id"], current["id"])
            c.commit()
            own = server.conversation_summary_context(c, p["id"], current["id"])
            foreign = server.conversation_summary_context(c, q["id"], other["id"])
            self.assertEqual(len(own), 1)
            self.assertIn("论文", own[0]["summary"])
            self.assertEqual(foreign, [])
            self.assertEqual(
                c.execute(
                    "SELECT count(*) FROM conversation_summaries WHERE user=?",
                    (q["id"],),
                ).fetchone()[0],
                0,
            )

    @patch("server.model", return_value=("", {"model": "TEST-STUB"}))
    def test_empty_vision_not_saved(self, mock):
        p, _ = self.create("memory")
        buf = io.BytesIO()
        Image.new("RGB", (32, 32), "blue").save(buf, format="PNG")
        with self.assertRaises(urllib.error.HTTPError):
            self.call(
                "/api/upload",
                {"image": base64.b64encode(buf.getvalue()).decode()},
                p["token"],
            )
        with server.connect() as c:
            self.assertEqual(
                c.execute(
                    "SELECT count(*) FROM assets WHERE user=?", (p["id"],)
                ).fetchone()[0],
                0,
            )

    @patch(
        "server.model",
        side_effect=[
            ("（轻轻点头）", {"model": "TEST-STUB"}),
            ('{"reply":"这轮又要重来，确实挺让人泄气的。"}', {"model": "TEST-REPAIR"}),
        ],
    )
    def test_action_only_draft_gets_real_second_generation(self, mock):
        p, s = self.create("memory")
        with patch("server.review_reply", self.real_review):
            result = self.call(
                "/api/chat",
                {"session": s["id"], "text": "稿子又被退回来了"},
                p["token"],
            )
        self.assertEqual(result["reply"], "这轮又要重来，确实挺让人泄气的。")
        self.assertEqual(mock.call_count, 2)
        with server.connect() as c:
            payload = json.loads(
                c.execute(
                    "SELECT payload FROM events WHERE user=? AND kind='generation'",
                    (p["id"],),
                ).fetchone()[0]
            )
            self.assertEqual(payload["repair"]["stats"]["model"], "TEST-REPAIR")

    def test_admin_protected(self):
        with self.assertRaises(urllib.error.HTTPError):
            self.call("/api/admin/export")

    def test_first_opening_is_immediate_once_and_shared_across_arms(self):
        from greeting_delivery import FIRST_OPENING
        with patch('server.model',side_effect=AssertionError('first opening does not need model calls')):
            for arm in ('T','G','M','D'):
                p=self.call('/api/admin/enroll',{'label':'QA-FIRST-'+arm,'arm':arm},admin=True)
                with self.assertRaises(urllib.error.HTTPError):
                    self.call('/api/opening',{},p['token'])
                self.call('/api/consent',{},p['token'])
                first=self.call('/api/opening',{},p['token'])
                self.assertTrue(first['created'])
                self.assertFalse(self.call('/api/opening',{},p['token'])['created'])
                state=self.call('/api/state',token=p['token'])
                self.assertEqual(len(state['messages']),1)
                self.assertEqual(state['messages'][0]['role'],'assistant')
                self.assertEqual(state['messages'][0]['text'],FIRST_OPENING)
                self.assertIsNone(state['messages'][0]['asset'])
                self.assertEqual(state['study']['current_day'],None)
                with server.connect() as c:
                    audit=json.loads(c.execute("SELECT payload FROM events WHERE user=? AND kind='proactive_greeting'",(p['id'],)).fetchone()[0])
                    self.assertEqual(audit['kind'],'first_opening')
                    self.assertFalse(audit['stats']['model_used'])

    def test_first_opening_does_not_backfill_an_existing_user_conversation(self):
        p,s=self.create('D')
        with server.connect() as c:
            c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?)',('before-opening',p['id'],s['id'],'user','你好',None,'[]','',server.now()))
        self.assertFalse(self.call('/api/opening',{},p['token'])['created'])
        state=self.call('/api/state',token=p['token'])
        self.assertEqual([m['role'] for m in state['messages']],['user'])

    def test_first_opening_multiple_devices_are_idempotent(self):
        p,s=self.create('D');results=[]
        threads=[threading.Thread(target=lambda:results.append(self.call('/api/opening',{},p['token']))) for _ in range(2)]
        for t in threads:t.start()
        for t in threads:t.join(5)
        self.assertEqual(sum(x['created'] for x in results),1)
        self.assertEqual(len(self.call('/api/state',token=p['token'])['messages']),1)

    @patch(
        "server.model",
        return_value=("你回来啦～\n论文那边想接着说，还是先放空一会儿？", {"model": "TEST-GREETING"}),
    )
    def test_proactive_greeting_once_per_visit_and_user_isolated(self, mock):
        p, s = self.create("memory")
        first = self.call(
            "/api/greeting",
            {"session": s["id"], "visit": "visit-one", "delivery": "visible-first-v1", "local_hour": 20},
            p["token"],
        )
        self.assertTrue(first["prepared"])
        self.assertFalse(self.call("/api/state", token=p["token"])["messages"])
        saved = self.call("/api/greeting/commit", {"draft": first["draft"]}, p["token"])
        self.assertTrue(saved["created"])
        self.assertTrue(self.call("/api/greeting/commit", {"draft": first["draft"]}, p["token"])["created"])
        again = self.call(
            "/api/greeting",
            {"session": s["id"], "visit": "visit-one", "delivery": "visible-first-v1", "local_hour": 20},
            p["token"],
        )
        self.assertFalse(first["created"])
        self.assertFalse(again["created"])
        self.assertEqual(first["reply"], again["reply"])
        state = self.call("/api/state", token=p["token"])
        self.assertEqual(len([m for m in state["messages"] if m["role"] == "assistant"]), 1)
        self.assertEqual(state["visit_times"], [greetings[0]["created"]] if (greetings := [m for m in state["messages"] if m["role"] == "assistant"]) else [])
        self.assertEqual(mock.call_count, 1)
        q, _ = self.create("control")
        with self.assertRaises(urllib.error.HTTPError):
            self.call(
                "/api/greeting",
                {"session": s["id"], "visit": "foreign", "delivery": "visible-first-v1", "local_hour": 20},
                q["token"],
            )
        with server.connect() as c:
            self.assertEqual(
                c.execute("SELECT count(*) FROM visits WHERE user=?", (q["id"],)).fetchone()[0],
                0,
            )

    @patch(
        "server.model",
        side_effect=[
            ("昨晚睡得还好吗？", {"model": "TEST-WRONG-TIME"}),
            ("你回来啦～刚才想聊的还可以接着说。", {"model": "TEST-CORRECTED"}),
        ],
    )
    def test_greeting_corrects_false_yesterday_reference(self, mock):
        p, s = self.create("memory")
        with server.connect() as c:
            c.execute(
                "INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?)",
                ("same-day-user-message", p["id"], s["id"], "user", "刚刚看了会儿电视，现在还没准备睡。", None, "[]", "", (server.datetime.now(server.timezone.utc)-server.timedelta(minutes=12)).isoformat()),
            )
            c.commit()
        result = self.call(
            "/api/greeting",
            {
                "session": s["id"],
                "visit": "same-day-visit",
                "delivery": "visible-first-v1",
                "local_hour": 21,
                "timezone_offset": -480,
            },
            p["token"],
        )
        self.assertNotIn("昨晚", result["reply"])
        self.assertNotIn("昨天", result["reply"])
        self.assertEqual(mock.call_count, 2)
        supplied = json.loads(mock.call_args_list[0].args[0][-1]["content"])
        self.assertIn("短暂离开后回来", supplied["本次来访模式"])
        self.assertEqual(supplied["最近对话"][-1]["local_day"], "今天")

    @patch("server.model", return_value=("回来啦～", {}))
    def test_unseen_greeting_draft_can_be_cancelled_without_history(self, model):
        p, s = self.create("M")
        draft = self.call("/api/greeting", {"session":s["id"],"visit":"cancel-draft","delivery":"visible-first-v1"}, p["token"])
        q, _ = self.create("M")
        self.assertFalse(self.call("/api/greeting/commit", {"draft":draft["draft"]}, q["token"])["created"])
        self.call("/api/greeting/cancel", {"visit":"cancel-draft"}, p["token"])
        self.assertFalse(self.call("/api/greeting/commit", {"draft":draft["draft"]}, p["token"])["created"])
        self.assertEqual(self.call("/api/state", token=p["token"])["messages"], [])

    def test_typing_cancels_greeting_while_model_is_still_running(self):
        p, s = self.create("M")
        started, finish = threading.Event(), threading.Event()
        result = []
        def slow_model(*args, **kwargs):
            started.set()
            self.assertTrue(finish.wait(5))
            return "回来啦～", {}
        with patch("server.model", side_effect=slow_model):
            thread = threading.Thread(target=lambda: result.append(self.call("/api/greeting", {"session":s["id"],"visit":"slow-visit","delivery":"visible-first-v1"}, p["token"])))
            thread.start()
            try:
                self.assertTrue(started.wait(5))
                self.call("/api/greeting/cancel", {"visit":"slow-visit"}, p["token"])
            finally:
                finish.set(); thread.join(5)
        self.assertFalse(result[0].get("prepared", False))
        self.assertEqual(self.call("/api/state", token=p["token"])["messages"], [])

    def test_fallback_replaces_slow_model_without_lock_or_late_duplicate(self):
        p, session = self.create("M")
        body={"session":session["id"],"visit":"deadline","delivery":"visible-first-v1","absence_seconds":600,"baseline_message_id":None,"timezone_offset":-480}
        started, finish = threading.Event(), threading.Event()
        result=[]
        def slow(*a,**kw):
            started.set(); self.assertTrue(finish.wait(6)); return "迟到的问候", {}
        with patch('server.model',side_effect=slow):
            thread=threading.Thread(target=lambda:result.append(self.call('/api/greeting',body,p['token'])))
            thread.start()
            try:
                self.assertTrue(started.wait(3))
                fast=self.call('/api/greeting/fallback',body,p['token'])
                self.assertTrue(fast['prepared'])
                again=self.call('/api/greeting/fallback',body,p['token'])
                self.assertEqual(again['draft'],fast['draft'])
                self.assertFalse(self.call('/api/state',token=p['token'])['messages'])
                self.assertTrue(self.call('/api/greeting/commit',{'draft':fast['draft']},p['token'])['created'])
            finally:
                finish.set(); thread.join(5)
        self.assertFalse(result[0].get('prepared'))
        messages=self.call('/api/state',token=p['token'])['messages']
        self.assertEqual(len(messages),1); self.assertNotIn('迟到',messages[0]['text'])

    @patch('server.model',side_effect=TimeoutError('model timed out'))
    def test_failed_model_fallback_and_cancel_before_begin(self,model):
        p,s=self.create('M')
        b={'session':s['id'],'visit':'failed-model','delivery':'visible-first-v1','absence_seconds':600,'baseline_message_id':None}
        with self.assertRaises(urllib.error.HTTPError):self.call('/api/greeting',b,p['token'])
        self.assertTrue(self.call('/api/greeting/fallback',b,p['token'])['prepared'])
        self.call('/api/greeting/cancel',{'visit':b['visit']},p['token'])
        self.assertFalse(self.call('/api/greeting/fallback',b,p['token']).get('prepared'))
        b['visit']='cancel-before-request'
        self.call('/api/greeting/cancel',{'visit':b['visit']},p['token'])
        self.assertEqual(self.call('/api/greeting',b,p['token'])['reason'],'cancelled')
        self.assertFalse(self.call('/api/greeting/fallback',b,p['token']).get('prepared'))

    def test_fallback_requires_same_history_absence_consent_and_owned_session(self):
        p,s=self.create('M');q,_=self.create('M')
        b={'session':s['id'],'visit':'fallback-check','delivery':'visible-first-v1','absence_seconds':599,'baseline_message_id':None}
        self.assertEqual(self.call('/api/greeting/fallback',b,p['token'])['reason'],'short_absence')
        b['absence_seconds']=600
        with self.assertRaises(urllib.error.HTTPError):self.call('/api/greeting/fallback',b,q['token'])
        with server.connect() as c:
            c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?)',('new-before-opening',p['id'],s['id'],'user','先发了',None,'[]','',server.now()));c.commit()
        self.assertEqual(self.call('/api/greeting/fallback',b,p['token'])['reason'],'superseded')
        self.assertEqual(len(self.call('/api/state',token=p['token'])['messages']),1)

    @patch("server.model", return_value=("回来啦～", {}))
    def test_new_user_message_invalidates_prepared_greeting(self, model):
        p, s = self.create("T")
        draft = self.call("/api/greeting", {"session":s["id"],"visit":"before-chat","delivery":"visible-first-v1"}, p["token"])
        self.call("/api/chat", {"session":s["id"],"text":"吃好饭了"}, p["token"])
        self.assertFalse(self.call("/api/greeting/commit", {"draft":draft["draft"]}, p["token"])["created"])
        state = self.call("/api/state", token=p["token"])
        self.assertEqual(len(state["messages"]), 2)
        self.assertEqual(state["messages"][0]["text"], "吃好饭了")
        recent = self.call("/api/greeting", {"session":s["id"],"visit":"quick-return","delivery":"visible-first-v1"}, p["token"])
        self.assertEqual(recent["reason"], "recent_conversation")

    @patch("server.model", return_value=("回来啦～", {}))
    def test_old_client_cannot_persist_a_hidden_greeting(self, model):
        p, s = self.create("M")
        result = self.call("/api/greeting", {"session":s["id"],"visit":"old-client"}, p["token"])
        self.assertFalse(result["created"])
        self.assertFalse(model.called)
        self.assertEqual(self.call("/api/state", token=p["token"])["messages"], [])

    @patch("server.model", return_value=("回来啦～之前说的旅行还在计划吗？", {}))
    def test_greeting_uses_absence_duration_and_supplies_objective_time(self, model):
        p, session = self.create("M")
        body = {"session": session["id"], "visit": "absence-check", "delivery": "visible-first-v1", "absence_seconds": 599, "timezone_offset": -480}
        result = self.call("/api/greeting", body, p["token"])
        self.assertEqual(result["reason"], "short_absence")
        self.assertFalse(model.called)
        body["absence_seconds"] = 600
        result = self.call("/api/greeting", body, p["token"])
        self.assertTrue(result["prepared"])
        context = json.loads(model.call_args.args[0][-1]["content"])
        self.assertEqual(context["本次离开秒数（不等于距上条消息）"], 600)
        self.assertIn("当前本地时间", context)
        self.assertIn("API事件状态分析", context)
        self.assertEqual(self.call("/api/state", token=p["token"])["messages"], [])
        self.call("/api/greeting/cancel", {"visit": body["visit"]}, p["token"])

    @patch("server.model", return_value=("好呀，我听你说。", {}))
    def test_completed_held_reply_can_be_withdrawn_before_next_message(self, model):
        p, session = self.create("T")
        self.call('/api/chat', {'session':session['id'],'text':'我今天有点累','turn':'held-turn'}, p['token'])
        q, _ = self.create("T")
        self.call('/api/cancel', {'turn':'held-turn','discard_reply':True}, q['token'])
        self.assertEqual(len(self.call('/api/state',token=p['token'])['messages']),2)
        self.call('/api/cancel', {'turn':'held-turn','discard_reply':True}, p['token'])
        messages=self.call('/api/state',token=p['token'])['messages']
        self.assertEqual([m['role'] for m in messages],['user'])
        self.call('/api/chat', {'session':session['id'],'text':'还有工作没做完','turn':'next-turn'}, p['token'])
        messages=self.call('/api/state',token=p['token'])['messages']
        self.assertEqual([m['role'] for m in messages],['user','user','assistant'])

    def test_user_can_interrupt_generation_and_continue_typing(self):
        p, s = self.create("memory")
        started = threading.Event()
        release = threading.Event()
        outcome = {}

        def slow_model(*args, **kwargs):
            started.set()
            release.wait(3)
            return "这条回复应该被停止", {"model": "TEST-SLOW"}

        def chat():
            try:
                self.call(
                    "/api/chat",
                    {"session": s["id"], "text": "我还没说完", "turn": "turn-1"},
                    p["token"],
                )
            except urllib.error.HTTPError as e:
                outcome["status"] = e.code
                outcome["body"] = json.load(e)

        with patch("server.model", side_effect=slow_model):
            thread = threading.Thread(target=chat)
            thread.start()
            self.assertTrue(started.wait(2))
            self.call("/api/cancel", {"turn": "turn-1"}, p["token"])
            release.set()
            thread.join(4)

        self.assertEqual(outcome["status"], 409)
        self.assertTrue(outcome["body"]["interrupted"])
        with server.connect() as c:
            roles = [
                r[0]
                for r in c.execute(
                    "SELECT role FROM messages WHERE user=? AND session=?",
                    (p["id"], s["id"]),
                )
            ]
            self.assertEqual(roles, ["user"])
            self.assertTrue(
                c.execute(
                    "SELECT 1 FROM events WHERE user=? AND kind='generation_interrupted'",
                    (p["id"],),
                ).fetchone()
            )

    @patch("server.model", side_effect=TimeoutError("timeout"))
    def test_failure_never_saved_as_reply(self, mock):
        p, s = self.create("memory")
        with self.assertRaises(urllib.error.HTTPError):
            self.call("/api/chat", {"session": s["id"], "text": "你好"}, p["token"])
        self.assertEqual(self.call("/api/state", token=p["token"])["messages"], [])

    def test_automatic_memory_loop_without_feedback(self):
        import learning
        def fake_model(messages, **kwargs):
            if kwargs.get("schema") == learning.SCHEMA:
                samples=json.loads(messages[-1]["content"])
                sample=next(x for x in reversed(samples) if x["expressions"])
                item=dict(source=sample["id"],expression=sample["expressions"][0],meaning="无奈",context="论文返工",evidence=sample["text"],evidence_message=sample["id"])
                return json.dumps({"observations":[item]}), {"model":"TEST-EXTRACTION"}
            return "反复修改确实让人累。", {"model":"TEST-CHAT"}
        with patch("server.model",side_effect=fake_model):
            for arm in ("memory","control"):
                p,s=self.create(arm)
                self.call("/api/chat",{"session":s["id"],"text":"又改论文了🙂"},p["token"])
                s2=self.call("/api/session",{},p["token"])
                self.call("/api/chat",{"session":s2["id"],"text":"又改论文了🙂"},p["token"])
                with server.connect() as c:
                    log=json.loads(c.execute("SELECT payload FROM events WHERE user=? AND kind='generation' ORDER BY id DESC",(p["id"],)).fetchone()[0])
                    self.assertEqual(bool(log["memory_ids"]),arm=="memory")
                    self.assertTrue(c.execute("SELECT 1 FROM memories WHERE user=? AND status='inferred'",(p["id"],)).fetchone())
                    self.assertFalse(c.execute("SELECT 1 FROM memories WHERE user=? AND status='confirmed'",(p["id"],)).fetchone())

    def test_three_arm_enrollment_and_balancing(self):
        for arm in ('T','G','M'):
            p=self.call('/api/admin/enroll',{'label':'QA-THREE','arm':arm},admin=True)
            self.assertEqual(p['arm'],arm)
            self.assertTrue(p['username'])
        for i in range(6):
            with server.connect() as c:
                counts={r[0]:r[1] for r in c.execute('SELECT arm,count(*) FROM users WHERE is_test=0 GROUP BY arm')}
            counts['M']=counts.get('M',0)+counts.get('memory',0)
            low=min(counts.get(x,0) for x in ('T','G','M'))
            p=self.call('/api/admin/enroll',{'label':'BALANCE','arm':'random'},admin=True)
            self.assertIn(p['arm'],('T','G','M'))
            self.assertEqual(counts.get(p['arm'],0),low)

    def test_binary_asset_endpoint_is_private_and_decodable(self):
        p, _ = self.create('M');q, _ = self.create('M')
        buf=io.BytesIO();Image.new('RGB',(24,24),'green').save(buf,format='PNG')
        asset=self.call('/api/library/import',{'image':base64.b64encode(buf.getvalue()).decode()},p['token'])
        path='/api/asset/'+asset['id']+'/content'
        req=urllib.request.Request(self.url+path,headers={'Authorization':'Bearer '+p['token']})
        with urllib.request.urlopen(req) as r:
            content=r.read();self.assertEqual(r.headers['Content-Type'],'image/jpeg')
            self.assertEqual(int(r.headers['Content-Length']),len(content))
            self.assertIn('private',r.headers['Cache-Control']);Image.open(io.BytesIO(content)).load()
        with self.assertRaises(urllib.error.HTTPError):self.call(path,token=q['token'])
        with self.assertRaises(urllib.error.HTTPError):self.call(path)

    def test_admin_material_crop_flow_auth_and_frozen_target(self):
        from test_screenshot_library import screenshot
        p=self.call('/api/admin/enroll',{'label':'QA-MATERIAL','arm':'D'},admin=True)
        for endpoint in ('preview','crop-import'):
            with self.assertRaises(urllib.error.HTTPError) as denied:
                self.call('/api/admin/generic-library/'+endpoint,{},token=p['token'])
            self.assertEqual(denied.exception.code,403)
        semantics=dict(description='测试图片',meaning='回应',situations='收到提醒',avoid='危机',style='二维卡通')
        with patch('server.model',return_value=(json.dumps(semantics),{})) as model:
            for pool in ('generic','divergent'):
                model.reset_mock()
                draft=self.call('/api/admin/generic-library/preview',{'image':screenshot(),'pool':pool},admin=True)
                model.assert_not_called()
                body={'draft':draft['draft'],'index':0,'pool':'invalid-not-used'}
                result=self.call('/api/admin/generic-library/crop-import',body,admin=True)
                self.assertEqual(result['pool'],pool);self.assertEqual(model.call_count,1)
                duplicate=self.call('/api/admin/generic-library/crop-import',body,admin=True)
                self.assertTrue(duplicate['duplicate']);self.assertEqual(model.call_count,1)
                self.call('/api/admin/generic-library/toggle',{'id':result['id'],'enabled':False},admin=True)
        with server.connect() as c:
            for table in ('assets','messages','memories'):
                self.assertEqual(c.execute('SELECT count(*) FROM '+table+' WHERE user=?',(p['id'],)).fetchone()[0],0)
        with urllib.request.urlopen(self.url+'/research') as r:html=r.read().decode()
        self.assertIn('genericCropPreview',html);self.assertIn('G 组公共表情库',html)
        self.assertIn('personalDLibrary',html);self.assertNotIn('id="libraryPool"',html)
        self.assertNotIn('__GENERIC_SCREENSHOT_IMPORT__',html)
        report=self.call('/api/admin/personal-library?user='+p['id'],admin=True)
        self.assertIsNotNone(report['divergence'])
        self.assertIsNone(report['divergence']['latest'])
        with self.assertRaises(urllib.error.HTTPError) as denied:
            self.call('/api/admin/personal-library?user='+p['id'],token=p['token'])
        self.assertEqual(denied.exception.code,403)

    def test_generic_library_admin_analysis_and_asset_isolation(self):
        import generic_stickers
        buf=io.BytesIO();Image.new('RGB',(12,12),'orange').save(buf,format='PNG')
        image=base64.b64encode(buf.getvalue()).decode()
        semantic=dict(description='测试',meaning='开心',situations='庆祝',avoid='悲伤',style='二维卡通、明亮配色')
        with self.assertRaises(urllib.error.HTTPError):
            self.call('/api/admin/generic-library/import',{'image':image})
        with patch('server.model',return_value=(json.dumps(semantic),{})) as model:
            result=self.call('/api/admin/generic-library/import',{'image':image},admin=True)
            self.assertTrue(model.call_args.kwargs['vision'])
        aid=result['id'];p,s=self.create('G');q,_=self.create('M')
        with server.connect() as c:
            self.assertEqual(c.execute('SELECT count(*) FROM memories WHERE user=?',(p['id'],)).fetchone()[0],0)
            c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?)',('generic-sent',p['id'],s['id'],'assistant','好耶',aid,'[]','',server.now()))
        self.assertTrue(self.call('/api/asset/'+aid,token=p['token'])['image'])
        with urllib.request.urlopen(urllib.request.Request(self.url+'/api/asset/'+aid+'/content',headers={'Authorization':'Bearer '+p['token']})) as r: Image.open(io.BytesIO(r.read())).load()
        with self.assertRaises(urllib.error.HTTPError):self.call('/api/asset/'+aid+'/content',token=q['token'])
        with self.assertRaises(urllib.error.HTTPError): self.call('/api/asset/'+aid,token=q['token'])
        self.call('/api/admin/generic-library/toggle',{'id':aid,'enabled':False},admin=True)
        self.assertTrue(self.call('/api/asset/'+aid,token=p['token'])['image'])
        with server.connect() as c: self.assertEqual(generic_stickers.candidates(c,set()),[])

    def test_t_g_chat_keeps_text_but_does_not_learn_personal_semantics(self):
        with patch('server.model',return_value=('今天终于忙完啦～',{})), patch('server.get_store',return_value=None), patch('server.learning.update') as learn:
            for arm in ('T','G'):
                p,s=self.create(arm)
                self.call('/api/chat',{'session':s['id'],'text':'今天忙完啦🙂'},p['token'])
                with server.connect() as c:
                    self.assertFalse(c.execute('SELECT 1 FROM memories WHERE user=?',(p['id'],)).fetchone())
                    msg=c.execute("SELECT text,asset FROM messages WHERE user=? AND role='assistant'",(p['id'],)).fetchone()
                    self.assertTrue(msg['text']);self.assertIsNone(msg['asset'])
            learn.assert_not_called()

    def test_g_d_final_send_blocks_personal_copies_even_if_selector_bypassed(self):
        from tests.test_sticker_identity import image
        encoded=image();copy=image(size=90,fmt='JPEG',quality=60)
        aid='g_send_guard'
        with server.connect() as c:
            c.execute('INSERT INTO generic_stickers VALUES(?,?,?,?,?)',
                (aid,copy,json.dumps(dict(description='卡通贴纸',meaning='收到',situations='回应提醒',avoid='危机')),1,server.now()))
        try:
            with patch('server.model',return_value=('收到啦，今天辛苦了～',{})), patch('server.learning.update'):
                for arm in ('G','D'):
                    p,s=self.create(arm)
                    owned=self.call('/api/library/import',{'image':encoded},p['token'])['id']
                    for bad in (aid,owned):
                        with patch('server.stickers.choose',return_value=(bad,{'asset_id':bad})):
                            reply=self.call('/api/chat',{'session':s['id'],'text':'终于忙完了'},p['token'])
                        self.assertIsNone(reply['asset']);self.assertTrue(reply['reply'])
                        with server.connect() as c:
                            row=c.execute('SELECT asset FROM messages WHERE id=?',(reply['message_id'],)).fetchone()
                            self.assertIsNone(row['asset'])
                            audit=json.loads(c.execute("SELECT payload FROM events WHERE user=? AND kind='sticker_selection' ORDER BY id DESC LIMIT 1",(p['id'],)).fetchone()[0])
                            self.assertEqual(audit['blocked_asset_id'],bad)
                            self.assertEqual(audit['personal_exclusion_policy'],'exclude-personal-images-v1')
                            self.assertIsNone(audit['asset_id'])
                            self.assertFalse(c.execute('SELECT 1 FROM memories WHERE user=?',(p['id'],)).fetchone())
        finally:
            with server.connect() as c:c.execute('DELETE FROM generic_stickers WHERE id=?',(aid,))
