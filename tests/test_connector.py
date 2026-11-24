import base64
import io
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit, parse_qs
from unittest.mock import patch
from PIL import Image

os.environ.setdefault('DEMO_DATA', tempfile.mkdtemp())
import server
import skill_connector
import stickers
import generic_stickers


class Connector(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        for mock in (patch.object(server,'DB',Path(self.tmp.name)/'study.sqlite3'),
                     patch.object(server,'DATA',Path(self.tmp.name)),
                     patch.object(server,'ADMIN','connector-test-admin'),
                     patch.dict(os.environ,{'XIAOYU_CONNECTOR_ENABLED':'1','XIAOYU_CONNECTOR_APP_ID':'test-app',
                                           'XIAOYU_CONNECTOR_REDIRECT_URIS':'["https://host.example/callback"]'})):
            mock.start(); self.addCleanup(mock.stop)
        server.init()
        self.http = server.ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
        threading.Thread(target=self.http.serve_forever,daemon=True).start()
        self.addCleanup(self.http.server_close)
        self.addCleanup(self.http.shutdown)
        self.url='http://127.0.0.1:'+str(self.http.server_port)
        self.model=patch('server.model',return_value=('今天辛苦啦～想聊就慢慢说。',{})).start()
        self.addCleanup(patch.stopall)
        patch('server.temporal.analyze',return_value=({'events':[]},{})).start()
        patch('server.rag_context',return_value={'memories':[],'policies':[]}).start()
        patch('server.companion_plan',return_value=({'expression_meaning':'','expression_context':''},{})).start()
        patch('server.review_reply',side_effect=lambda p,r:(r,{})).start()
        patch('server.get_store',return_value=None).start()
        self.learn=patch('server.learning.update').start()

    def request(self,path,body=None,headers=None):
        req=urllib.request.Request(self.url+path,data=json.dumps(body).encode() if body is not None else None,
            headers={'Content-Type':'application/json',**(headers or {})})
        with urllib.request.urlopen(req,timeout=10) as response:
            return json.load(response)

    def participant(self,arm='M'):
        return self.request('/api/admin/enroll',{'label':'QA-CONNECTOR','arm':arm},{'X-Admin-Key':server.ADMIN})

    def authorize(self,p,**changes):
        return self.request('/api/connector/authorize',{'app_id':'test-app','redirect_uri':'https://host.example/callback',
            'state':'test-state','consent':True,**changes},{'Authorization':'Bearer '+p['token']})

    def connect(self,p,openid='open-a'):
        code=parse_qs(urlsplit(self.authorize(p)['redirect_uri']).query)['code'][0]
        data=self.request('/api/connector/token',{'app_id':'test-app','code':code})['data']
        return {'X-DB-ACCESS-TOKEN':data['access_token'],'X-DB-OPENID':openid},data,code

    def rpc(self,headers,name,args=None):
        return self.request('/api/connector/mcp',{'jsonrpc':'2.0','id':1,'method':'tools/call',
            'params':{'name':name,'arguments':args or {}}},headers)['result']

    def image(self,color='orange'):
        f=io.BytesIO();Image.new('RGB',(24,24),color).save(f,format='PNG')
        return base64.b64encode(f.getvalue()).decode()

    def test_authorization_is_single_use_redirect_checked_and_opt_in(self):
        p=self.participant()
        for changes in ({'consent':False},{'redirect_uri':'https://attacker.example'},{'app_id':'wrong'}):
            with self.assertRaises(urllib.error.HTTPError): self.authorize(p,**changes)
        headers,_,code=self.connect(p)
        self.assertNotIn('isError',self.rpc(headers,'xiaoyu_status'))
        with self.assertRaises(urllib.error.HTTPError):
            self.request('/api/connector/token',{'app_id':'test-app','code':code})
        with patch.dict(os.environ,{'XIAOYU_CONNECTOR_ENABLED':'0'}):
            with self.assertRaises(urllib.error.HTTPError):self.rpc(headers,'xiaoyu_status')

    def test_identity_binding_refresh_revocation_and_arm_lock(self):
        p=self.participant(); h,t,_=self.connect(p)
        self.rpc(h,'xiaoyu_status')
        with self.assertRaises(urllib.error.HTTPError): self.rpc({**h,'X-DB-OPENID':'other'},'xiaoyu_status')
        q=self.participant(); other,_,_=self.connect(q)
        with self.assertRaises(urllib.error.HTTPError):self.rpc(other,'xiaoyu_status')
        fresh=self.request('/api/connector/refresh',{'app_id':'test-app','refresh_token':t['refresh_token']})['data']
        with self.assertRaises(urllib.error.HTTPError):self.rpc(h,'xiaoyu_status')
        h['X-DB-ACCESS-TOKEN']=fresh['access_token'];self.rpc(h,'xiaoyu_status')
        with server.connect() as c:c.execute('UPDATE users SET arm="T" WHERE id=?',(p['id'],))
        with self.assertRaises(urllib.error.HTTPError):self.rpc(h,'xiaoyu_status')
        with server.connect() as c:c.execute('UPDATE users SET arm="M" WHERE id=?',(p['id'],))
        self.request('/api/admin/connector/revoke',{'user':p['id']},{'X-Admin-Key':server.ADMIN})
        with self.assertRaises(urllib.error.HTTPError):self.rpc(h,'xiaoyu_status')

    def test_import_is_not_learning_and_private_assets_cannot_cross_users(self):
        p=self.participant();q=self.participant()
        a,_,_=self.connect(p);b,_,_=self.connect(q,'open-b')
        imported=self.rpc(a,'xiaoyu_library_import',{'image':self.image()})
        aid=json.loads(imported['content'][0]['text'])['id']
        self.model.assert_not_called();self.learn.assert_not_called()
        with server.connect() as c:
            self.assertEqual(c.execute('SELECT count(*) FROM messages').fetchone()[0],0)
            self.assertEqual(c.execute('SELECT count(*) FROM memories').fetchone()[0],0)
            self.assertEqual(stickers.candidates(c,p['id'],'any','M'),[])
        self.assertEqual(json.loads(self.rpc(b,'xiaoyu_library_list')['content'][0]['text'])['items'],[])
        bad=self.rpc(b,'xiaoyu_chat',{'turn_id':'1','asset_id':aid})
        self.assertTrue(bad['isError'])
        for forbidden in ({'arm':'D'},{'user':p['id']},{'behavior':{'response_ms':20}}):
            self.assertTrue(self.rpc(a,'xiaoyu_chat',{'turn_id':'1','text':'你好',**forbidden})['isError'])

    def test_chat_retry_recovery_and_no_fabricated_exposure(self):
        p=self.participant('T');h,_,_=self.connect(p)
        args={'turn_id':'host-message-1','text':'今天好累'}
        first=self.rpc(h,'xiaoyu_chat',args)
        self.assertNotIn('isError',first)
        self.assertEqual(first,self.rpc(h,'xiaoyu_chat',args))
        self.assertEqual(self.model.call_count,1)
        self.assertTrue(self.rpc(h,'xiaoyu_chat',{**args,'text':'不同消息'})['isError'])
        with server.connect() as c:
            self.assertEqual(c.execute('SELECT count(*) FROM messages WHERE user=?',(p['id'],)).fetchone()[0],2)
            self.assertEqual(c.execute('SELECT count(*) FROM message_views').fetchone()[0],0)
            self.assertEqual(c.execute('SELECT count(*) FROM image_views').fetchone()[0],0)
            self.assertEqual(c.execute('SELECT count(*) FROM activity_intervals').fetchone()[0],0)
            self.assertIsNone(c.execute('SELECT response_ms FROM message_sends').fetchone()[0])
            c.execute('UPDATE connector_turns SET state="pending",outcome=NULL WHERE user=?',(p['id'],))
        self.assertEqual(first,self.rpc(h,'xiaoyu_chat',args))
        self.assertEqual(self.model.call_count,1)
        self.assertFalse(first['structuredContent']['display_verified'])
        q=self.participant();qheaders,_,_=self.connect(q,'open-b')
        self.assertEqual(json.loads(self.rpc(qheaders,'xiaoyu_history')['content'][0]['text'])['messages'],[])

    def test_mcp_initialization_and_per_user_history_survives_reauthorization(self):
        p=self.participant();h,_,_=self.connect(p)
        init=self.request('/api/connector/mcp',{'jsonrpc':'2.0','id':2,'method':'initialize','params':{'protocolVersion':'2025-03-26'}},h)
        self.assertEqual(init['result']['protocolVersion'],'2025-03-26')
        self.rpc(h,'xiaoyu_chat',{'turn_id':'one','text':'我喜欢看电影'})
        new,_,_=self.connect(p)
        history=json.loads(self.rpc(new,'xiaoyu_history')['content'][0]['text'])['messages']
        self.assertEqual(history[0]['text'],'我喜欢看电影')
        listing=self.request('/api/connector/mcp',{'jsonrpc':'2.0','id':3,'method':'tools/list'},new)['result']['tools']
        self.assertEqual(len(listing),5)
        self.assertTrue(all('arm' not in t['inputSchema']['properties'] for t in listing))

    def test_pending_request_prevents_reordering(self):
        p=self.participant();h,_,_=self.connect(p)
        with server.connect() as c:
            c.execute('INSERT INTO connector_turns VALUES(?,?,?,?,?,NULL)',(p['id'],'pending','hash','pending',server.now()))
        self.assertTrue(self.rpc(h,'xiaoyu_chat',{'turn_id':'new','text':'后来呢'})['isError'])
        self.model.assert_not_called()

    def test_four_arm_allocation_is_balanced_without_reassigning_legacy(self):
        old=[self.participant('T') for _ in range(3)]
        assigned=[self.participant('random4')['arm'] for _ in range(8)]
        self.assertEqual({x:assigned.count(x) for x in ('T','G','M','D')},{'T':2,'G':2,'M':2,'D':2})
        with server.connect() as c:
            for p in old:self.assertEqual(c.execute('SELECT arm FROM users WHERE id=?',(p['id'],)).fetchone()[0],'T')

    def test_common_text_policy_and_learning_only_for_m_and_d(self):
        prompts=[]
        for arm in ('T','G','M','D'):
            p=self.participant(arm);h,_,_=self.connect(p,'open-'+arm)
            self.model.reset_mock();self.learn.reset_mock()
            response=self.rpc(h,'xiaoyu_chat',{'turn_id':'same-context','text':'下班啦，想休息一下'})
            self.assertNotIn('isError',response)
            prompts.append(self.model.call_args[0][0][0]['content'])
            if arm in ('T','G'):self.learn.assert_not_called()
            else:self.learn.assert_called_once()
            with server.connect() as c:
                event=c.execute('SELECT payload FROM events WHERE user=? AND kind="generation"',(p['id'],)).fetchone()
                self.assertEqual(json.loads(event[0])['text_policy'],'shared-companion-v1')
        self.assertTrue(all(x.startswith(server.PERSONA) for x in prompts))
        # No arm names/candidate style instructions enter the text generator.
        normalized=[s.split('当前用户本地时间：')[0] for s in prompts]
        self.assertEqual(len(set(normalized)),1)

    def test_divergent_pool_requires_observed_style_and_semantic_selection(self):
        admin={'X-Admin-Key':server.ADMIN}
        semantic={'description':'写实猫猫庆祝','meaning':'开心','situations':'完成工作','avoid':'难过','style':'写实、含蓄'}
        with patch('server.model',return_value=(json.dumps(semantic),{})):
            g=self.request('/api/admin/generic-library/import',{'image':self.image(),'pool':'generic'},admin)['id']
            d=self.request('/api/admin/generic-library/import',{'image':self.image('blue'),'pool':'divergent','style':'写实、含蓄'},admin)['id']
        p=self.participant('D')
        with server.connect() as c:
            self.assertEqual([x['asset_id'] for x in generic_stickers.candidates(c,set())],[g])
            self.assertEqual(stickers.candidates(c,p['id'],'s','D'),[])
            c.execute('INSERT INTO assets VALUES(?,?,?,?,?)',('mine',p['id'],self.image('pink'),'黑白简笔人物，无文字，夸张卡通笑脸','{}'))
            self.assertEqual(stickers.candidates(c,p['id'],'s','D'),[])
            c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?)',('source',p['id'],'s','user','太好啦，值得庆祝','mine','[]','真心开心，值得庆祝',server.now()))
            for target in (g,d):
                c.execute('INSERT INTO sticker_intent_reviews VALUES(?,?,?)',(target,json.dumps(['laugh']),'fixture'))
            self.assertEqual({x['asset_id'] for x in stickers.candidates(c,p['id'],'s','D')},{g,d})
            calls=[]
            def choose(messages,**kwargs):
                from test_divergence import choice, profile
                calls.append(messages)
                a='黑白简笔人物，无文字，夸张卡通笑脸';b='写实猫猫庆祝'
                return json.dumps(choice(d,'mine',a,b,
                    profile('line_drawing','human','none','monochrome',a),
                    profile('photo','animal','unknown','unknown',b))),{}
            with patch('sticker_style_choice.distances',return_value={g:.71,d:1.1}):
                aid,audit=stickers.choose(c,p['id'],'s','D','太好啦，值得庆祝','太好了',None,choose,rng=lambda n:0)
            self.assertEqual(aid,d)
            self.assertEqual(audit['source'],'divergent')
            self.assertIn('夸张卡通笑脸',calls[0][1]['content'])
            self.assertIn('语义',calls[0][0]['content'])

    def test_original_image_return_is_not_a_public_url_or_seen_event(self):
        p=self.participant();h,_,_=self.connect(p)
        imported=self.rpc(h,'xiaoyu_library_import',{'image':self.image()})
        aid=json.loads(imported['content'][0]['text'])['id']
        with server.connect() as c:
            c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?)',('old',p['id'],'s','user','好耶',aid,'[]','',server.now()))
        with patch('server.stickers.choose',return_value=(aid,{})):
            reply=self.rpc(h,'xiaoyu_chat',{'turn_id':'celebrate','text':'完成了'})
        self.assertEqual(reply['content'][1]['type'],'image')
        image=Image.open(io.BytesIO(base64.b64decode(reply['content'][1]['data'])))
        self.assertEqual(image.format,'JPEG')
        self.assertFalse(reply['structuredContent']['display_verified'])

    def test_mobile_setup_requires_login_and_keeps_library_isolated(self):
        with self.assertRaises(urllib.error.HTTPError):
            self.request('/api/mobile/setup')
        p=self.participant();q=self.participant()
        p_headers={'Authorization':'Bearer '+p['token']}
        q_headers={'Authorization':'Bearer '+q['token']}
        self.request('/api/consent',{},p_headers)
        self.request('/api/library/import',{'image':self.image()},p_headers)
        a=self.request('/api/mobile/setup',headers=p_headers)
        b=self.request('/api/mobile/setup',headers=q_headers)
        self.assertEqual((a['library_count'],b['library_count']),(1,0))
        self.assertNotIn('arm',a)
        self.assertNotIn('openid',a)
        self.assertFalse(a['entry_ready'])
        self.assertIsNone(a['entry_url'])
        self.model.assert_not_called();self.learn.assert_not_called()

    def test_mobile_link_requires_explicit_verified_configuration(self):
        p=self.participant();headers={'Authorization':'Bearer '+p['token']}
        with patch.dict(os.environ,{'XIAOYU_MOBILE_ENTRY_URL':'https://www.doubao.com/share/test-fixture','XIAOYU_MOBILE_VERIFIED':'0'}):
            self.assertFalse(self.request('/api/mobile/setup',headers=headers)['entry_ready'])
        for bad in ('javascript:alert(1)','https://evil.example','https://www.doubao.com.evil.example','https://user@www.doubao.com','https://www.doubao.com:bad/'):
            with patch.dict(os.environ,{'XIAOYU_MOBILE_ENTRY_URL':bad,'XIAOYU_MOBILE_VERIFIED':'1'}):
                self.assertIsNone(self.request('/api/mobile/setup',headers=headers)['entry_url'])
        with patch.dict(os.environ,{'XIAOYU_MOBILE_ENTRY_URL':'https://www.doubao.com/share/test-fixture','XIAOYU_MOBILE_VERIFIED':'1'}):
            self.assertTrue(self.request('/api/mobile/setup',headers=headers)['entry_ready'])
            with patch.dict(os.environ,{'XIAOYU_CONNECTOR_ENABLED':'0'}):
                self.assertFalse(self.request('/api/mobile/setup',headers=headers)['entry_ready'])


if __name__=='__main__':unittest.main()
