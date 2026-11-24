"""Opt-in MCP bridge for Xiaoyu. All study state stays in the existing backend.

The Doubao developer console integration still requires end-to-end verification.
No tool can choose a participant, arm, or fabricate browser exposure telemetry.
"""
import base64
import hashlib
import json
import os
import re
import secrets
import time
from datetime import datetime, timezone
from urllib.parse import urlencode, urlsplit, urlunsplit, parse_qsl

NAME = 'xiaoyu-companion'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def init(c):
    c.executescript('''
    CREATE TABLE IF NOT EXISTS connector_codes(
        hash TEXT PRIMARY KEY,user TEXT NOT NULL,app_id TEXT NOT NULL,
        expires REAL NOT NULL,used INTEGER NOT NULL DEFAULT 0,challenge TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS connector_grants(
        id TEXT PRIMARY KEY,user TEXT NOT NULL,app_id TEXT NOT NULL,arm TEXT NOT NULL,
        access_hash TEXT UNIQUE NOT NULL,refresh_hash TEXT UNIQUE NOT NULL,
        access_expires REAL NOT NULL,refresh_expires REAL NOT NULL,revoked INTEGER NOT NULL DEFAULT 0);
    CREATE TABLE IF NOT EXISTS connector_bindings(
        user TEXT NOT NULL,app_id TEXT NOT NULL,openid TEXT NOT NULL,created TEXT NOT NULL,
        PRIMARY KEY(user,app_id),UNIQUE(app_id,openid));
    CREATE TABLE IF NOT EXISTS connector_turns(
        user TEXT NOT NULL,turn TEXT NOT NULL,input_hash TEXT NOT NULL,
        state TEXT NOT NULL,created TEXT NOT NULL,outcome TEXT,
        PRIMARY KEY(user,turn));
    ''')


def config():
    if os.environ.get('XIAOYU_CONNECTOR_ENABLED') != '1':
        raise PermissionError('研究连接器尚未开放')
    app = os.environ.get('XIAOYU_CONNECTOR_APP_ID', '')
    redirects = json.loads(os.environ.get('XIAOYU_CONNECTOR_REDIRECT_URIS', '[]'))
    if not app or not isinstance(redirects, list) or not redirects:
        raise PermissionError('研究连接器尚未配置应用与回调地址')
    return app, redirects


def mobile_setup(c, user):
    """Phone onboarding status. A configured link is not evidence of a verified app."""
    url = os.environ.get('XIAOYU_MOBILE_ENTRY_URL', '').strip()
    try:
        parsed = urlsplit(url)
        allowed = parsed.scheme == 'https' and parsed.hostname in ('www.doubao.com','doubao.com') and not parsed.username and not parsed.password and parsed.port in (None,443)
    except ValueError:
        allowed = False
    try:
        config()
        configured = True
    except (PermissionError,ValueError):
        configured = False
    ready = configured and allowed and os.environ.get('XIAOYU_MOBILE_VERIFIED') == '1'
    app = os.environ.get('XIAOYU_CONNECTOR_APP_ID','')
    linked = bool(c.execute('''SELECT 1 FROM connector_bindings b JOIN connector_grants g
        ON g.user=b.user AND g.app_id=b.app_id WHERE b.user=? AND b.app_id=?
        AND g.revoked=0 AND g.refresh_expires>? LIMIT 1''', (user['id'],app,time.time())).fetchone())
    account = c.execute('SELECT username FROM accounts WHERE user=?',(user['id'],)).fetchone()
    return {'username':account[0] if account else '', 'consent':bool(user['consent']),
            'library_count':c.execute('SELECT count(*) FROM assets WHERE user=?',(user['id'],)).fetchone()[0],
            'connected':linked,'entry_ready':ready,'entry_url':url if ready else None}


def authorize(c, user, body):
    app, redirects = config()
    redirect = body.get('redirect_uri', '')
    parsed = urlsplit(redirect)
    if body.get('app_id') != app or redirect not in redirects or parsed.scheme != 'https' or parsed.fragment:
        raise PermissionError('应用或回调地址不匹配')
    if body.get('consent') is not True:
        raise PermissionError('请先阅读并同意研究连接说明')
    state = body.get('state', '')
    challenge = body.get('code_challenge', '')
    if not isinstance(state, str) or len(state) > 1024:
        raise ValueError('state 无效')
    if challenge and (body.get('code_challenge_method') != 'S256' or not re.fullmatch(r'[A-Za-z0-9_-]{43}', challenge)):
        raise ValueError('只支持 S256 PKCE')
    code = secrets.token_urlsafe(32)
    c.execute('INSERT INTO connector_codes VALUES(?,?,?,?,0,?)', (digest(code), user['id'], app, time.time()+300, challenge))
    c.execute('UPDATE users SET consent=CASE WHEN consent="" THEN ? ELSE consent END WHERE id=?', (stamp(), user['id']))
    # The allowlisted URI is retained exactly except for OAuth response params.
    query = [(k,v) for k,v in parse_qsl(parsed.query, keep_blank_values=True) if k not in ('code','state')]
    target = urlunsplit(parsed._replace(query=urlencode(query+[('code',code),('state',state)])))
    c.commit()
    return {'redirect_uri': target}


def token(c, body, refresh=False):
    app, _ = config()
    if body.get('app_id') != app:
        raise PermissionError('应用不匹配')
    c.execute('BEGIN IMMEDIATE')
    t = time.time()
    if refresh:
        raw = body.get('refresh_token', '')
        old = c.execute('SELECT * FROM connector_grants WHERE refresh_hash=? AND app_id=? AND revoked=0 AND refresh_expires>?', (digest(str(raw)), app, t)).fetchone()
        if not old: raise PermissionError('授权已过期，请重新连接')
        uid, arm, grant = old['user'], old['arm'], old['id']
    else:
        code = c.execute('SELECT * FROM connector_codes WHERE hash=? AND app_id=? AND used=0 AND expires>?', (digest(str(body.get('code',''))), app, t)).fetchone()
        if not code: raise PermissionError('授权码无效或已使用')
        if code['challenge']:
            verifier = str(body.get('code_verifier', ''))
            challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b'=').decode()
            if not secrets.compare_digest(challenge, code['challenge']): raise PermissionError('PKCE 校验失败')
        uid, grant = code['user'], secrets.token_hex(16)
        row = c.execute('SELECT arm FROM users WHERE id=?', (uid,)).fetchone()
        if not row: raise PermissionError('被试账号不存在')
        arm = row['arm']
        c.execute('UPDATE connector_codes SET used=1 WHERE hash=?', (code['hash'],))
    access, refresh_value = secrets.token_urlsafe(32), secrets.token_urlsafe(40)
    c.execute('''INSERT INTO connector_grants VALUES(?,?,?,?,?,?,?,?,0)
        ON CONFLICT(id) DO UPDATE SET access_hash=excluded.access_hash,
        refresh_hash=excluded.refresh_hash,access_expires=excluded.access_expires,
        refresh_expires=excluded.refresh_expires''',
        (grant,uid,app,arm,digest(access),digest(refresh_value),t+7200,t+30*86400))
    c.commit()
    return {'code':0,'msg':'success','data':{'access_token':access,'expires_in':7200,
            'refresh_token':refresh_value,'refresh_token_expires_in':30*86400}}


def principal(c, headers):
    app, _ = config()
    raw = headers.get('X-DB-ACCESS-TOKEN', '')
    openid = headers.get('X-DB-OPENID', '')
    if not raw or not isinstance(openid, str) or not 1 <= len(openid) <= 200:
        raise PermissionError('请通过研究账号授权连接')
    c.execute('BEGIN IMMEDIATE')
    grant = c.execute('SELECT * FROM connector_grants WHERE access_hash=? AND app_id=? AND revoked=0 AND access_expires>?', (digest(raw),app,time.time())).fetchone()
    if not grant: raise PermissionError('连接授权无效或已过期')
    user = c.execute('SELECT * FROM users WHERE id=?', (grant['user'],)).fetchone()
    if not user or user['arm'] != grant['arm'] or not user['consent']:
        raise PermissionError('研究账号状态已变化，请联系研究者')
    binding = c.execute('SELECT openid FROM connector_bindings WHERE user=? AND app_id=?', (user['id'], app)).fetchone()
    other = c.execute('SELECT user FROM connector_bindings WHERE app_id=? AND openid=?', (app,openid)).fetchone()
    if (binding and binding['openid'] != openid) or (other and other['user'] != user['id']):
        raise PermissionError('该连接与研究账号绑定不一致')
    c.execute('INSERT OR IGNORE INTO connector_bindings VALUES(?,?,?,?)', (user['id'],app,openid,stamp()))
    c.commit()
    return dict(user)


def tool(name, description, properties=None, required=()):
    return {'name':name,'description':description,'inputSchema':{'type':'object',
        'properties':properties or {},'required':list(required),'additionalProperties':False}}


TOOLS = [
    tool('xiaoyu_status', '检查当前研究账号连接、问卷日程；不返回隐藏分组。'),
    tool('xiaoyu_chat', '把用户实际发出的一条研究聊天交给小屿。后台生成统一风格文字并决定附图；原样呈现，不自行重写。重试须保留 turn_id。', {
        'turn_id':{'type':'string','minLength':1,'maxLength':100},
        'text':{'type':'string','maxLength':2000},
        'asset_id':{'type':'string','maxLength':100},
        'timezone_offset':{'type':'integer','minimum':-840,'maximum':720}}, ('turn_id',)),
    tool('xiaoyu_library_import', '保存一张用户授权导入的原始表情图片；仅入库，不分析、不学习、不视为发送。图片必须来自真实附件字节，不可编造 base64。', {
        'image':{'type':'string','minLength':1,'maxLength':8100000}}, ('image',)),
    tool('xiaoyu_library_list', '列出当前被试的个人图库，可选择 asset_id；不会触发学习。'),
    tool('xiaoyu_history', '恢复当前研究账号近期已记录的聊天内容，不读取其他豆包会话。', {
        'limit':{'type':'integer','minimum':1,'maximum':50}}),
]


def validate(name, args):
    spec = next((t for t in TOOLS if t['name']==name), None)
    if not spec: raise ValueError('未知工具')
    schema = spec['inputSchema']
    if not isinstance(args, dict) or set(args)-set(schema['properties']) or set(schema['required'])-set(args):
        raise ValueError('工具参数无效；账号和分组只能由后台决定')
    for key, value in args.items():
        p = schema['properties'][key]
        if p['type']=='string' and (not isinstance(value,str) or not p.get('minLength',0)<=len(value)<=p.get('maxLength',10**9)):
            raise ValueError('参数长度或类型无效：'+key)
        if p['type']=='integer' and (type(value) is not int or not p['minimum']<=value<=p['maximum']):
            raise ValueError('整数参数无效：'+key)


def result(data):
    return {'content':[{'type':'text','text':json.dumps(data,ensure_ascii=False)}]}


def chat_result(c, user, outcome, display_image):
    content = [{'type':'text','text':outcome['reply']}]
    aid = outcome.get('asset')
    if aid:
        image = c.execute('SELECT image FROM assets WHERE id=? AND user=?', (aid,user)).fetchone()
        if not image:
            image = c.execute('''SELECT image FROM generic_stickers WHERE id=? AND EXISTS(
                SELECT 1 FROM messages WHERE user=? AND role='assistant' AND asset=generic_stickers.id)''', (aid,user)).fetchone()
        if not image: raise ValueError('回复图片不可用，未生成替代图片')
        content.append({'type':'image','mimeType':'image/jpeg','data':base64.b64encode(display_image(image['image'])).decode()})
    return {'content':content,'structuredContent':{
        'reply':outcome['reply'],'asset_id':aid,
        'user_message_id':outcome['message_id'],'assistant_message_id':outcome['assistant_message_id'],
        'delivery_state':'returned_to_host','display_verified':False}}


def recover(c, uid, turn):
    row = c.execute('''SELECT payload FROM events WHERE user=? AND kind='generation'
        AND json_extract(payload,'$.turn')=? ORDER BY id DESC LIMIT 1''', (uid,turn)).fetchone()
    if not row: return None
    payload = json.loads(row[0])
    msg = c.execute("SELECT * FROM messages WHERE id=? AND user=? AND role='assistant'", (payload.get('assistant_message'),uid)).fetchone()
    if not msg: return None
    return {'reply':msg['text'],'asset':msg['asset'],'message_id':payload['message'],'assistant_message_id':msg['id']}


def call(name, args, user, connect, internal_request, display_image):
    validate(name,args)
    uid = user['id']
    with connect() as c:
        if name == 'xiaoyu_status':
            import behavior
            return result({'connected':True,'study':behavior.schedule(c,uid),
                'observed_metrics':['message_content','message_count','asset_id','server_received_at'],
                'unavailable_metrics':['foreground_duration','response_latency_from_visible','actual_image_exposure']})
        if name == 'xiaoyu_library_list':
            items = [dict(r) for r in c.execute('''SELECT a.id,EXISTS(
                SELECT 1 FROM messages m WHERE m.user=a.user AND m.role='user' AND m.asset=a.id) AS used_in_chat
                FROM assets a WHERE a.user=? ORDER BY a.rowid DESC''', (uid,))]
            return result({'items':items})
        if name == 'xiaoyu_history':
            rows = [dict(r) for r in c.execute('SELECT id,role,text,asset,created FROM messages WHERE user=? ORDER BY rowid DESC LIMIT ?', (uid,args.get('limit',20)))][::-1]
            return result({'messages':rows})
        if name == 'xiaoyu_library_import':
            status, data = internal_request('/api/library/import', {'image':args['image']}, user)
            if status != 200: raise ValueError(data.get('error','导入失败'))
            return result(data)
        text, asset = args.get('text','').strip(), args.get('asset_id') or None
        if not text and not asset: raise ValueError('请输入消息或选择已导入图片')
        if asset and not c.execute('SELECT id FROM assets WHERE id=? AND user=?',(asset,uid)).fetchone():
            raise PermissionError('图片不属于当前被试')
        # A provider retry uses the same request ID; never duplicate a successful turn.
        turn = 'connector:'+digest(args['turn_id'])
        body_hash = digest(json.dumps({'text':text,'asset':asset,'timezone_offset':args.get('timezone_offset',-480)}, sort_keys=True,ensure_ascii=False))
        c.execute('BEGIN IMMEDIATE')
        previous = c.execute('SELECT * FROM connector_turns WHERE user=? AND turn=?',(uid,turn)).fetchone()
        if previous:
            if previous['input_hash'] != body_hash: raise ValueError('同一 turn_id 不能对应不同消息')
            saved = json.loads(previous['outcome']) if previous['outcome'] else recover(c,uid,turn)
            if saved and saved.get('reply'):
                c.execute('UPDATE connector_turns SET state="done",outcome=? WHERE user=? AND turn=?',(json.dumps(saved,ensure_ascii=False),uid,turn))
                c.commit()
                return chat_result(c,uid,saved,display_image)
            raise ValueError('该消息仍在处理或上次未完成，请勿换编号自动重发')
        if c.execute('SELECT 1 FROM connector_turns WHERE user=? AND state="pending"',(uid,)).fetchone():
            raise ValueError('上一条消息仍在处理，请稍后重试')
        c.execute('INSERT INTO connector_turns VALUES(?,?,?,?,?,NULL)',(uid,turn,body_hash,'pending',stamp()))
        sid = c.execute('SELECT id FROM sessions WHERE user=? ORDER BY rowid DESC LIMIT 1',(uid,)).fetchone()
        if sid:
            sid=sid[0]
        else:
            sid=secrets.token_hex(12)
            c.execute('INSERT INTO sessions VALUES(?,?,?)',(sid,uid,stamp()))
        c.commit()
    try:
        status, data = internal_request('/api/chat', {'session':sid,'text':text,'asset':asset,'turn':turn,'timezone_offset':args.get('timezone_offset',-480)}, user)
    except Exception:
        # Unknown outcome stays pending: recovery checks the committed generation.
        raise ValueError('连接中断，结果待核验；重试须保留原 turn_id')
    with connect() as c:
        c.execute('UPDATE connector_turns SET state=?,outcome=? WHERE user=? AND turn=?', ('done' if status==200 else 'failed',json.dumps(data,ensure_ascii=False),uid,turn))
        c.execute('INSERT INTO events(user,kind,payload,created) VALUES(?,?,?,?)', (uid,'connector_returned',json.dumps({'turn':turn,'status':status,'message':data.get('assistant_message_id'),'display_verified':False}),stamp()))
        c.commit()
        if status != 200: raise ValueError(data.get('error','聊天请求未完成'))
        return chat_result(c,uid,data,display_image)


def rpc(body, user, connect, internal_request, display_image):
    if not isinstance(body,dict) or body.get('jsonrpc')!='2.0':
        return {'jsonrpc':'2.0','id':None,'error':{'code':-32600,'message':'Invalid Request'}}
    method, rid = body.get('method'), body.get('id')
    if method in ('notifications/initialized','notifications/cancelled') and 'id' not in body:
        return None
    if method=='initialize':
        requested=(body.get('params') or {}).get('protocolVersion')
        value={'protocolVersion':requested if requested in ('2024-11-05','2025-03-26','2025-06-18') else '2025-03-26',
               'capabilities':{'tools':{}},'serverInfo':{'name':NAME,'version':'0.1.0'}}
    elif method=='ping': value={}
    elif method=='tools/list': value={'tools':TOOLS}
    elif method=='tools/call':
        try:
            params=body.get('params') or {}
            value=call(params.get('name'),params.get('arguments',{}),user,connect,internal_request,display_image)
        except (ValueError,PermissionError) as e:
            value={'content':[{'type':'text','text':str(e)}],'isError':True}
    else:
        return {'jsonrpc':'2.0','id':rid,'error':{'code':-32601,'message':'Method not found'}}
    return {'jsonrpc':'2.0','id':rid,'result':value}
