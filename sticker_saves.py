"""Save a received sticker into an account's library without learning from it."""
import base64
import hashlib
import json
from datetime import datetime, timezone


def init(c):
    c.execute('''CREATE TABLE IF NOT EXISTS sticker_saves(
        user TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE, source_asset TEXT NOT NULL,
        source_message TEXT NOT NULL, asset TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
        created TEXT NOT NULL, PRIMARY KEY(user,source_asset))''')


def save(c, user, data, event):
    message = data.get('message')
    if not isinstance(message, str) or not message or len(message) > 100:
        raise ValueError('请选择聊天中收到的表情')
    # Serialize repeated clicks; the source must be an actual received message
    # belonging to this participant, never an arbitrary asset or URL.
    if not c.in_transaction:
        c.execute('BEGIN IMMEDIATE')
    row = c.execute("SELECT asset FROM messages WHERE id=? AND user=? AND role='assistant'",
                    (message, user)).fetchone()
    if not row or not row['asset']:
        raise ValueError('这条表情消息已不可用')
    source = row['asset']
    own = c.execute('SELECT id FROM assets WHERE id=? AND user=?', (source, user)).fetchone()
    if own:
        return {'id':own['id'], 'duplicate':True}
    previous = c.execute('SELECT asset FROM sticker_saves WHERE user=? AND source_asset=?',
                         (user, source)).fetchone()
    if previous:
        return {'id':previous['asset'], 'duplicate':True}
    picture = c.execute('SELECT image FROM generic_stickers WHERE id=?', (source,)).fetchone()
    if not picture:
        raise ValueError('这张表情暂时无法添加')
    encoded = picture['image']
    binary = base64.b64decode(encoded, validate=True)
    # Reuse stored bytes: no further lossy recompression and no model call.
    existing = c.execute('SELECT id FROM assets WHERE user=? AND image=?', (user,encoded)).fetchone()
    aid = existing['id'] if existing else hashlib.sha256(user.encode()+binary).hexdigest()[:24]
    created = datetime.now(timezone.utc).isoformat()
    if not existing:
        c.execute('INSERT INTO assets VALUES(?,?,?,?,?)', (aid,user,encoded,'',json.dumps({
            'library_import':True, 'saved_from_chat':True, 'source_asset':source,
            'source_message':message, 'imported':created},ensure_ascii=False)))
    c.execute('INSERT INTO sticker_saves VALUES(?,?,?,?,?)', (user,source,message,aid,created))
    event(c,user,'library_saved_from_chat',{'asset':aid,'source_asset':source,
        'source_message':message,'created_new':not bool(existing)})
    return {'id':aid, 'duplicate':bool(existing)}
