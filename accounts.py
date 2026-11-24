"""Researcher-issued participant accounts and revocable device sessions."""
import hashlib
import secrets
import time

DEFAULT_PASSWORD = '66666666'

def digest(password, salt):
    return hashlib.pbkdf2_hmac('sha256', password.encode(), bytes.fromhex(salt), 260000).hex()

def create(c, uid):
    existing = c.execute('SELECT username FROM accounts WHERE user=?', (uid,)).fetchone()
    if existing:
        return existing[0]
    username = 'P-' + secrets.token_hex(6).upper()
    salt = secrets.token_hex(16)
    c.execute('INSERT INTO accounts(user,username,salt,password_hash) VALUES(?,?,?,?)',
              (uid, username, salt, digest(DEFAULT_PASSWORD, salt)))
    return username

def init(c):
    c.executescript('''
    CREATE TABLE IF NOT EXISTS accounts(user TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,username TEXT UNIQUE NOT NULL,salt TEXT NOT NULL,password_hash TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS login_sessions(token_hash TEXT PRIMARY KEY,user TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,expires REAL NOT NULL);
    CREATE TABLE IF NOT EXISTS admin_sessions(token_hash TEXT PRIMARY KEY,username TEXT NOT NULL,expires REAL NOT NULL);
    CREATE TABLE IF NOT EXISTS login_limits(key TEXT PRIMARY KEY,failures INTEGER NOT NULL,expires REAL NOT NULL);
    ''')
    for row in c.execute('SELECT id FROM users WHERE id NOT IN (SELECT user FROM accounts)').fetchall():
        create(c, row[0])

def login(c, username, password, ip, admin_key=None):
    username = str(username).strip().upper()
    password = str(password)
    if len(username) > 50 or len(password) > 128:
        raise PermissionError('账号或密码不正确')
    stamp = time.time()
    keys = [hashlib.sha256(('ip:' + ip).encode()).hexdigest(), hashlib.sha256(('account:' + username).encode()).hexdigest()]
    c.execute('BEGIN IMMEDIATE')
    c.execute('DELETE FROM login_limits WHERE expires<?', (stamp,))
    c.execute('DELETE FROM login_sessions WHERE expires<?', (stamp,))
    c.execute('DELETE FROM admin_sessions WHERE expires<?', (stamp,))
    for key, limit in zip(keys, (30, 8)):
        row = c.execute('SELECT failures FROM login_limits WHERE key=?', (key,)).fetchone()
        if row and row[0] >= limit:
            c.commit()
            raise PermissionError('尝试次数过多，请 10 分钟后再试')
    row = c.execute('SELECT * FROM accounts WHERE username=?', (username,)).fetchone()
    candidate = digest(password, row['salt'] if row else '0' * 32)
    is_admin = username == 'ADMIN' and bool(admin_key) and secrets.compare_digest(hashlib.sha256(password.encode()).digest(), hashlib.sha256((admin_key or '').encode()).digest())
    if not is_admin and (not row or not secrets.compare_digest(candidate, row['password_hash'])):
        for key in keys:
            c.execute('INSERT INTO login_limits VALUES(?,1,?) ON CONFLICT(key) DO UPDATE SET failures=failures+1', (key, stamp + 600))
        c.commit()
        raise PermissionError('账号或密码不正确')
    c.execute('DELETE FROM login_limits WHERE key=?', (keys[1],))
    token = secrets.token_urlsafe(32)
    if is_admin:
        c.execute('INSERT INTO admin_sessions VALUES(?,?,?)', (hashlib.sha256(token.encode()).hexdigest(), username, stamp + 12*3600))
    else:
        c.execute('INSERT INTO login_sessions VALUES(?,?,?)', (hashlib.sha256(token.encode()).hexdigest(), row['user'], stamp + 30*86400))
    c.commit()
    return {'token':token, 'username':username, 'role':'admin' if is_admin else 'participant'}

def user(c, token):
    return c.execute('SELECT users.* FROM users JOIN login_sessions ON users.id=login_sessions.user WHERE token_hash=? AND expires>?',
                     (hashlib.sha256(token.encode()).hexdigest(),time.time())).fetchone()

def admin(c, token):
    return c.execute('SELECT username FROM admin_sessions WHERE token_hash=? AND expires>?',
                     (hashlib.sha256(token.encode()).hexdigest(), time.time())).fetchone()

def logout(c, token):
    c.execute('UPDATE users SET token=? WHERE token=?', (secrets.token_urlsafe(24), token))
    c.execute('DELETE FROM login_sessions WHERE token_hash=?', (hashlib.sha256(token.encode()).hexdigest(),))
    c.execute('DELETE FROM admin_sessions WHERE token_hash=?', (hashlib.sha256(token.encode()).hexdigest(),))
    c.commit()
