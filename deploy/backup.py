"""Daily consistent SQLite backup. Images and memory evidence are stored in SQLite."""
import datetime
import pathlib
import sqlite3
root = pathlib.Path('/opt/xiaoyu/data')
dest = root / 'backups'
dest.mkdir(mode=0o700, exist_ok=True)
name = dest / ('study-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.sqlite3')
with sqlite3.connect(root / 'study.sqlite3') as source, sqlite3.connect(name) as target:
    source.backup(target)
name.chmod(0o600)
# Keep 30 latest daily copies. Live data is never removed.
for old in sorted(dest.glob('study-*.sqlite3'))[:-30]:
    old.unlink()
