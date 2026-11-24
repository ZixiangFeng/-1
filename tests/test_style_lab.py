import base64
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from PIL import Image
import style_lab as lab


class RankingTests(unittest.TestCase):
    def test_semantic_gate_cannot_be_overridden_by_large_style_distance(self):
        items = [{'key': 'wrong', 'style_vector': [-1,0], 'semantic_score': .2},
                 {'key': 'right', 'style_vector': [0,1], 'semantic_score': .9}]
        self.assertEqual([r['key'] for r in lab.rank([1,0], items, 'bge', .8, set())], ['right'])

    def test_manual_candidates_and_style_only_are_distinct(self):
        items = [{'key': 'a', 'style_vector': [0,1]}, {'key': 'b', 'style_vector': [-1,0]}]
        self.assertEqual(lab.rank([1,0], items, 'manual', .75, {'a'})[0]['key'], 'a')
        self.assertEqual(lab.rank([1,0], items, 'style_only', .75, set())[0]['style_distance'], 2)
        self.assertEqual(lab.rank([1,0], items, 'manual', .75, set()), [])

    def test_invalid_mode_and_vectors_fail_closed(self):
        with self.assertRaises(ValueError): lab.rank([1], [], 'gse', .7, set())
        with self.assertRaises(ValueError): lab.rank([1], [], 'bge', float('nan'), set())
        with self.assertRaises(ValueError): lab.cosine([1,2], [1])
        with self.assertRaises(ValueError): lab.cosine([0,0], [1,0])


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = Path(self.tmp.name)
        self.c = sqlite3.connect(':memory:')
        self.c.row_factory = sqlite3.Row
        self.c.executescript('''CREATE TABLE users(id TEXT);
        CREATE TABLE generic_stickers(id TEXT,image TEXT,semantics TEXT,enabled INTEGER);
        CREATE TABLE assets(id TEXT,user TEXT,image TEXT,metadata TEXT);
        CREATE TABLE messages(user TEXT,role TEXT,asset TEXT,interpretation TEXT);
        INSERT INTO users VALUES('alice'),('bob');''')
        def image(color):
            out = io.BytesIO();Image.new('RGB',(40,40),color).save(out,'PNG')
            return base64.b64encode(out.getvalue()).decode()
        for id,color in [('a','red'),('b','blue'),('c','green')]:
            self.c.execute('INSERT INTO generic_stickers VALUES(?,?,?,1)', (id,image(color),json.dumps({'meaning':'收到'})))
        self.c.execute('INSERT INTO assets VALUES(?,?,?,?)', ('mine','alice',image('blue'),'{}'))
        self.c.execute('INSERT INTO assets VALUES(?,?,?,?)', ('secret','bob',image('black'),'{}'))
        for r in lab.catalog(self.c,'alice'):
            with lab.cache(self.data) as db:
                db.execute('INSERT OR REPLACE INTO features VALUES(?,?,?,?)', (r['hash'],lab.STYLE_ID,json.dumps([1.,float(ord(r['id'][0]))]),lab.utc()))

    def tearDown(self):
        self.c.close();self.tmp.cleanup()

    def test_private_scope_does_not_mix_users_or_infer_import_meaning(self):
        rows=lab.catalog(self.c,'alice')
        self.assertNotIn('secret',[r['id'] for r in rows])
        self.assertEqual(next(r for r in rows if r['id']=='mine')['meaning'],'')

    def test_exclude_owned_public_copy_and_keep_source_tables_unchanged(self):
        before=list(self.c.iterdump())
        result=lab.search(self.c,self.data,self.data,{'user':'alice','query':'public:a','mode':'style_only'})
        self.assertEqual([r['id'] for r in result['results']],['c'])
        self.assertEqual(result['counts']['excluded_owned_or_duplicate'],2)
        self.assertEqual(before,list(self.c.iterdump()))

    def test_foreign_query_and_missing_index_rejected(self):
        with self.assertRaises(ValueError):lab.search(self.c,self.data,self.data,{'user':'alice','query':'personal:secret'})
        with lab.cache(self.data) as db:db.execute('DELETE FROM features')
        with self.assertRaises(ValueError):lab.search(self.c,self.data,self.data,{'query':'public:a'})


if __name__=='__main__':unittest.main()
