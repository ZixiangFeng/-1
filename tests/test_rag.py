import sqlite3, tempfile, unittest
from pathlib import Path
from rag import RagStore


class TinyEmbedder:
    """Deterministic test vectors; never presented as real semantic model evidence."""

    def embed(self, texts):
        for text in texts:
            yield [1.0, 3.0 if "论文" in text else 0.0, 3.0 if "生日" in text else 0.0]

    query_embed = embed


class VectorIsolation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "vectors"
        self.rag = RagStore(self.path, self.temp.name, TinyEmbedder())
        self.c = sqlite3.connect(":memory:")
        self.c.row_factory = sqlite3.Row
        self.c.executescript(
            "CREATE TABLE memories(id TEXT,user TEXT,expression TEXT,meaning TEXT,context TEXT,status TEXT); CREATE TABLE assets(id TEXT,user TEXT,description TEXT);"
        )
        self.c.executemany(
            "INSERT INTO memories VALUES(?,?,?,?,?,?)",
            [
                ("paper", "a", "emoji:🙂", "无奈", "论文", "confirmed"),
                ("birthday", "a", "emoji:🙂", "开心", "生日", "confirmed"),
                ("private", "b", "emoji:🙂", "秘密", "论文", "confirmed"),
                ("candidate", "a", "emoji:🙂", "猜测", "论文", "candidate"),
            ],
        )

    def tearDown(self):
        self.rag.close()
        self.c.close()
        self.temp.cleanup()

    def test_context_ranking_and_user_scope(self):
        r = self.rag.context(
            self.c, "a", "memory", "论文又要改了🙂", None, ["emoji:🙂"]
        )
        self.assertEqual(r["memories"][0]["id"], "paper")
        self.assertEqual({x["id"] for x in r["memories"]}, {"paper", "birthday"})
        self.assertTrue(r["policies"])

    def test_control_and_superseded(self):
        self.rag.context(self.c, "a", "memory", "论文🙂", None, ["emoji:🙂"])
        self.c.execute("UPDATE memories SET status='superseded' WHERE id='paper'")
        r = self.rag.context(self.c, "a", "memory", "论文🙂", None, ["emoji:🙂"])
        self.assertNotIn("paper", [x["id"] for x in r["memories"]])
        r = self.rag.context(self.c, "a", "control", "论文🙂", None, ["emoji:🙂"])
        self.assertEqual(r["memories"], [])
        self.assertTrue(r["policies"])

    def test_persistent_database(self):
        self.rag.context(self.c, "a", "memory", "论文🙂", None, ["emoji:🙂"])
        self.rag.close()
        self.rag = RagStore(self.path, self.temp.name, TinyEmbedder())
        self.assertIn(
            "paper", [x["id"] for x in self.rag.search("论文", "a", "memory")]
        )

    def test_image_ownership_before_search(self):
        self.c.execute("INSERT INTO assets VALUES(?,?,?)", ("secret", "b", "图片"))
        with self.assertRaises(ValueError):
            self.rag.context(self.c, "a", "memory", "论文", "secret", ["image:secret"])

    def test_inferred_episodes_only_experiment_and_stale_removal(self):
        self.c.execute("INSERT INTO memories VALUES(?,?,?,?,?,?)", ("auto", "a", "emoji:🙂", "自嘲", "论文", "inferred"))
        result=self.rag.context(self.c,"a","memory","论文🙂",None,["emoji:🙂"])
        self.assertIn("auto",[m["id"] for m in result["memories"]])
        self.assertEqual(self.rag.context(self.c,"a","control","论文🙂",None,["emoji:🙂"])["memories"],[])
        self.c.execute("UPDATE memories SET status='superseded' WHERE id='auto'")
        self.assertNotIn("auto",[m["id"] for m in self.rag.context(self.c,"a","memory","论文🙂",None,["emoji:🙂"])["memories"]])
