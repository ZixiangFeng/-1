import sys, tempfile, sqlite3, json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from rag import RagStore

with tempfile.TemporaryDirectory() as tmp:
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(
        "CREATE TABLE memories(id TEXT,user TEXT,expression TEXT,meaning TEXT,context TEXT,status TEXT); CREATE TABLE assets(id TEXT,user TEXT,description TEXT);"
    )
    c.executemany(
        "INSERT INTO memories VALUES(?,?,?,?,?,?)",
        [
            (
                "paper",
                "a",
                "emoji:🙂",
                "强颜欢笑，很无奈",
                "导师反复要求修改毕业论文",
                "confirmed",
            ),
            (
                "birthday",
                "a",
                "emoji:🙂",
                "真心开心，感到被关爱",
                "过生日收到朋友礼物和祝福",
                "confirmed",
            ),
            ("other", "b", "emoji:🙂", "开心", "导师反复要求修改毕业论文", "confirmed"),
        ],
    )
    r = RagStore(Path(tmp) / "vec", root / "models/embeddings")
    report = []
    for text, expected in [
        ("老师让我把这篇毕业设计再重写一次🙂", "paper"),
        ("今天朋友送我生日礼物了🙂", "birthday"),
    ]:
        result = r.context(c, "a", "memory", text, None, ["emoji:🙂"])
        report.append({"query": text, "hits": result["memories"]})
        assert result["memories"][0]["id"] == expected
        assert all(x["id"] != "other" for x in result["memories"])
    r.close()
    (root / "data/live-vector-ranking.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    print(json.dumps(report, ensure_ascii=False))
