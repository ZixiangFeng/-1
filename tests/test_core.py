import os, tempfile, unittest, json

os.environ.setdefault("DEMO_DATA", tempfile.mkdtemp())
import server


class MemoryIsolation(unittest.TestCase):
    def setUp(self):
        server.init()
        with server.connect() as c:
            for t in ("screenshot_drafts", "sticker_inventory_features", "sticker_inventory_profiles", "messages", "memories", "assets", "sessions", "users"):
                c.execute("DELETE FROM " + t)
            for u in ("a", "b"):
                c.execute(
                    "INSERT INTO users(id,token,label,arm,created,consent) VALUES(?,?,?,?,?,?)",
                    (u, u, u, "memory", server.now(), server.now()),
                )
            c.execute("INSERT INTO sessions VALUES(?,?,?)", ("new", "a", server.now()))
            c.execute("INSERT INTO sessions VALUES(?,?,?)", ("old", "a", server.now()))
            for mid, u, status in [
                ("yes", "a", "confirmed"),
                ("guess", "a", "candidate"),
                ("other", "b", "confirmed"),
            ]:
                c.execute(
                    "INSERT INTO memories VALUES(?,?,?,?,?,?,?,?)",
                    (
                        mid,
                        u,
                        "emoji:🙂",
                        "无奈",
                        "论文",
                        status,
                        "source",
                        server.now(),
                    ),
                )

    def test_confirmed_only_and_user_isolation(self):
        with server.connect() as c:
            self.assertEqual(
                [m["id"] for m in server.retrieve(c, "a", "memory", ["emoji:🙂"])],
                ["yes"],
            )

    def test_control_no_memory(self):
        with server.connect() as c:
            self.assertEqual(server.retrieve(c, "a", "control", ["emoji:🙂"]), [])

    def test_other_expression_not_retrieved(self):
        with server.connect() as c:
            self.assertEqual(server.retrieve(c, "a", "memory", ["emoji:😭"]), [])

    def test_no_cross_session_leakage(self):
        with server.connect() as c:
            c.execute(
                "INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    "m",
                    "a",
                    "old",
                    "user",
                    "SECRET_OLD_MEANING",
                    None,
                    "[]",
                    "",
                    server.now(),
                ),
            )
            prompt, mem = server.make_prompt(
                c, "a", "new", "control", "你好🙂", None, ["emoji:🙂"]
            )
            self.assertNotIn("SECRET_OLD_MEANING", json.dumps(prompt))
            self.assertEqual(mem, [])

    def test_long_prompt_budget_and_equal_history(self):
        with server.connect() as c:
            for i in range(12):
                c.execute(
                    "INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        str(i),
                        "a",
                        "new",
                        "user" if i % 2 == 0 else "assistant",
                        str(i) + "长对话" * 800,
                        None,
                        "[]",
                        "",
                        server.now(),
                    ),
                )
            learned, memory = server.make_prompt(
                c, "a", "new", "memory", "新" * 2000, None, ["emoji:🙂"]
            )
            control, _ = server.make_prompt(
                c, "a", "new", "control", "新" * 2000, None, ["emoji:🙂"]
            )
            self.assertLessEqual(
                server.prompt_bytes(learned), server.PROMPT_BYTE_BUDGET
            )
            self.assertIn(server.PERSONA, learned[0]["content"])
            self.assertIn("无奈", learned[0]["content"])
            self.assertEqual([m["id"] for m in memory], ["yes"])
            self.assertEqual(learned[1:], control[1:])
            self.assertEqual(learned[-1]["content"], "新" * 2000)

    def test_emoji_sequences(self):
        self.assertEqual(
            server.expressions("你好👨‍👩‍👧‍👦👍🏽🙂🙂"),
            ["emoji:👨‍👩‍👧‍👦", "emoji:👍🏽", "emoji:🙂"],
        )

    def test_image_ownership(self):
        with server.connect() as c:
            c.execute(
                "INSERT INTO assets VALUES(?,?,?,?,?)", ("pic", "b", "", "desc", "{}")
            )
            with self.assertRaises(ValueError):
                server.make_prompt(c, "a", "new", "memory", "", "pic", ["image:pic"])


if __name__ == "__main__":
    unittest.main()
