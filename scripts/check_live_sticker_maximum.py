"""Run the normal chat chooser against an isolated synthetic participant and real PNGs."""
import base64
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import generic_stickers
import stickers
OUT = ROOT / 'outputs/sticker-external-holdout-20261119'

def image(id):
    return base64.b64encode((OUT/'images'/(id+'.png')).read_bytes()).decode()

def main():
    with tempfile.TemporaryDirectory() as data:
        os.environ['DEMO_DATA']=data
        with sqlite3.connect(':memory:') as c:
            c.row_factory=sqlite3.Row
            c.executescript("""CREATE TABLE assets(id,user,image,description,metadata);
                CREATE TABLE messages(id,user,session,role,text,asset,interpretation,created);
                CREATE TABLE memories(id,user,expression,meaning,context,status,source,created);
                CREATE TABLE events(id INTEGER PRIMARY KEY,user,kind,payload);""")
            stickers.init(c);generic_stickers.init(c)
            q='真实摄影小孩，无字，彩色'
            c.execute('INSERT INTO assets VALUES(?,?,?,?,?)',('ref','QA',image('X038'),q,'{}'))
            c.execute('INSERT INTO messages VALUES(?,?,?,?,?,?,?,?)',('source','QA','s','user','真无语了','ref','无语，无奈得不知道说什么','now'))
            for key,file,desc,g in [('a','X041','彩色手绘角色，无字','speechless'),
                                     ('b','X076','三维黄色表情，无字','speechless'),
                                     ('joy','X068','三维笑脸','laugh')]:
                c.execute('INSERT INTO generic_stickers VALUES(?,?,?,?,?)',(key,image(file),json.dumps(dict(
                    description=desc,meaning='无语' if g=='speechless' else '开心',situations='测试用例',avoid='冒犯')),1,'now'))
                c.execute('INSERT INTO sticker_library_metadata VALUES(?,?,?)',(key,'generic',desc))
                version={r['id']:r['review_version'] for r in generic_stickers.reviewed_catalog(c)}[key]
                generic_stickers.save_intents(c,key,[g],version)
            records=[]
            def checker(messages,**kwargs):
                # The actual generator may confirm or abstain, never choose a lower entry.
                # This smoke test exercises its explicit abstain branch without a paid API.
                packet=json.loads(messages[1]['content'].split('\n')[0])
                records.append([o['asset_id'] for o in packet['candidates']])
                return json.dumps(dict(asset_id='',reason='离线检验，核对器弃权')),{}
            runs=[]
            for _ in range(2):
                aid,audit=stickers.choose(c,'QA','s','D','真无语了','听起来确实挺无奈。','ref',checker,rng=lambda n:0)
                assert aid is None and {r['id'] for r in audit['style_ranking']}=={'a','b'},audit['reason']
                ranks=audit['style_ranking'];top=min(ranks,key=lambda r:(-r['distance'],r['id']))
                assert audit['maximum_asset_id']==top['id'] and audit['maximum_style_distance']==top['distance']
                assert audit['maximum_style_distance']>=.65 and records[-1]==[top['id']]
                runs.append(dict(winner=top['id'],ranking=ranks,semantic_group=audit['semantic_group']))
            assert runs[0]==runs[1]
            # A second isolated case covers delivery. Here the reference is a
            # stylized yellow face, and both candidates were coded as speechless.
            face = '彩色三维黄色抽象表情，斜眼流汗，无文字'
            child = '彩色真人摄影，侧眼看人的小孩，无文字'
            c.execute("UPDATE assets SET image=?,description=? WHERE id='ref'", (image('X076'),face))
            c.execute("UPDATE generic_stickers SET image=?,semantics=? WHERE id='b'", (image('X038'),
                      json.dumps(dict(description=child,meaning='无语',situations='侧眼回应',avoid='冒犯'))))
            c.execute("UPDATE sticker_library_metadata SET style=? WHERE asset='b'", (child,))
            b = next(r for r in generic_stickers.reviewed_catalog(c) if r['id']=='b')
            generic_stickers.save_intents(c,'b',['speechless'],b['review_version'])
            def visible(medium,subject,quote):
                values = dict(medium=medium,subject=subject,palette='color',text_mode='none')
                return dict(values,evidence=dict.fromkeys(values,quote))
            def confirm(messages,**kwargs):
                packet = json.loads(messages[1]['content'].split('\n')[0])
                assert [v['asset_id'] for v in packet['candidates']] == ['b']
                return json.dumps(dict(asset_id='b',reference_asset_id='ref',semantic_fit=True,
                    reason='无语地斜眼回应，抽象表情与人物摄影的形式有明显区别',template_relation='different',
                    reference_profile=visible('three_d','abstract',face),candidate_profile=visible('photo','human',child))),{}
            actual, logged = stickers.choose(c,'QA','s','D','真无语了','听起来确实挺无奈。','ref',confirm,rng=lambda _:0)
            assert actual=='b' and logged.get('group_maximum_verified') is True,logged['reason']
            delivered = dict(reference='X076',winner='X038',distance=logged['maximum_style_distance'],
                             semantic_group=logged['semantic_group'],group_size=logged['semantic_group_size'])
    (OUT/'chat-path-check.json').write_text(json.dumps(dict(policy=stickers.sticker_intent.POLICY,real_ALADIN=True,
             semantic_model=stickers.sticker_ranker.metric().digest,final_selector='fixed offline abstention and confirmation',
             runs=runs,delivered=delivered),
             ensure_ascii=False,indent=2))
    print(runs[0])
if __name__=='__main__':main()
