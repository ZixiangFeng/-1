"""Application-level support skills, not Codex plugins or clinical diagnosis."""

import re
from difflib import SequenceMatcher

VERSION = "support-0.3"
SKILLS = {
    "reflect_validate": "具体回应这次经历及感受，承认这种感受可以理解，不替用户定义人格，不空泛说别担心或一定会好。",
    "explore_meaning": "对生活重复、失去期待与意义感先安慰和共情，不马上布置兴趣爱好清单。温和区分失去新鲜感、连接感或掌控感，一次只问一个贴近原话的问题。",
    "listen_without_fixing": "用户只想被听见时，具体承接，不转向解决问题或连续追问。",
    "collaborative_step": "用户明确要建议时，先承接感受，再共同选择一个很小、可撤回的下一步，征求意愿，不说教。",
    "gentle_safety_check": "当个人表达活着无意义且情绪低落，先承接空落感，再温和询问只是迷茫还是也出现伤害自己的想法；不要仅凭意义问题就诊断或断言有自杀倾向。",
    "immediate_support": "明确表达想伤害自己或正在危险中时，先表达关切，直接确认当下安全和是否已采取行动，鼓励联系身边可信的人及当地急救，不要让对方独自面对。避免方法细节，不承诺自动求援，不把AI当作唯一支持。",
    "celebrate": "自然分享喜悦，认可这次具体努力，不把开心解释成痛苦。",
    "casual": "回应当前闲聊，不虚构现实活动，也不复用欢迎语覆盖情绪变化。",
}


def route(text, previous=None):
    previous = previous or {}
    quoted = bool(
        re.search(r"小说|台词|角色说|论文研究|如何识别|翻译|这句话|机器人|模型", text)
    )
    explicit = bool(
        re.search(
            r"我(?:真的|现在|已经|快|很)?(?:不想活|想死|要自杀|想自杀|想伤害自己)|我.*(?:准备|打算).*(?:自杀|结束生命)",
            text,
        )
    )
    negated = bool(
        re.search(
            r"没有.{0,5}(?:自杀|伤害自己|想死)|不(?:会|想)伤害自己|不是想死|不想自杀",
            text,
        )
    )
    existential = bool(
        re.search(
            r"活着.{0,8}(?:意义|意思)|活下去.{0,6}意义|人生.{0,6}(?:意义|没意思)", text
        )
    )
    low = bool(
        re.search(
            r"枯燥|每天.{0,8}重复|没盼头|空虚|麻木|撑不住|没意思|绝望|无望|难过|好累|很累|孤独|委屈|崩溃|压力|焦虑",
            text,
        )
    )
    asking = bool(
        re.search(r"怎么办|怎么做|给.{0,4}建议|帮我.{0,5}想|可以做什么", text)
    )
    only_listen = bool(
        re.search(r"别.{0,4}建议|不想.{0,4}建议|只想.{0,5}(?:听|陪|说)", text)
    )
    risk = (
        "explicit"
        if explicit and not quoted and not negated
        else (
            "clarify" if existential and low and not quoted and not negated else "none"
        )
    )
    topic = (
        "meaning"
        if existential and not quoted
        else ("distress" if low else ("practical" if asking else "casual"))
    )
    if risk == "explicit":
        skills = ["reflect_validate", "immediate_support"]
    elif risk == "clarify":
        skills = ["reflect_validate", "explore_meaning", "gentle_safety_check"]
    elif only_listen:
        skills = ["reflect_validate", "listen_without_fixing"]
    elif topic == "meaning":
        skills = ["reflect_validate", "explore_meaning"]
    elif negated and previous.get("risk") in ("clarify", "explicit"):
        skills = ["reflect_validate", "explore_meaning"]
        topic = "meaning"
    elif asking:
        skills = ["reflect_validate", "collaborative_step"]
    elif low:
        skills = ["reflect_validate", "listen_without_fixing"]
    else:
        skills = ["casual"]
    return {
        "version": VERSION,
        "topic": topic,
        "risk": risk,
        "skills": skills,
        "instructions": [SKILLS[k] for k in skills],
        "topic_changed": bool(previous and previous.get("topic") != topic),
        "previous_topic": previous.get("topic"),
        "source": "heuristic routing, not diagnosis",
    }


def repetition(reply, history):
    def normalize(s):
        return re.sub(r"[\W_]+", "", s)

    r = normalize(reply)
    return bool(
        len(r) > 12
        and any(
            SequenceMatcher(None, r, normalize(h)).ratio() >= 0.82 for h in history[-4:]
        )
    )


def violations(reply, history, support):
    issues = []
    if repetition(reply, history):
        issues.append("重复近期回复，没有推进对话")
    if support["topic"] != "casual" and re.search(
        r"你刚忙完|找个地方放空|你想聊什么|你有什么想聊", reply
    ):
        issues.append("用闲聊欢迎语回避当前困扰")
    if support["risk"] == "explicit" and not re.search(
        r"安全|身边|急救|紧急|可信|伤害自己", reply
    ):
        issues.append("未回应明确的安全风险")
    if re.search(r"你(?:就是|患有|得了)(?:抑郁症|焦虑症)", reply):
        issues.append("未经评估下诊断")
    return issues
