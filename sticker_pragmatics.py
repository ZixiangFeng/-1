"""Ground a sticker's conversational use in language before comparing visual styles."""
import hashlib
import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

VERSION = 'xiaoyu-pragmatic-eight-v7'
DIRECTORY = Path(__file__).resolve().parent / 'research/sticker_encoder/v7'
OUTSIDE = 'outside'

# These eight labels are trained directly and are the only retrieval partitions.
FAMILIES = {
    'joy': ('开心与玩笑', ('laugh', 'proud', 'party')),
    'response': ('回应与认同', ('ack', 'agree')),
    'question': ('疑问与惊讶', ('confused', 'doubt', 'surprise')),
    'distress': ('低落与感叹', ('cry', 'speechless', 'tired', 'fear', 'grin')),
    'support': ('亲近与支持', ('comfort', 'cheer', 'seek_hug', 'love', 'praise')),
    'objection': ('不满与拒绝', ('refuse', 'attack', 'mock')),
    'social': ('问候与礼貌', ('thanks', 'sorry', 'leave', 'sleep')),
    'daily': ('日常需求与邀请', ('hungry', 'invite', 'money')),
}
NAMES = {key: value[0] for key, value in FAMILIES.items()}
LEGACY = {old: key for key, (_, members) in FAMILIES.items() for old in members}


def family(key):
    return key if key in NAMES else LEGACY.get(key, '')


# These expressions operate on utterances or usage glosses. They never see artist,
# image ID, medium, character, visual distance, or the target split.
CUES = {
    'laugh':r'开心|开怀|笑死|爆笑|好笑|大笑|放声.{0,2}笑|被逗乐|逗.{0,3}(?:笑|乐)|忍俊不禁|哈哈哈|喜悦|欢乐|欢呼|庆祝|乐开花|喜出望外|兴高采烈',
    'grin':r'尬笑|尴尬.{0,5}笑|笑.{0,5}尴尬|礼貌.{0,4}笑|陪笑|赔笑|强颜欢笑|强笑|假笑|苦笑|勉强.{0,3}笑|掩饰窘',
    'mock':r'幸灾乐祸|嘲笑|奚落|看.{0,3}笑话|取笑|嘲弄|冷嘲热讽',
    'ack':r'收到(?:了)?(?:通知|消息)|(^收到[了啦呀！!。]?$)|表示.{0,5}(?:收到|明白)|听懂|听明白|恍然大悟|弄明白|我明白|我懂了|我知道了|知晓|记下了',
    'agree':r'赞同|赞成|认同|同意|有道理|说得对|我支持.{0,3}(?:想法|观点|说法)',
    'confused':r'困惑|费解|迷惑|疑惑|不理解|搞不懂|没(?:有)?(?:听|看|搞|弄|整)懂|什么意思|满头问号|懵了|糊涂|搞不明白',
    'surprise':r'惊讶|惊呆|震惊|大吃一惊|错愕|猝不及防|意想不到|没想到|竟然|出乎意料|被吓一跳',
    'comfort':r'安慰(?:你|他|对方|别人)|安抚|我陪(?:着)?你|给你.{0,5}(?:拥抱|抱抱)|抱抱(?:你|他|对方)|心疼你|陪伴对方|我在听',
    'seek_hug':r'求(?:抱抱|安慰|拥抱|陪伴)|(?:渴望|请求|想要|寻求).{0,5}(?:拥抱|安慰|陪伴)|抱抱我|陪陪我|想被.{0,4}抱',
    'praise':r'称赞|称许|夸奖(?:别人|他|你|对方)|夸赞|赞许|赞赏|佩服|点赞|你真厉害|你真棒|做得(?:漂亮|真好)',
    'proud':r'得意|自豪|自夸|骄傲|炫耀|夸自己|我.{0,2}真厉害',
    'cry':r'伤心|心酸|委屈|泪流|哭泣|难过|想哭|掉眼泪|哭诉',
    'tired':r'疲惫|疲倦|疲劳|好累|累[死趴坏瘫]|累到|累得|精疲力尽|筋疲力尽|耗尽|提不起劲',
    'speechless':r'无语|无奈|无话可说|无言以对|哑口无言|无言|语塞|不知道.{0,4}(?:接话|说什么)|不知如何(?:接话|回应)|一言难尽',
    'refuse':r'拒绝|不答应|不同意|不接受|明确反对|婉拒|不奉陪|觉得不行',
    'attack':r'发火|气死|很生气|暴怒|气愤|愤怒|威胁|警告对方|怒斥|想揍|扬言',
    'fear':r'害怕|恐惧|发抖|发毛|瑟瑟发抖|不敢.{0,3}(?:靠近|进去)|心慌|胆怯',
    'love':r'喜爱|喜欢你|爱意|示爱|心动|比心|爱你|爱慕|表白|亲吻|飞吻',
    'thanks':r'感谢|感激|谢谢|多谢|谢啦|道谢|致谢|幸亏你|多亏你',
    'sorry':r'道歉|对不起|抱歉|认错|赔不是|我的错|对不住',
    'cheer':r'加油|打气|鼓励|鼓劲|你能行|挺住|再坚持一下',
    'sleep':r'晚安|去睡|睡觉了|先睡|好梦|该睡了|关灯睡',
    'invite':r'邀请|邀约|欢迎.{0,4}(?:来|加入)|一起来|要不要.{0,4}一起|约你|来玩吧',
    'leave':r'告别|道别|再见|回头见|拜拜|先走了|告辞|先撤|下次再聊',
    'doubt':r'怀疑|质疑|不相信|不太信|难以相信.{0,4}说法|靠谱吗|真的(?:吗|么)|真的假的',
    'money':r'讨.{0,2}红包|(?:发|要|求|索要|讨要|给我).{0,3}红包|给我.{0,2}(?:打钱|转账)|索要.{0,4}(?:转账|钱)',
    'hungry':r'我饿|肚子饿|饿了|饥饿|馋得|嘴馋|好想吃|我想吃|干饭|想吃(?:饭|东西|火锅)',
}
CUES.update({
    'solitude': r'寂寞|孤单|孤独|落寞|没人(?:陪|理|和我说话|跟我聊)|无人(?:理解|陪)|不被理解的空虚|一个人.{0,4}(?:好冷清|真冷清|有点空虚)',
    'bored': r'无聊|闲得慌|百无聊赖|无所事事|打发时间',
    'insult': r'(?:指|说|骂|吐槽|嫌弃|嘲讽).{0,7}(?:变态|不正常|有病|脑子有问题)|你.{0,3}(?:变态|有病|不正常)|辱骂|人身攻击',
    'hello': r'早上好|上午好|下午好|晚上好|早安|你好呀|打招呼|问好|寒暄',
    'offer_food': r'给你.{0,3}(?:尝|吃)|给对方.{0,4}(?:吃|食物)|分[享给].{0,4}(?:食物|吃)|把吃的递|给你留.{0,4}(?:点心|披萨)|请你.{0,2}吃',
    'no_laugh': r'笑不出|不好笑|不觉得.{0,2}好笑|没(?:什么|有)?笑点',
})
SIGNAL_FAMILY = dict(LEGACY, solitude='distress', bored='distress', insult='objection',
                     hello='social', offer_food='daily', no_laugh='objection')
CUES = {k:re.compile(v) for k,v in CUES.items()}
VETO = {
    'laugh':r'笑不出|不觉得.{0,2}好笑|不好笑|不想笑|不开心|没(?:什么|有)?笑点|假(?:装)?笑|干笑|尬笑|强颜|嘲讽|嘲笑|苦笑|讽刺|阴阳怪气|笑话他|陪笑|笑里藏刀|冷笑|哭笑不得',
    'ack':r'没(?:有)?.{0,2}(?:懂|明白|收到)|没看懂|不明白|收到.{0,4}[吗么？?]|等.{0,3}(?:消息|通知)|收件',
    'agree':r'不同意|不认同|反对|不赞成|勉强.{0,3}(?:同意|答应)|不太.{0,2}(?:认同|同意)',
    'thanks':r'不用谢|不必.{0,2}谢|别谢|不感谢|无需.{0,2}谢',
    'sorry':r'不用.{0,2}道歉|别道歉|拒绝道歉|不用说.{0,2}对不起|不觉得.{0,3}有错',
    'hungry':r'你.{0,4}饿|饿不饿|给你.{0,3}吃|请.{0,4}吃|分[享给].{0,4}(?:食物|吃)|(?:食物|吃的).{0,5}(?:递|送|分享)|递给对方|不饿',
    'praise':r'自夸|夸(?:奖)?自己|不是.{0,3}夸你|假意夸|讽刺|阴阳怪气|反讽',
    'comfort':r'求安慰|寻求安慰|安慰我|谁来陪我|抱抱我|陪陪我|想被.{0,4}抱',
    'seek_hug':r'我.{0,3}(?:安慰|抱抱|陪着|陪伴|心疼)你|我给你.{0,5}抱|拒绝.{0,3}安慰',
    'attack':r'邀战|放马过来|挑战自己|没有生气|没生气|不要生气|别生气',
    'invite':r'邀战|来.{0,4}打架|约架|放马过来',
    'fear':r'不要害怕|别害怕|我不怕|我不害怕',
    'money':r'不给钱|发给你.{0,3}红包|给对方.{0,4}(?:钱|红包)',
    'love':r'不喜欢|不爱你|挥手.{0,5}(?:招呼|问好)|打招呼',
    'refuse':r'没有拒绝|不想拒绝|并未拒绝',
}
VETO = {k:re.compile(v) for k,v in VETO.items()}
MARKER = re.compile(r'常用来|主要用来|用来表达|用来表示|常用于|适合(?:用于|用来)?|(?:[，,；;。]|^)\s*(?:表达(?:出|了)?|表示|传达|意在)')
JOY_CELEBRATION = re.compile('庆祝|恭喜|祝贺|欢呼|获奖|夺冠|胜利')
JOY_AMUSEMENT = re.compile('好笑|逗乐|被逗|笑话|忍俊不禁|爆笑|搞笑|笑点|段子')


def wording(raw):
    s = unicodedata.normalize('NFKC', raw or '').strip().lower()
    return re.sub(r'\s+', '', s)[:600]


def meaning_part(text):
    marks = list(MARKER.finditer(text))
    if marks:
        tail = text[marks[-1].end():].strip('，,。；; ')
        if len(tail)>=2:
            return tail
    return text


def char_grams(text):
    parts = re.findall(r'[\u3400-\u9fff]+|[a-z0-9]+', text)
    return {word[i:i+n] for word in parts for n in (1,2,3) for i in range(len(word)-n+1)}


def make_vocab(texts):
    freq = Counter(g for t in texts for g in char_grams(t))
    keys = sorted(g for g,n in freq.items() if n>=2)
    return keys, np.array([math.log((len(texts)+1)/(freq[g]+1))+1 for g in keys],dtype=np.float32)


def lexical_features(texts, index, idf):
    out = np.zeros((len(texts),len(index)),dtype=np.float32)
    for i,t in enumerate(texts):
        cols = [index[g] for g in char_grams(t) if g in index]
        out[i,cols] = idf[cols]
    return out / np.maximum(np.linalg.norm(out,axis=1,keepdims=True),1e-8)


def language_kernel(z, words, anchor_z, anchor_words):
    return .72*(z @ anchor_z.T)**3 + .28*(words @ anchor_words.T)


@lru_cache(maxsize=12000)
def _public_signals(text):
    # Only use this cache for the public gallery or local evaluation.
    return signals(text)


def signals(text):
    full = wording(text)
    main = meaning_part(full)
    return frozenset(k for k, pattern in CUES.items() if pattern.search(main)
                     and not (k in VETO and VETO[k].search(full)))


def classifier_decisions(raw_texts, scores, labels, minimum, margin):
    out = []
    for text, values in zip(raw_texts, scores):
        tags = signals(text)
        direct = {SIGNAL_FAMILY[tag] for tag in tags}
        adjusted = values.copy()
        for j, label in enumerate(labels):
            if label in direct:
                adjusted[j] += .28
        # A first-person loneliness statement and a directed insult have explicit
        # pragmatic force, including when the gloss calls them self-mockery or a joke.
        unambiguous = {SIGNAL_FAMILY[tag] for tag in tags & {'solitude', 'insult', 'no_laugh'}}
        if len(unambiguous) == 1:
            out.append(unambiguous.pop())
            continue
        first, second = np.argsort(-adjusted, kind='stable')[:2]
        best = labels[first]
        if best == OUTSIDE or adjusted[first] < minimum or adjusted[first] - adjusted[second] < margin:
            best = ''
        out.append(best)
    return out


# These are contrasts in speaker, stance or speech act, not extra retrieval bins.
# A broader family can accept paraphrases without a matching old subclass.
CONTRASTS = (
    ({'comfort', 'cheer'}, {'seek_hug'}), ({'hungry'}, {'offer_food'}),
    ({'laugh'}, {'grin', 'mock', 'no_laugh', 'insult'}), ({'praise'}, {'proud', 'mock', 'insult'}),
    ({'ack', 'agree'}, {'confused', 'doubt', 'refuse'}), ({'sorry'}, {'thanks', 'hello', 'leave'}),
    ({'insult', 'mock'}, {'solitude', 'bored', 'cry', 'speechless'}),
    ({'love'}, {'praise', 'cheer'}), ({'seek_hug'}, {'praise'}),
    ({'hello'}, {'leave', 'sleep', 'thanks'}), ({'money'}, {'hungry', 'offer_food'}),
    ({'fear'}, {'tired', 'bored'}), ({'surprise'}, {'confused', 'doubt'}),
)
EQUIVALENT = ({'ack', 'agree'}, {'comfort', 'cheer'}, {'sleep', 'leave'})
# With these explicit meanings, a merely related statement is particularly risky.
SPECIFIC = {'solitude', 'insult', 'no_laugh', 'offer_food'}


def compatible_signals(left_tags, right_tags):
    if (left_tags ^ right_tags) & SPECIFIC:
        return False
    shared = left_tags & right_tags
    for one, other in CONTRASTS:
        if (left_tags & one and right_tags & other) or (right_tags & one and left_tags & other):
            if not (shared & (one | other)):
                return False
    if left_tags and right_tags and not shared:
        return any(left_tags & soft and right_tags & soft for soft in EQUIVALENT)
    return True


def compatible_use(group, left, right, left_tags=None, right_tags=None):
    if left_tags is None:
        left_tags = signals(left)
    if right_tags is None:
        right_tags = signals(right)
    if not compatible_signals(left_tags, right_tags):
        return False
    if group == 'joy':
        def purpose(text):
            return (bool(JOY_CELEBRATION.search(text)), bool(JOY_AMUSEMENT.search(text)))
        if (purpose(left), purpose(right)) in (((True,False),(False,True)), ((False,True),(True,False))):
            return False
    return True


def gallery_result(query, rows, labels, group, query_text, scores, distances, identity, floor):
    import sticker_intent
    query_tags = signals(query_text)
    allowed = []
    broad_count = 0
    for row, label, score in zip(rows, labels, scores):
        accepted = bool(group and label == group and row['quality'] != 'uncertain')
        if accepted:
            if not identity(query, row):
                broad_count += 1
            accepted = bool(np.isfinite(score) and score >= floor and compatible_use(
                group, query_text, row['meaning'], query_tags, _public_signals(row['meaning'])))
        allowed.append(group if accepted else '')
    result = sticker_intent.gallery_result(query, rows, allowed, group, scores, distances, identity)
    result.update(group_name=NAMES.get(group, ''), broad_count=broad_count, pair_floor=floor,
                  policy=VERSION + '/' + sticker_intent.POLICY)
    return result


class IntentModel:
    def __init__(self, path=DIRECTORY/'language-intent.npz'):
        import sticker_ranker
        with np.load(path,allow_pickle=False) as a:
            if str(a['version'])!=VERSION or str(a['base'])!=sticker_ranker.metric().digest:
                raise ValueError('Unexpected language classifier or embedding weights')
            self.labels = list(a['labels'])
            self.anchor_z = a['anchor_z']
            self.index = {g:i for i,g in enumerate(a['vocab'])}
            self.idf = a['idf']
            self.anchor_words = lexical_features(list(a['texts']),self.index,self.idf)
            self.alpha = a['alpha']
            self.minimum,self.margin,self.pair_floor = (float(a[k]) for k in ('minimum','margin','pair_floor'))
        self.digest = hashlib.sha256(path.read_bytes()).hexdigest()

    def groups(self, projected, texts):
        if len(projected)!=len(texts):
            raise ValueError('Missing query or embedding')
        words = lexical_features([meaning_part(wording(t)) for t in texts],self.index,self.idf)
        scores = language_kernel(projected,words,self.anchor_z,self.anchor_words) @ self.alpha
        return classifier_decisions(texts,scores,self.labels,self.minimum,self.margin)


@lru_cache(maxsize=1)
def model():
    return IntentModel()
