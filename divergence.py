"""Auditable visual contrast gate, not a validated perceptual/CAT scale.

The existing selection model codes visible descriptions and contextual fit in
one call. Code checks grounded excerpts and computes the contrast separately.
Nothing here infers a preference from a participant's un-sent inventory.
"""
from collections import OrderedDict
import re
import unicodedata

POLICY = 'strong-visual-contrast-v2'
VALIDATION_POLICY = 'axis-evidence-v2'
MIN_SCORE = 5
VALUES = {
    'medium': ('photo', 'cartoon', 'line_drawing', 'three_d', 'typography', 'mixed', 'unknown'),
    'subject': ('animal', 'human', 'object', 'abstract', 'unknown'),
    'text_mode': ('none', 'caption', 'text_dominant', 'unknown'),
    'palette': ('color', 'monochrome', 'unknown'),
}
LABELS = {'medium':'画风', 'subject':'主体类别', 'text_mode':'文字载体', 'palette':'色彩'}
PROFILE_SCHEMA = {'type':'object', 'properties':{
    **{k:{'type':'string','enum':list(v)} for k,v in VALUES.items()},
    'evidence':{'type':'object','properties':{k:{'type':'string'} for k in VALUES},
                'required':list(VALUES),'additionalProperties':False}},
    'required':list(VALUES)+['evidence'],'additionalProperties':False}


def diverse_candidates(options, limit=18):
    """Retain semantic rank within each style; avoid a top-k of only animal photos."""
    groups=OrderedDict()
    for option in options:
        groups.setdefault(option.get('style',''),[]).append(option)
    selected=[]
    while groups and len(selected)<limit:
        for key in list(groups):
            selected.append(groups[key].pop(0))
            if not groups[key]:del groups[key]
            if len(selected)==limit:break
    # Keep grounded visible descriptions intact. Shorten only generic semantics.
    return [dict(o,usages=[{k:(v[:180] if isinstance(v,str) else v) for k,v in u.items()}
                           for u in o.get('usages',[])[:1]]) for o in selected]


def _profile_valid(profile, source=None):
    # Shape errors need correction. Evidence quality is assessed per axis below;
    # a bad colour quote must not discard otherwise grounded form differences.
    if not isinstance(profile,dict) or set(profile)!=set(VALUES)|{'evidence'}:return False
    evidence=profile.get('evidence')
    if not isinstance(evidence,dict) or set(evidence)!=set(VALUES):return False
    for axis,values in VALUES.items():
        value=profile[axis];quote=evidence[axis]
        if value not in values or not isinstance(quote,str) or len(quote)>200:return False
    return True


def _normal(text):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text)).translate(
        str.maketrans({'“':'"','”':'"','‘':"'",'’':"'"}))


def grounded(quote, source):
    """Accept literal excerpts or enumerated literal excerpts, never fuzzy paraphrases."""
    quote,source=_normal(quote),_normal(source)
    if len(quote)<2:return False
    if quote in source:return True
    parts=[p.strip('。.') for p in re.split(r'[、,;；\n]|\.{3}|…+', quote) if p.strip('。.')]
    return len(parts)>1 and all(len(p)>=2 and p in source for p in parts)


def evaluate(selected, reference, candidate):
    result={'policy':POLICY,'validation_policy':VALIDATION_POLICY,'threshold':MIN_SCORE,'score':0,'passed':False,
            'dimensions':[], 'reference_asset_id':reference['asset_id'],
            'candidate_asset_id':candidate['asset_id'],
            'semantic_fit':selected.get('semantic_fit'),
            'reference_profile':selected.get('reference_profile'),
            'candidate_profile':selected.get('candidate_profile')}
    a=selected.get('reference_profile');b=selected.get('candidate_profile')
    if not _profile_valid(a) or not _profile_valid(b):
        return dict(result,reason='视觉特征输出格式不完整',validation_error='profile_schema')
    sources=(reference.get('visible_description',''),candidate.get('description','')+'\n'+candidate.get('style',''))
    ignored=[]
    for axis in VALUES:
        for label,p,source in zip(('reference','candidate'),(a,b),sources):
            if p[axis]!='unknown' and not grounded(p['evidence'][axis],source):
                ignored.append({'axis':axis,'side':label,'reason':'描述中未找到所引证据'})
    result['ignored_evidence']=ignored
    # A broad legacy catalog label is not evidence of absence/presence of text.
    # Keep the model's raw coding in the audit, but never score this ambiguity.
    ambiguous_text=any('或无文字' in p['evidence']['text_mode'] or '有无文字' in p['evidence']['text_mode']
                       for p in (a,b))
    dimensions=[]
    for axis in VALUES:
        if any(x['axis']==axis for x in ignored):continue
        if axis=='text_mode' and ambiguous_text:continue
        x,y=a[axis],b[axis]
        if x==y or 'unknown' in (x,y):continue
        if axis=='medium':
            if 'mixed' in (x,y):continue  # Mixed media is not a clear opposite.
            # Photo -> monochrome line art is a strong form change even when
            # both depict humans. It need not also switch humans to animals.
            rendered={'photo','three_d'};graphic={'line_drawing','typography'}
            weight=4 if (x in rendered and y in graphic) or (y in rendered and x in graphic) else (
                2 if {x,y}=={'cartoon','line_drawing'} else 3)
        elif axis=='subject':weight=2
        elif axis=='text_mode':weight=2 if 'text_dominant' in (x,y) else 1
        else:weight=1
        dimensions.append({'axis':axis,'label':LABELS[axis],'from':x,'to':y,'weight':weight,
                           'reference_evidence':a['evidence'][axis],'candidate_evidence':b['evidence'][axis]})
    score=sum(d['weight'] for d in dimensions)
    major=any(d['axis']=='medium' or (d['axis']=='text_mode' and d['weight']==2) for d in dimensions)
    passed=selected.get('semantic_fit') is True and score>=MIN_SCORE and len(dimensions)>=2 and major
    why='语义适切且通过强差异校验' if passed else (
        '本轮语义适切性不足' if selected.get('semantic_fit') is not True else '视觉变化不足：仅换角色、动物或增加短配字不算强趋异')
    return dict(result,dimensions=dimensions,score=score,passed=passed,reason=why,
                validation_error='ungrounded_evidence' if ignored and not passed else None)


PROMPT = '''
强差异版本：不要仅换猫/鸭、角色身份、动作或几字短配文就当作趋异。
必须保持本轮沟通意图、情绪方向与文字支持性；不能靠嘲讽、相反情绪、冷淡或错误语义拉大差异。
在语义适切的候选中优先选择画风和表达载体有明确对比的图片。例如彩色卡通动物→黑白简笔人物；写实动物照片→简笔人物插画。它们只是形式例子，具体含义仍需符合当前对话。
如果本轮有用户图片，必须用current_asset为参照，不能挑旧图凑差异；纯文字轮次才选语境相关的历史真实用图。
对所选的一对图片输出reference_profile和candidate_profile，按可见描述编码medium、subject、text_mode、palette及evidence。evidence各字段必须逐字摘录对应图片description/style中的短片段（200字以内），不得把语义推测当作可见证据。用户参照只引用visible_description。无法判定填unknown、证据空字符串。
同一维度可以用顿号连接多个原文短片段，每个片段都必须确实来自描述。不要为了文字润色增加原文没有的词。后端按维度核验证据，未核实的维度不计分，其他有效维度仍独立计分。
medium: photo=写实摄影，cartoon=卡通插画，line_drawing=简笔线稿，three_d=三维玩偶，typography=文字构成画面主体，mixed=无法分清主导画风的混合。
subject按主导主体分animal/human/object/abstract，猫和鸭均为animal；穿人衣服的猫仍为animal，不能为了得分改成人物。简笔小人归human。text_mode: none=明确无文字，caption=短配字，text_dominant=文字主导画面。少量或无文字无法区分则unknown。palette只有明确彩色/黑白才编码color/monochrome，未知勿猜。
后端独立计算：写实摄影/三维↔简笔线稿/文字主导画风4分，其他跨画风3分（卡通↔简笔2分）；跨主体类别2分；无字↔短配字1分，文字主导↔其他2分；彩色↔黑白1分。未知不计，mixed画风不计。至少5分、至少两项变化，且包含画风或文字主导载体变化，才可发图。彩色人物照片→黑白简笔人物可由画风4分+色彩1分通过，无需强制改变主体类别。
所以卡通动物→彩色写实动物加短配字通常只有4分，不合格；写实猫→写实鸭不合格。优先挑已有足够证据的强差异候选，不要虚构标签/引文过门槛。
semantic_fit为布尔值，只有语义和双方角色都适合才为true。reason说明本轮意图、参照和形式对比。不合适或无法通过门槛则asset_id/reference_asset_id返回空字符串、semantic_fit=false、profile各字段unknown、evidence各字段空字符串。不要强行发图。
'''
