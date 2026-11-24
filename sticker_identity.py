"""User-relative image exclusion, never semantic learning from an inventory.

File hashes catch identical bytes. Canonical pixels and a conservative thumbnail
comparison also catch ordinary resizing/recompression. This is not a classifier
of meaning or style, nor a guarantee for arbitrarily edited/cropped images.
"""
import base64
import hashlib
import io
import json
import math
import sqlite3
from collections import OrderedDict
from threading import RLock

from PIL import Image, ImageOps

POLICY = 'exclude-personal-images-v1'
_cache = OrderedDict()
_lock = RLock()


def fingerprint(encoded):
    # Cache compact features by content digest, not large base64 strings; a changed
    # file is automatically a different key. No participant ownership is cached.
    encoded = encoded or ''
    try:
        raw = base64.b64decode(encoded.split(',')[-1], validate=True)
    except (ValueError, TypeError):
        raw = encoded.encode()
    key = hashlib.sha256(raw).hexdigest()
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    result = {'file': key}
    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.width * source.height > 16_000_000:
                raise ValueError('Image too large')
            image = ImageOps.exif_transpose(source).convert('RGBA')
            background = Image.new('RGBA', image.size, 'white')
            image = Image.alpha_composite(background, image).convert('RGB')
            result.update(aspect=image.width/image.height,
                          pixels=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest(),
                          coarse=image.resize((8,8), Image.Resampling.LANCZOS).tobytes(),
                          thumbnail=image.resize((64,64), Image.Resampling.LANCZOS).tobytes())
    except (OSError, ValueError, Image.DecompressionBombError):
        # Legacy invalid files can still be matched exactly; all current upload
        # routes validate real static image bytes before creating an asset.
        pass
    with _lock:
        _cache[key] = result
        if len(_cache) > 1024:
            _cache.popitem(last=False)
    return result


def same_image(a, b):
    if a['file'] == b['file']:
        return 'same_file'
    if 'thumbnail' not in a or 'thumbnail' not in b:
        return None
    if a['pixels'] == b['pixels']:
        return 'same_pixels'
    if abs(a['aspect']/b['aspect']-1) > .03:
        return None
    if sum(abs(x-y) for x,y in zip(a['coarse'],b['coarse']))/192 > 8:
        return None
    # Both colour and local structure must agree. A perceptual hash alone would
    # incorrectly merge different solid colours or visually different stickers.
    diffs = [abs(x-y) for x,y in zip(a['thumbnail'],b['thumbnail'])]
    if sum(diffs)/len(diffs) <= 6 and sum(d > 40 for d in diffs)/len(diffs) <= .02:
        return 'near_duplicate'
    return None


def exclusions(c, user, only_ids=None):
    """Return blocked public IDs/reasons for ONE user's entire private inventory.

    Inventory membership may veto a picture, but must never serve as evidence of
    personal meaning, preference or divergence. No records are written here.
    """
    owned = c.execute('SELECT id,image,metadata FROM assets WHERE user=?',(user,)).fetchall()
    sources = {r['id']:'personal_asset_id' for r in owned}
    for row in owned:
        try:
            meta = json.loads(row['metadata'] or '{}')
            if isinstance(meta,dict) and isinstance(meta.get('source_asset'),str):
                sources[meta['source_asset']] = 'saved_source'
        except (ValueError,TypeError):
            pass
    # Defensive support for historical direct use of a public asset ID.
    sources.update({r['asset']:'user_sent_source' for r in c.execute(
        "SELECT DISTINCT asset FROM messages WHERE user=? AND role='user' AND asset IS NOT NULL",(user,))})
    personal = [fingerprint(r['image']) for r in owned if r['image']]
    rows = c.execute('SELECT id,image FROM generic_stickers WHERE enabled=1').fetchall()
    blocked = {}
    for row in rows:
        aid = row['id']
        if only_ids is not None and aid not in only_ids:
            continue
        reason = sources.get(aid)
        if not reason:
            candidate = fingerprint(row['image'])
            reason = next((match for own in personal if (match := same_image(candidate,own))),None)
        if reason:
            blocked[aid] = reason
    return blocked


def guard_reply(c, user, arm, asset, audit):
    """Last server-side check, even if a selector/cache returns an invalid ID."""
    if not asset or arm not in ('G','D'):
        return asset,audit
    audit = dict(audit, personal_exclusion_policy=POLICY)
    try:
        public = c.execute('''SELECT g.enabled,COALESCE(m.pool,'generic') AS pool
            FROM generic_stickers g LEFT JOIN sticker_library_metadata m ON m.asset=g.id
            WHERE g.id=?''',(asset,)).fetchone()
        reason = ('not_enabled_public_material' if not public or not public['enabled'] else
                  'outside_G_pool' if arm=='G' and public['pool']!='generic' else
                  exclusions(c,user,{asset}).get(asset))
    except Exception:
        reason = 'exclusion_check_failed'
    if reason:
        audit.update(asset_id=None, blocked_asset_id=asset,
                     exclusion_reason=reason,reason='图片未通过个人图库排除校验，保留文字回复')
        return None,audit
    audit['personal_exclusion_verified'] = True
    if arm == 'D':
        try:
            verified = _confirmed_maximum(c, asset, audit)
        except (ValueError, TypeError, KeyError, sqlite3.Error):
            verified = False
        if not verified:
            audit.update(asset_id=None, blocked_asset_id=asset, exclusion_reason='unverified_group_maximum',
                         reason='本轮用意或整组最大风格距离未核实，保留文字回复')
            return None, audit
        audit['group_maximum_verified'] = True
    return asset,audit


def _confirmed_maximum(c, asset, audit):
    import sticker_intent
    if audit.get('selection_rule') != sticker_intent.POLICY or audit.get('maximum_asset_id') != asset:
        return False
    import sticker_pragmatics as use
    review = c.execute('SELECT groups_json,token FROM sticker_intent_reviews WHERE asset=?', (asset,)).fetchone()
    if not review:
        return False
    groups = json.loads(review[0])
    if audit.get('semantic_policy') == use.VERSION:
        group = audit.get('semantic_group')
        if not group or group not in {use.family(g) for g in groups}:
            return False
        candidate = next((o for o in audit.get('candidates', []) if o.get('asset_id')==asset), {})
        if candidate.get('review_token') != review[1]:
            return False
    elif audit.get('semantic_group') not in groups:
        return False
    ranking = audit.get('style_ranking') or []
    candidates = {item['asset_id'] for item in audit.get('candidates', [])}
    if (not ranking or len(ranking) != len(candidates) or len(ranking) != audit.get('semantic_group_size')
            or {item['id'] for item in ranking} != candidates):
        return False
    if any(not math.isfinite(item['distance']) for item in ranking):
        return False
    winner = min(ranking, key=lambda item: (-item['distance'], item['id']))
    return (winner['id'] == asset and winner['distance'] == audit.get('maximum_style_distance')
            and winner['distance'] >= sticker_intent.MIN_STYLE)
