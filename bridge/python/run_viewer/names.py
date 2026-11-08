"""名称表与道具图标（可选）：从 analysis/resources 里的 Repentance 配置（repentance-a/config）与 AB+ 解包
（gibbed/config、gibbed/graphics）读实体名/碰撞半径、道具/饰品/卡/药名、道具图标。找不到就返回空表，查看器照常工作。

名称规则：Repentance 的 xml 里名称是本地化键（"#THE_SAD_ONION_NAME"），这里把键还原成可读名（"The Sad Onion"）；
若 AB+ 解包里同一 id 的英文名去掉标点后与键一致，就用 AB+ 的英文原名（保留撇号等）。
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

_ATTR = re.compile(r'([A-Za-z_][\w\-]*)="([^"]*)"')


def default_resources_dir():
    # rl/bridge/python/run_viewer/names.py -> parents[4] = 仓库外层工作目录（D:\Projects\fortune\Isaac）
    return Path(__file__).resolve().parents[4] / 'analysis' / 'resources'


def _tags(path, tag):
    try:
        text = Path(path).read_text(encoding='utf-8', errors='replace')
    except OSError:
        return []
    out = []
    for m in re.finditer(r'<%s\s([^>]*?)/?>' % tag, text):
        out.append(dict(_ATTR.findall(m.group(1))))
    return out


def _norm(s):
    return re.sub(r'[^a-z0-9]', '', (s or '').lower())


def pretty_key(name):
    """'#THE_SAD_ONION_NAME' -> 'The Sad Onion'；不是键就原样返回。"""
    if not name or not name.startswith('#'):
        return name
    s = name[1:]
    if s.endswith('_NAME'):
        s = s[:-5]
    words = [w for w in s.split('_') if w]
    small = {'of', 'the', 'a', 'and', 'in', 'on', 'to'}
    out = []
    for i, w in enumerate(words):
        lw = w.lower()
        out.append(lw if (i and lw in small) else lw.capitalize())
    return ' '.join(out)


def _best(rep_name, abp_name):
    p = pretty_key(rep_name)
    if abp_name and _norm(abp_name) == _norm(p):
        return abp_name
    return p or abp_name


def load_names(res_dir=None, want_icons=None):
    """返回 dict(entities, items, trinkets, cards, pills, source)；want_icons = {'c': set(道具 id), 't': set(饰品 id)}
    时另返回 icons {'c<id>': data URI, 't<id>': data URI}（只嵌入本局出现过的）。"""
    res = Path(res_dir) if res_dir else default_resources_dir()
    rep = res / 'repentance-a' / 'config'
    abp = res / 'gibbed' / 'config' / 'resources'
    gfx = res / 'gibbed' / 'graphics' / 'resources' / 'gfx' / 'items'
    out = dict(entities={}, items={}, trinkets={}, cards={}, pills={}, icons={}, source=str(res))
    if not rep.is_dir():
        out['source'] = None
        return out
    # 实体：键 "type.variant.subtype" / "type.variant" / "type" -> [名称, 碰撞半径, boss]
    abp_ent = {}
    for a in _tags(abp / 'entities2.xml', 'entity'):
        abp_ent[(a.get('id'), a.get('variant', '0'), a.get('subtype'))] = a.get('name')
    for a in _tags(rep / 'entities2.xml', 'entity'):
        t, v, s = a.get('id'), a.get('variant', '0'), a.get('subtype')
        if t is None:
            continue
        nm = _best(a.get('name'), abp_ent.get((t, v, s)))
        try:
            rad = float(a.get('collisionRadius', '0'))
        except ValueError:
            rad = 0.0
        rec = [nm, rad, 1 if a.get('boss') == '1' else 0]
        if s is not None:
            out['entities'].setdefault(f'{t}.{v}.{s}', rec)
        out['entities'].setdefault(f'{t}.{v}', rec)
        out['entities'].setdefault(t, rec)
    # 道具 / 饰品
    abp_items = {a.get('id'): a for tag in ('passive', 'active', 'familiar') for a in _tags(abp / 'items.xml', tag)}
    abp_tr = {a.get('id'): a for a in _tags(abp / 'items.xml', 'trinket')}
    item_gfx, tr_gfx = {}, {}
    for tag in ('passive', 'active', 'familiar'):
        for a in _tags(rep / 'items_2.xml', tag):
            i = a.get('id')
            if i is None:
                continue
            out['items'][i] = _best(a.get('name'), (abp_items.get(i) or {}).get('name'))
            item_gfx[i] = a.get('gfx') or (abp_items.get(i) or {}).get('gfx')
    for a in _tags(rep / 'items_2.xml', 'trinket'):
        i = a.get('id')
        if i is None:
            continue
        out['trinkets'][i] = _best(a.get('name'), (abp_tr.get(i) or {}).get('name'))
        tr_gfx[i] = a.get('gfx')
    abp_cards = {a.get('id'): a.get('name') for a in _tags(abp / 'pocketitems.xml', 'card')}
    abp_pills = {a.get('id'): a.get('name') for a in _tags(abp / 'pocketitems.xml', 'pilleffect')}
    for a in _tags(rep / 'pocketitems.xml', 'card'):
        if a.get('id') is not None:
            nm = pretty_key(a.get('name'))
            ab = abp_cards.get(a['id'])
            # AB+ 卡名带编号前缀（"0 - The Fool"），去掉前缀再比
            ab2 = ab.split(' - ', 1)[1] if ab and ' - ' in ab else ab
            out['cards'][a['id']] = ab if ab2 and _norm(ab2) == _norm(nm) else nm
    for a in _tags(rep / 'pocketitems.xml', 'pilleffect'):
        if a.get('id') is not None:
            out['pills'][a['id']] = _best(a.get('name'), abp_pills.get(a['id']))
    # 图标（AB+ 解包里有的才有；Repentance 新增道具没有）
    if want_icons and gfx.is_dir():
        for kind, ids, table, sub in (('c', want_icons.get('c', ()), item_gfx, 'collectibles'),
                                      ('t', want_icons.get('t', ()), tr_gfx, 'trinkets')):
            for i in ids:
                fn = table.get(str(i))
                if not fn:
                    continue
                p = gfx / sub / fn.lower()
                if p.is_file():
                    out['icons'][f'{kind}{i}'] = 'data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode()
    return out
