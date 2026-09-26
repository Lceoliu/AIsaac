"""Item prior table for AB+ v1.06: definitions and pools from the AB+ archive, quality from Rep+.

Sources
- resources/items.xml and resources/itempools.xml inside AB+ afterbirthp.a (read through
  archive.py): kind, name, cache flags, charges, devil price, unlock achievement; 26 pools with
  Weight / DecreaseBy / RemoveOn per item.
- Quality (0-4) and tags come from Repentance+ (analysis/resources/repentance-a/config/items.xml,
  the file the Rep+ archive calls items.xml with <item id quality tags>). AB+ has no quality data;
  collectible IDs 1-552 are shared, but Rep+ rebalanced some items, so treat it as a prior only.
- WEAPON_CHANGERS is a hand-made list (wiki knowledge, not game data): items that replace or
  reshape the tear attack, which the combat policy has never been trained with.

usage: python -m isaac_macro.items [--csv out.csv] [--pools]
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .archive import ArchiveSet
from .roomconfig import default_archive_path

# Replace or reshape the tear attack (hand-made list, AB+ IDs).
WEAPON_CHANGERS = {
    52: 'Dr. Fetus', 68: 'Technology', 69: "Chocolate Milk", 114: "Mom's Knife", 118: 'Brimstone',
    149: 'Ipecac', 152: 'Technology 2', 168: 'Epic Fetus', 222: 'Anti-Gravity', 229: "Monstro's Lung",
    233: 'Tiny Planet', 316: 'Cursed Eye', 329: 'The Ludovico Technique', 394: 'Marked', 395: 'Tech X',
    531: 'Haemolacria',
}
# Quality-0 items that hurt a bot directly (hand-made list, see rl/docs/MACRO_PLANNING.md §4.3).
HARMFUL = {
    475: 'Plan C', 40: 'Kamikaze!', 186: 'Blood Rights', 126: 'Razor Blade', 233: 'Tiny Planet',
    358: 'The Wiz', 258: 'Missing No.',
}


@dataclass
class Item:
    id: int
    kind: str                     # passive / active / familiar / trinket
    name: str
    description: str = ''
    cache: tuple = ()
    max_charges: int = 0
    devil_price: int = 1          # heart containers (AB+ default 1 when the attribute is absent)
    achievement: int | None = None
    special: bool = False
    quality: int | None = None    # Rep+ quality, None when Rep+ has no entry
    tags: tuple = ()
    pools: dict = field(default_factory=dict)   # pool name -> weight

    @property
    def quest(self) -> bool:
        return 'quest' in self.tags

    @property
    def weapon_changer(self) -> bool:
        return self.kind != 'trinket' and self.id in WEAPON_CHANGERS

    @property
    def harmful(self) -> bool:
        return self.kind != 'trinket' and self.id in HARMFUL


@dataclass
class PoolEntry:
    id: int
    weight: float
    decrease_by: float
    remove_on: float


def _attrs(text: str) -> dict:
    return dict(re.findall(r'(\w+)="([^"]*)"', text))


def load_items(archives: ArchiveSet) -> dict[tuple[str, int], Item]:
    """Every entry of AB+ items.xml, keyed by ('collectible' | 'trinket', id)."""
    xml = archives.read('resources/items.xml').decode('utf-8-sig')
    out = {}
    for m in re.finditer(r'<(passive|active|familiar|trinket)\s([^>]*?)/?>', xml):
        kind, a = m.group(1), _attrs(m.group(2))
        item = Item(int(a['id']), kind, a.get('name', ''), a.get('description', ''),
                    tuple(a.get('cache', '').split()), int(a.get('maxcharges', 0) or 0),
                    int(a.get('devilprice', 1) or 1),
                    int(a['achievement']) if a.get('achievement') else None,
                    a.get('special', 'false') == 'true')
        out[('trinket' if kind == 'trinket' else 'collectible', item.id)] = item
    return out


def load_pools(archives: ArchiveSet) -> dict[str, list[PoolEntry]]:
    xml = archives.read('resources/itempools.xml').decode('utf-8-sig')
    pools = {}
    for m in re.finditer(r'<Pool\s+Name="([^"]+)"[^>]*>(.*?)</Pool>', xml, re.S):
        entries = []
        for e in re.finditer(r'<Item\s([^>]*?)/?>', m.group(2)):
            a = _attrs(e.group(1))
            entries.append(PoolEntry(int(a['Id']), float(a.get('Weight', 1)), float(a.get('DecreaseBy', 1)),
                                     float(a.get('RemoveOn', 0.1))))
        pools[m.group(1)] = entries
    return pools


def default_rep_metadata_path() -> str:
    return str(Path(__file__).resolve().parents[3] / 'analysis' / 'resources' / 'repentance-a' / 'config'
               / 'items.xml')


def load_rep_metadata(path: str | None = None) -> dict[int, tuple[int | None, tuple]]:
    """Rep+ <item id quality tags> entries: id -> (quality, tags). Empty when the file is absent."""
    p = Path(path or default_rep_metadata_path())
    if not p.exists():
        return {}
    out = {}
    for m in re.finditer(r'<item\s([^>]*?)/?>', p.read_text(encoding='utf-8-sig')):
        a = _attrs(m.group(1))
        out[int(a['id'])] = (int(a['quality']) if 'quality' in a else None, tuple(a.get('tags', '').split()))
    return out


class ItemTable:
    def __init__(self, archives: ArchiveSet | None = None, rep_metadata: str | None = None):
        archives = archives or ArchiveSet([default_archive_path()])
        self.items = load_items(archives)
        self.pools = load_pools(archives)
        meta = load_rep_metadata(rep_metadata)
        for (kind, i), item in self.items.items():
            if kind == 'collectible' and i in meta:
                item.quality, item.tags = meta[i]
        for name, entries in self.pools.items():
            for e in entries:
                it = self.items.get(('collectible', e.id))
                if it is not None:
                    it.pools[name] = e.weight

    def collectible(self, item_id: int) -> Item:
        return self.items[('collectible', item_id)]

    def collectibles(self) -> list[Item]:
        return sorted((it for (k, _), it in self.items.items() if k == 'collectible'), key=lambda it: it.id)

    def pool_quality(self, pool: str) -> dict:
        """Quality distribution of one draw from a fresh pool (P(item) proportional to Weight)."""
        entries = self.pools[pool]
        total = sum(e.weight for e in entries)
        dist = [0.0] * 5
        unknown = 0.0
        for e in entries:
            it = self.items.get(('collectible', e.id))
            q = it.quality if it else None
            if q is None:
                unknown += e.weight / total
            else:
                dist[q] += e.weight / total
        known = sum(dist)
        mean = sum(q * p for q, p in enumerate(dist)) / known if known else float('nan')
        return dict(pool=pool, items=len(entries), total_weight=total, mean_quality=mean,
                    p_q3plus=(dist[3] + dist[4]) / known if known else float('nan'),
                    p_q4=dist[4] / known if known else float('nan'), unknown=unknown,
                    dist=[d / known if known else float('nan') for d in dist])

    def prior_value(self, item: Item) -> float | None:
        """Rule-level prior used by the v0 planner: quality, with weapon changers and harmful items
        at -1 and quest items left to the route rules (None)."""
        if item.quest:
            return None
        if item.harmful or item.weapon_changer:
            return -1.0
        return float(item.quality) if item.quality is not None else None

    def rows(self) -> list[dict]:
        out = []
        for it in self.collectibles():
            out.append(dict(id=it.id, name=it.name, kind=it.kind, quality=it.quality,
                            prior=self.prior_value(it), quest=it.quest, weapon_changer=it.weapon_changer,
                            harmful=it.harmful, max_charges=it.max_charges, devil_price=it.devil_price,
                            achievement=it.achievement, cache=' '.join(it.cache), tags=' '.join(it.tags),
                            pools=' '.join(f'{k}:{v:g}' for k, v in sorted(it.pools.items())),
                            description=it.description))
        return out


def main(argv=None) -> None:
    import argparse
    import csv
    import sys
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--csv', help='write the collectible table to this CSV file')
    ap.add_argument('--pools', action='store_true', help='print the per-pool quality table')
    args = ap.parse_args(argv)
    table = ItemTable()
    cols = table.collectibles()
    print(f'{len(cols)} collectibles, {sum(1 for (k, _) in table.items if k == "trinket")} trinkets, '
          f'{len(table.pools)} pools; with Rep+ quality: {sum(1 for c in cols if c.quality is not None)}')
    if args.pools:
        print(f"{'pool':16s} {'items':>5s} {'meanQ':>6s} {'P(q>=3)':>8s} {'P(q=4)':>7s}")
        for name in table.pools:
            s = table.pool_quality(name)
            print(f"{name:16s} {s['items']:5d} {s['mean_quality']:6.2f} {s['p_q3plus']:8.2f} {s['p_q4']:7.2f}")
    if args.csv:
        rows = table.rows()
        with open(args.csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print('wrote', args.csv, file=sys.stderr)


if __name__ == '__main__':
    main()
