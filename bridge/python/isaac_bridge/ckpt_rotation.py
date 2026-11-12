"""Which of a training run's checkpoints the rotation keeps (train_tok's rotate_checkpoints; ISAAC_CKPT_KEEP /
ISAAC_CKPT_MILESTONE): only files named exactly update-<6 digits>.pt take part; the newest `keep` of them by update
number are kept, and every one whose update is a multiple of `milestone` (> 0); the rest are rotated away. No GPU,
no file system: names in, names out."""
import re

_NAME = re.compile(r'update-(\d{6})\.pt')


def update_of(name):
    """The update number of a rotation file name, None for any other name."""
    m = _NAME.fullmatch(name)
    return int(m.group(1)) if m else None


def rotation_drop(names, keep, milestone=0):
    """The names to rotate away (oldest first); keep <= 0: none (the rotation off)."""
    ups = sorted((u, n) for n in names if (u := update_of(n)) is not None)
    if keep <= 0:
        return []
    return [n for u, n in ups[:-keep] if not (milestone > 0 and u % milestone == 0)]
