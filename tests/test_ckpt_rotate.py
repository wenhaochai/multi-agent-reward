"""miles_team.ckpt_rotate + ckpt_consistency on fake checkpoint dirs (no GPU, no miles): the save sequences of an
EasyPPO run (critic-only phase, first actor save, steady state) and wall-clock kills between the actor and the critic
save.  python3 tests/test_ckpt_rotate.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib.machinery  # noqa: E402
import types  # noqa: E402

_M = Path(__file__).resolve().parents[1] / "miles_team"
_src = _M / "ckpt_rotate.py.new" if (_M / "ckpt_rotate.py.new").exists() else _M / "ckpt_rotate.py"
R = types.ModuleType("ckpt_rotate")
importlib.machinery.SourceFileLoader("ckpt_rotate", str(_src)).exec_module(R)
sys.modules["miles_team.ckpt_rotate"] = R
import miles_team.ckpt_consistency as C  # noqa: E402

T = "latest_checkpointed_iteration.txt"


def save(role: Path, it: int, rotate=True):
    d = role / f"iter_{it:07d}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text("{}")
    (role / T).write_text(f"{it}\n")
    if rotate:
        R.keep_latest(None, it, str(d))


def debug_only(role: Path, it: int):
    (role / f"iter_{it:07d}" / "debug_events").mkdir(parents=True, exist_ok=True)


with tempfile.TemporaryDirectory() as td:
    a, c = Path(td) / "ckpt", Path(td) / "ckpt_critic"
    a.mkdir(), c.mkdir()
    # critic-only phase: the actor writes only debug_events dirs, the critic keeps only its newest save
    for it in (9, 19, 29):
        debug_only(a, it)
        save(c, it)
    assert R.iterations(c) == [29] and R.iterations(a) == [], (R.iterations(c), R.iterations(a))
    assert sorted(p.name for p in a.iterdir()) == ["iter_0000009", "iter_0000019", "iter_0000029"]
    # first actor save: until the critic saves 39 the critic keeps 29
    save(a, 39)
    assert R.iterations(a) == [39] and R.iterations(c) == [29]
    # kill here: no common iteration, actor newer -> actor tracker moved aside (critic-only resume)
    msg = C.repair(a, c)
    assert "moved aside" in msg and not (a / T).exists() and (a / f"{T}.moved_39").exists(), msg
    (a / f"{T}.moved_39").rename(a / T)
    save(c, 39)
    assert R.iterations(a) == [39] and R.iterations(c) == [39]
    # steady state: actor 49 written, critic still 39 -> both keep 39
    save(a, 49)
    assert R.iterations(a) == [39, 49] and R.iterations(c) == [39]
    # kill here: common 39 -> the actor tracker goes back to 39
    msg = C.repair(a, c)
    assert "both 39" in msg and (a / T).read_text().strip() == "39" and (c / T).read_text().strip() == "39", msg
    (a / T).write_text("49\n")
    save(c, 49)
    assert R.iterations(a) == [49] and R.iterations(c) == [49], (R.iterations(a), R.iterations(c))
    assert C.repair(a, c).startswith("consistent")
    # a run without a critic dir: only the role's newest save is kept
    solo = Path(td) / "solo" / "ckpt"
    solo.mkdir(parents=True)
    for it in (4, 9, 14):
        save(solo, it)
    assert R.iterations(solo) == [14]
print("ALL OK")
