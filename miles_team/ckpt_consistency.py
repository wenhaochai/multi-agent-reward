"""Before a (resumed) miles run starts: make the actor and critic trackers point at a common iteration.
run_miles_in_container.sh runs it on <run>/ckpt and <run>/ckpt_critic. A wall-clock kill between the actor save and
the critic save leaves the trackers at different iterations; miles then fails its restored-rollout assert and every
chained segment crashes. Repairs (each printed):
  * both hold a common iteration -> point the newer tracker at the newest common one;
  * the actor's only checkpoint is newer than the critic's newest -> the critic is still in its critic-only phase (the
    actor saves only from --num-critic-only-steps on, while the actor is frozen at its initial weights), so the actor
    tracker is moved aside and miles resumes at the critic's rollout with the initial actor (easyppo 6a8976bdf);
  * anything else is left alone and reported (the run's own assert stops it).
usage: python3 -m miles_team.ckpt_consistency <actor_ckpt_dir> <critic_ckpt_dir>
"""
import sys
from pathlib import Path

from miles_team.ckpt_rotate import iterations

TRACKER = "latest_checkpointed_iteration.txt"


def read(d: Path):
    f = d / TRACKER
    try:
        return int(f.read_text().strip())
    except (OSError, ValueError):
        return None


def repair(actor: Path, critic: Path) -> str:
    a, c = read(actor), read(critic)
    if a is None or c is None or a == c:
        return f"consistent (actor {a}, critic {c})"
    common = sorted(set(iterations(actor)) & set(iterations(critic)))
    if common:
        k = common[-1]
        for d, t in ((actor, a), (critic, c)):
            if t != k:
                (d / TRACKER).write_text(f"{k}\n")
        return f"repaired: actor {a}, critic {c} -> both {k}"
    if a > c:
        (actor / TRACKER).rename(actor / (TRACKER + f".moved_{a}"))
        return (f"repaired: actor {a} has no common iteration with critic {c}; actor tracker moved aside, the run "
                f"resumes at the critic's rollout with the initial actor (valid only inside the critic-only phase)")
    return f"NOT repaired: actor {a}, critic {c}, no common iteration"


if __name__ == "__main__":
    print(f"[ckpt_consistency] {repair(Path(sys.argv[1]), Path(sys.argv[2]))}", flush=True)
