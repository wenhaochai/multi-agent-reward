"""The team's git repositories. Each agent works in a repository of its own that only it changes (work/<name>); after
each of its commands the harness commits its source files under its name and publishes its HEAD to a bare team.git as
branch <name> (only the harness writes team.git); before each command it fetches every branch into the agent's
repository as team/agent<k>, so a teammate's work is one `git diff`, `git show` or `git checkout` away. A lone agent
works in one repository (work itself), committed the same way.

The harness's git runs with no global config and no hooks, fsmonitor, signing or background gc from the repository's
.git; any other config an agent planted there (filter.*.clean, ...) runs in the agents' sandbox, as the whole git call
does (sandbox.run_git; audit 2026-10-24)."""
from __future__ import annotations

import hashlib
import os
import subprocess

from . import Aborted
from . import sandbox as C

_GIT_SAFE = ["-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false", "-c", "commit.gpgSign=false",
             "-c", "tag.gpgSign=false", "-c", "gc.auto=0"]
MAX_COMMIT_BYTES = 1 << 20   # files the harness commits; a large .in/.txt stays uncommitted (git timeout, review 2026-10-20)
MAX_FILE = 256 << 10         # bytes of a submitted file
SOURCES = (".cpp", ".cc", ".h", ".hpp", ".py", ".txt", ".md", ".sh", ".in", ".out", ".ans")  # binaries stay out


def git(work: str, *args, env: dict | None = None, read: tuple = ()) -> subprocess.CompletedProcess:
    e = {"PATH": "/usr/bin:/bin", "LANG": "C", "HOME": work, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
         **(env or {})}
    try:
        return C.run_git(work, [*_GIT_SAFE, *args], e, read=read)
    except (subprocess.SubprocessError, OSError, C.J._WorkerDied) as ex:   # a timeout or a failing preexec_fn: the
        # episode is aborted and retried
        raise Aborted(f"harness git {args[:1]}: {type(ex).__name__}") from ex


def _ok(r):
    if r.returncode:
        raise Aborted(f"own repos: git {r.args[1 + len(_GIT_SAFE):][:2]} exit {r.returncode}: {r.stderr.strip()[:300]}")
    return r


def repo_init(work: str) -> None:
    git(work, "init", "-q", "-b", "main")
    who = {"GIT_AUTHOR_NAME": "harness", "GIT_AUTHOR_EMAIL": "harness@team", "GIT_COMMITTER_NAME": "harness",
           "GIT_COMMITTER_EMAIL": "harness@team"}
    git(work, "commit", "-q", "--allow-empty", "-m", "start", env=who)


def commit_as(work: str, name: str, step: int, staged: bool = False) -> str:
    """The harness's commit after a command: the repository's source files, under the acting agent's name. staged: also
    commit what the agent staged itself when no file needs adding (`git checkout <rev> -- file` and `git add` update
    the index, so ls-files --modified misses them; own repositories). Returns "" or the git steps that failed ("nothing
    to commit" is not a failure)."""
    who = {"GIT_AUTHOR_NAME": name, "GIT_AUTHOR_EMAIL": f"{name}@team", "GIT_COMMITTER_NAME": name,
           "GIT_COMMITTER_EMAIL": f"{name}@team"}
    ls = git(work, "ls-files", "-z", "--others", "--modified", "--deleted", "--exclude-standard")
    changed = ls.stdout
    files = sorted({f for f in changed.split("\0") if f.endswith(SOURCES)
                    and not (os.path.isfile(os.path.join(work, f)) and os.path.getsize(os.path.join(work, f)) > MAX_COMMIT_BYTES)})
    fail = [f"git ls-files exit {ls.returncode}"] if ls.returncode else []
    if files:  # (a pathspec that matches nothing would fail the whole add)
        r = git(work, "add", "-A", "--", *files, env=who)
        if r.returncode:
            fail.append(f"git add exit {r.returncode}")
        r = git(work, "commit", "-q", "-m", f"{name} step {step}", env=who)
        if r.returncode and not any(t in r.stdout for t in ("nothing to commit", "nothing added to commit",
                                                              "no changes added to commit")):
            fail.append(f"git commit exit {r.returncode}")
    elif staged and git(work, "diff", "--cached", "--quiet").returncode == 1:
        r = git(work, "commit", "-q", "-m", f"{name} step {step}", env=who)
        if r.returncode:
            fail.append(f"git commit exit {r.returncode}")
    return "; ".join(fail)


def blame_shares(work: str, path: str = "solution.cpp", rev: str = "HEAD") -> dict:
    """{author: share of the file's non-blank lines}, from git blame of the committed file."""
    r = git(work, "blame", "--line-porcelain", rev, "--", path)
    if r.returncode:
        return {}
    counts, author = {}, None
    for line in r.stdout.splitlines():
        if line.startswith("author "):
            author = line[7:]
        elif line.startswith("\t"):
            if line[1:].strip():
                counts[author] = counts.get(author, 0) + 1
    total = sum(counts.values())
    return {a: c / total for a, c in counts.items()} if total else {}


def own_init(work: str, names: list) -> None:
    """work/team.git (bare) and a repository work/<name> per agent, all on one empty start commit, each with the agent
    as its git user and SCORES.md and BOARD.md (written by the harness) never committed."""
    team = os.path.join(work, "team.git")
    dirs = [os.path.join(work, n) for n in names]
    for d in [team] + dirs:
        os.makedirs(d)
    repo_init(dirs[0])
    _ok(git(team, "init", "-q", "--bare", "-b", "main"))
    _ok(publish(work, names[0]))
    for n, d in zip(names, dirs):
        if d != dirs[0]:
            _ok(git(d, "init", "-q", "-b", "main"))
            _ok(fetch_team(work, d))
            _ok(git(d, "reset", "-q", "--hard", f"team/{names[0]}"))
        _ok(git(d, "config", "user.name", n))
        _ok(git(d, "config", "user.email", f"{n}@team"))
        os.makedirs(os.path.join(d, ".git", "info"), exist_ok=True)
        with open(os.path.join(d, ".git", "info", "exclude"), "a") as f:
            f.write("SCORES.md\nBOARD.md\n")
    for n, d in zip(names, dirs):
        _ok(publish(work, n))
    for d in dirs:
        _ok(fetch_team(work, d))


def publish(work: str, name: str):
    """team.git's branch <name> := HEAD of agent <name>'s repository: git with team.git's rights, reading the agent's
    repository (whose hooks and config cannot write team.git: _GIT_SAFE, and upload-pack runs no repository hook)."""
    d = os.path.join(work, name)
    return git(os.path.join(work, "team.git"), "fetch", "-q", "--no-tags", d, f"+HEAD:refs/heads/{name}", read=(d,))


def fetch_team(work: str, d: str):
    """Every team.git branch into the agent's repository d as refs/remotes/team/*: by path, not by a remote that the
    agent could redefine; with the agent's rights, reading team.git."""
    team = os.path.join(work, "team.git")
    return git(d, "fetch", "-q", "--no-tags", "--prune", team, "+refs/heads/*:refs/remotes/team/*", read=(team,))


# a git operation in progress in an agent's repository (its state files in .git): the harness's commit would conclude
# it, conflict markers and the teammate's authorship included (audit 2026-10-25), so own_commit skips the commit
BUSY_OP = {"MERGE_HEAD": "merge", "CHERRY_PICK_HEAD": "cherry-pick", "REVERT_HEAD": "revert", "rebase-merge": "rebase",
           "rebase-apply": "rebase"}


def own_commit(work: str, name: str, step: int) -> dict:
    """After each action of agent <name>: commit its source files (commit_as) unless its .git is not a plain directory
    or a git operation is in progress there, then publish its HEAD. Returns {"head": HEAD or "", "skipped": the reason
    nothing was committed or "", "commit_fail": "" or the failed steps, "publish_rc": 0 or the exit code}; a repository
    without .git is neither committed nor published."""
    d = os.path.join(work, name)
    gd = os.path.join(d, ".git")
    out = {"head": "", "skipped": "", "commit_fail": "", "publish_rc": 0}
    if os.path.islink(gd) or not os.path.isdir(gd):
        out["skipped"] = "no .git directory"
        return out
    out["skipped"] = next((f for f in BUSY_OP if os.path.exists(os.path.join(gd, f))), "")
    if not out["skipped"]:
        out["commit_fail"] = commit_as(d, name, step, staged=True)
    out["publish_rc"] = publish(work, name).returncode
    h = git(d, "rev-parse", "-q", "--verify", "HEAD^{commit}")
    out["head"] = h.stdout.strip() if h.returncode == 0 else ""
    return out


def holds(d: str, head: str, path: str, code: str) -> str:
    """The repository path of the file at real path `path` when commit `head` of repository d has it with exactly
    `code`, else ""."""
    rel = os.path.relpath(path, os.path.realpath(d))
    r = git(d, "show", f"{head}:{rel}")
    return rel if r.returncode == 0 and r.stdout == code else ""


def sol_sha(d: str) -> str | None:
    """sha1 of d/solution.cpp (a regular file, at most MAX_FILE bytes read), or None."""
    p = os.path.join(d, "solution.cpp")
    try:
        if os.path.islink(p) or not os.path.isfile(p):
            return None
        with open(p, "rb") as f:
            return hashlib.sha1(f.read(MAX_FILE + 1)).hexdigest()
    except OSError:
        return None
