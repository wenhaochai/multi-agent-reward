"""One team or lone episode, for test_dedup: python3 tests/dedup_ab_run.py TREE OUT. TREE is the code tree whose flint
package runs (this tree, or rl-train67's 864a904 exported). Real ChatSession over a fake HTTP server, so
OUT gets every request body the model would get, byte for byte, then the trace with wall-clock fields dropped and git
hashes and work-dir names as H. Scripted replies (two calls in one response, submissions, posts, a teammate's file,
a finish command, submits of unchanged code, a trim every few steps), stubbed judge, real problem 0. The game's
settings come from the environment (MA_LG_N, MA_LG_TOKENS, MA_LG_CTX_CHARS, MA_LG_DEDUP_SUBMIT). OUT also gets the
number of judge calls."""
import asyncio
import json
import json as J   # _Http.post's argument is named json (aiohttp's keyword)
import re
import sys

from helpers import CLOCK, FIN, LABEL, MSG, env, inp, traces

TREE, OUT = sys.argv[1], sys.argv[2]
env()
sys.path.insert(0, TREE)   # before helpers' ROOT
from flint import chat as CH  # noqa: E402
from flint import game as G  # noqa: E402

assert G.__file__.startswith(TREE), (G.__file__, TREE)
HEX = re.compile(r"\b[0-9a-f]{7,40}\b")
SCRIPTS = {
    "agent1": ["printf 'int main(){return 0;}// P=40\\n' > solution.cpp", "submit", "post a1 wrote v1 (40)", "submit",
               "post a1 again",
               ["echo n1 >> notes.md", "cat notes.md"], "git log --format='%an %s' team/agent2 | head -3",
               "printf 'int main(){int a;return 0;}// P=55\\n' > solution.cpp", FIN, "post a1 v2 (55)", FIN, "true"],
    "agent2": ["ls", "printf 'int main(){return 2;}// P=70\\n' > solution.cpp", "cat BOARD.md", "submit",
               "post a2 70", "git checkout team/agent1 -- solution.cpp && cat solution.cpp", "cat SCORES.md",
               ["true", "echo two"], "true"],
    "agent3": ["echo hi > b.txt", "submit b.txt", "touch x", "post a3 b.txt", "submit b.txt", "post a3 b.txt again",
               "cat SCORES.md BOARD.md", "true"],
    "agent4": ["true", "<no call>", "<no call>", "<no call>", "ls", "true"],
}


class _Resp:
    def __init__(self, d):
        self.d, self.status = d, 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    async def json(self):
        return self.d


class _Http:
    """The model of one agent: its k-th call answers SCRIPT[k] (a list: several calls in one response; "<no call>":
    a reply without a call, a format error)."""

    def __init__(self, name, log):
        self.name, self.log, self.k = name, log, 0

    def post(self, url, json):
        self.log.append([self.name, self.k, J.dumps(json, sort_keys=True)])   # the body as sent, now
        s = SCRIPTS[self.name] if self.name in SCRIPTS else SCRIPTS["agent1"]
        step = s[self.k] if self.k < len(s) else "true"
        self.k += 1
        cmds = [] if step == "<no call>" else (step if isinstance(step, list) else [step])
        tcs = [{"id": "x", "type": "function", "function": {"name": "bash", "arguments": J.dumps({"command": c})}}
               for c in cmds]
        usage = {"prompt_tokens": 100 * len(json["messages"]), "completion_tokens": 1000}
        return _Resp({"choices": [{"message": {"reasoning_content": f"r{self.k}", "content": f"plan {self.k}",
                                               "tool_calls": tcs}, "finish_reason": "tool_calls"}], "usage": usage})


judged = []


async def fake_judge(label, code):
    judged.append(code)
    m = re.search(r"P=(\d+)", code)
    return {"cases": [int(m.group(1)) / 100 if m else 0.1] * 10, "status": "ok", "infra_error": "", "msg": ""}


log = []


def new_session(input, name):
    c = CH.Client(["http://x"], "flint")
    c.http = _Http(name, log)
    return CH.ChatSession(c, input, name, G.MAX_LEN)


def norm(x):
    if isinstance(x, dict):
        return {k: norm(v) for k, v in x.items() if k not in CLOCK and k != "time"}
    if isinstance(x, (list, tuple)):
        return [norm(v) for v in x]
    return sub(x) if isinstance(x, str) else x


def sub(s: str) -> str:   # the run's TMPDIR and git hashes differ between any two runs
    return HEX.sub("H", s.replace(TMP, "TMPDIR"))


TMP = __import__("os").environ["TMPDIR"]
G.judge_code = fake_judge
agents, S, info = asyncio.run(G.episode(inp(2, 0), MSG, LABEL, new_session))
rec = traces()[-1]
# git log --all lists commits of the same second in no fixed order (main against itself differs): compared sorted
rec["git_log"] = sorted(rec["git_log"].splitlines())
with open(OUT, "w") as f:
    json.dump({"requests": [[n, k, sub(b)] for n, k, b in log],
               "trace": norm(rec), "S": S, "info": norm({k: v for k, v in info.items() if not k.endswith("_min")}),
               "chats": norm([a.chat for a in agents]), "judged": len(judged)}, f, sort_keys=True)
print("requests", len(log), "rounds", rec["rounds"], "subs", len(rec["submissions"]),
      "trimmed turns", [sum(bool(t.get("dropped")) for t in a.turns) for a in agents], "restarts",
      [a.restarts for a in agents])
