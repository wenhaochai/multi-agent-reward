"""miles' GPU TITO check (scripts/tools/verify_session_tito_tokenizer.py in miles v0.1.3) for Flint, on the Ray head
this job started: the verifier's own arguments, agent and assertions, with two changes. train.py runs here instead of
through miles' launcher, which kills every sglang and ray process it can see; and Flint's serving flags, which the
verifier does not pass on, follow the "--". miles is the source on PYTHONPATH (our fork, which loads SGLang's plugins
before it checks SGLang server args). Usage: tito_verify.py <verifier flags> -- <extra train.py flags>. MILES_TITO_MODEL and
MILES_SESSION_VERIFY_METRICS_PATH must be set before the Ray head starts: its rollout workers write the metrics."""
import os
import runpy
import shlex
import subprocess
import sys

import miles
from miles.utils.arguments import parse_args
from miles.utils.chat_template_utils import resolve_reasoning_and_tool_call_parser
from miles.utils.test_utils import session_verify_runner as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(miles.__file__)))   # the miles source in use
VERIFY = runpy.run_path(os.path.join(ROOT, "scripts/tools/verify_session_tito_tokenizer.py"))   # its functions, not its main
cut = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
extra = sys.argv[cut + 1:]
sys.argv = [sys.argv[0], *VERIFY["_with_session_verify_defaults"](sys.argv[1:cut])]
args = parse_args(add_custom_arguments=R.session_verify_extras)
args.sglang_reasoning_parser, args.sglang_tool_call_parser = resolve_reasoning_and_tool_call_parser(
    args.tito_model, args.sglang_reasoning_parser, args.sglang_tool_call_parser)
VERIFY["_print_action_table"](list(VERIFY["fixed_template_append_roles"](args.tito_model)), cycles=args.session_verify_cycles)
R._ensure_prompt_data()
metrics = os.environ["MILES_SESSION_VERIFY_METRICS_PATH"]
assert os.environ.get("MILES_TITO_MODEL") == args.tito_model, "MILES_TITO_MODEL must match --tito-model"
cmd = [sys.executable, os.path.join(ROOT, "train.py"), *shlex.split(R.namespace_to_train_args(args)), *extra]
print("train.py", " ".join(cmd[2:]), flush=True)
rc = subprocess.call(cmd)
if rc:
    sys.exit(f"Verdict: FAIL -- train.py exit {rc}")
R.assert_session_verify_metrics(metrics, assistant_text_threshold=args.assistant_text_threshold,
                                special_token_count_threshold=getattr(args, "ci_tito_special_token_count_threshold", 0.0),
                                require_append_tool=True)
print("Verdict: PASS -- TITO incremental tokenization matched standard re-tokenize across all required driver actions.")
