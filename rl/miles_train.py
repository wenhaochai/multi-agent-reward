"""miles' train.py with SGLang's plugins loaded first: the Flint plugin (serving/flint_sglang, on PYTHONPATH) registers
the model type with transformers, which miles reads at argument parsing, and the flint parsers and grammar backend
among miles' --sglang-* choices. Usage: python3 rl/miles_train.py <train.py arguments> (in miles-v0.1.3.sif)."""
import runpy
import sys

from sglang.srt.plugins import load_plugins

load_plugins()
sys.argv[0] = "/root/miles/train.py"
runpy.run_path(sys.argv[0], run_name="__main__")
