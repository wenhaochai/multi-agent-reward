#!/bin/bash
# mas-rl: every test, each in its own process, in miles.sif (python with aiohttp and yaml), on a node with Landlock
# (della-vis1 or a compute node; not a login node). Usage: bash tests/run_all.sh. Exit 0 when all pass.
cd "$(dirname "$0")/.."
SIF=/scratch/gpfs/GROUP/USER/project/miles-q38-build/miles.sif
export TMPDIR=${TMPDIR:-/scratch/gpfs/GROUP/USER/tmp}/mas_rl_tests_$$
mkdir -p "$TMPDIR"
fail=()
t() {   # t NAME ENV... -- runs tests/NAME.py with the environment given
  local name=$1; shift
  local out
  out=$(apptainer exec --bind /scratch/gpfs/GROUP "$SIF" env "$@" python3 "tests/$name.py" 2>&1)
  local rc=$?
  echo "$out" | grep -E "^FAIL " | sed "s/^/  $name: /"
  if [ $rc -eq 0 ]; then echo "PASS $name $*"; else echo "FAIL $name $* (exit $rc)"; echo "$out" | tail -5; fail+=("$name $*"); fi
}
t test_game MA_LG_N=4
t test_game MA_LG_N=1
t test_restart X=1
t test_finish X=1
t test_dedup X=1
t test_post X=1
t test_trace X=1
t test_resume X=1
t test_logs_symlink X=1
t test_own_repos FCS_CASE_PROC=0
t test_own_repos FCS_CASE_PROC=1
t test_chat X=1
t test_run_eval X=1
t test_harness_bin X=1
t test_cmd_escape FCS_CASE_PROC=1
t test_cmd_escape FCS_CASE_PROC=0
t test_git_nproc X=1
t test_judge_paths X=1
t test_judge_runs X=1
t test_judge_ratio X=1
t test_checker_fixes X=1
# miles' session matcher lives in miles v0.1.3: this test runs in its container
SIF=/scratch/gpfs/GROUP/USER/project/miles-q38-build/miles-v0.1.3.sif t test_miles_agent X=1
SIF=/scratch/gpfs/GROUP/USER/project/miles-q38-build/miles-v0.1.3.sif t test_wall_cap X=1
SIF=/scratch/gpfs/GROUP/USER/project/miles-q38-build/miles-v0.1.3.sif t test_train67 X=1
SIF=/scratch/gpfs/GROUP/USER/project/miles-q38-build/miles-v0.1.3.sif t test_train67_seed X=1
rm -rf "$TMPDIR"
echo "${#fail[@]} failed${fail:+: ${fail[*]}}"
[ ${#fail[@]} -eq 0 ]
