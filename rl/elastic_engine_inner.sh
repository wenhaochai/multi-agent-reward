#!/bin/bash
# Inside miles-v0.1.3.sif (rl/elastic_engine.sh): one SGLang engine of an elastic engine job, serving exactly as the
# anchored run's own colocated engines (rl/elastic_engine_argv.py derives the argv from the run's spec and prints what
# differs). Env: IP (the node's routable address), PORT, BASE (40 free ports from BASE), SEED_OFFSET, PKG (the built
# Flint plugin), TMPDIR, SPEC (RUN/engine_spec.json), REPO and MILES (the run's code, from the spec), and
# CUDA_VISIBLE_DEVICES (this engine's GPUs). Usage: elastic_engine_inner.sh serve | argv (print the launch line, exit).
set -uo pipefail
: "${IP:?}" "${PORT:?}" "${BASE:?}" "${SEED_OFFSET:?}" "${PKG:?}" "${TMPDIR:?}" "${SPEC:?}" "${REPO:?}" "${MILES:?}"
export HOME=$TMPDIR/home_$PORT TRITON_CACHE_DIR=$TMPDIR/triton_$PORT TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor_$PORT
export SGLANG_CACHE_DIR=$TMPDIR/sglang_$PORT FLASHINFER_WORKSPACE_BASE=$TMPDIR/flashinfer_$PORT
mkdir -p "$HOME"
export PYTHONPATH=$PKG:$REPO:$MILES PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export no_proxy=127.0.0.1,localhost,$IP
# the weight push from the trainer's node: NCCL between separate apptainer instances needs cuMem off on both sides
# (w545); bootstrap over eno8303, IB found by NCCL
export NCCL_CUMEM_ENABLE=0 NCCL_SOCKET_IFNAME=eno8303
out=$(python3 "$REPO/rl/elastic_engine_argv.py" "$SPEC" "$IP" "$PORT" "$BASE" "$SEED_OFFSET") || { echo "[engine] argv failed"; exit 1; }
env_line=$(grep '^ENV ' <<< "$out" | cut -d' ' -f2-)
argv_line=$(grep '^ARGV ' <<< "$out" | cut -d' ' -f2-)
[ -n "$argv_line" ] || { echo "[engine] no argv: $out"; exit 1; }
for kv in $env_line; do export "$kv"; done   # miles' engine environment of the colocated run, then the spec's
echo "[engine] env: $env_line"
echo "[engine] differs from the run's colocated engine in: $(grep '^DIFF ' <<< "$out" | cut -d' ' -f2-)"
echo "[engine] $(date -u +%FT%TZ) launch: $argv_line"
[ "${1:-serve}" = argv ] && exit 0
eval "exec $argv_line"
