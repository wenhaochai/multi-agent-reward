#!/bin/bash
# Build serving/flint_sglang into a directory of its own, for SGLang 0.5.23 in miles-v0.1.3.sif; serve with that
# directory on PYTHONPATH. The build runs on a copy, so the source tree stays clean. Usage: build_flint_sglang.sh
# <target dir> (on della-vis1 or in a job, not on a login node).
set -euo pipefail
T=${1:?target dir}
SRC=$(cd "$(dirname "$0")/flint_sglang" && pwd)
SIF=${MILES_SIF:-/scratch/gpfs/GROUP/USER/project/miles-q38-build/miles-v0.1.3.sif}
[ ! -e "$T" ] || { echo "$T exists: give a new directory"; exit 1; }
mkdir -p "$T"
COPY=$(mktemp -d)
trap 'rm -rf -- "$COPY"' EXIT
cp -r "$SRC/." "$COPY/"
apptainer exec --bind /scratch/gpfs/GROUP --bind "$COPY" "$SIF" \
  pip install -q --no-deps --no-build-isolation --no-index --target "$T" "$COPY"
apptainer exec --bind /scratch/gpfs/GROUP --env PYTHONPATH="$T" "$SIF" python3 -c '
from importlib.metadata import entry_points, version
assert any(e.value == "flint_sglang.plugin:register" for e in entry_points(group="sglang.srt.plugins"))
print("flint-sglang", version("flint-sglang"), "installed")'
