# Sourced by the check jobs: they kill only what they started, never by GPU or by user (a pilot host runs other jobs
# of the same user on its other GPUs). start CMD...: run CMD in the background in a session of its own (setsid), so its
# process group holds it and its children; the group id is recorded in STARTED (and in $STARTED_FILE if that is set,
# for a wrapper that outlives this shell). stop GID...: SIGKILL each group and every process descended from its leader.
STARTED=()
start() {
  setsid "$@" &
  STARTED+=($!)
  [ -z "${STARTED_FILE:-}" ] || echo $! >> "$STARTED_FILE"
}
_descendants() { local c; for c in $(ps -o pid= --ppid "$1" 2>/dev/null); do echo "$c"; _descendants "$c"; done; }
stop() {
  local g pids
  for g in "$@"; do
    pids=$(_descendants "$g")
    kill -KILL -- "-$g" $pids "$g" 2>/dev/null
  done
  return 0
}
