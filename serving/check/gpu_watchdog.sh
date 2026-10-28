# Sourced by the check jobs. gpu_watchdog SHELL_PID, in the background: every minute it reads the utilization of this
# job's GPUs (CUDA_VISIBLE_DEVICES); when one has stayed at 0% for IDLE_MAX minutes in a row (default 10), it prints
# GPU_IDLE_ABORT and sends SIGTERM to the job's shell, whose exit trap stops what the job started (procs.sh). It kills
# nothing itself, so a hang never leaves GPUs idle for long (a host whose GPU idles 90 min is cancelled with
# everything on it) and nothing outside the job is touched.
gpu_watchdog() {
  local idle=() u i
  while sleep 60; do
    mapfile -t u < <(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
    for i in "${!u[@]}"; do
      if [ "${u[$i]// /}" = 0 ]; then idle[i]=$(( ${idle[i]:-0} + 1 )); else idle[i]=0; fi
      if [ "${idle[i]}" -ge "${IDLE_MAX:-10}" ]; then
        echo "GPU_IDLE_ABORT: GPU $i of $CUDA_VISIBLE_DEVICES at 0% for ${idle[i]} min"
        kill -TERM "$1"
        return
      fi
    done
  done
}
