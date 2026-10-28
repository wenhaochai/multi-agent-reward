# Sourced by the check jobs. gpu_watchdog PID...: in the background, every minute, read the utilization of this job's
# GPUs (CUDA_VISIBLE_DEVICES); when one has stayed at 0% for IDLE_MAX minutes in a row (default 10), print
# GPU_IDLE_ABORT and kill the given processes, so a hang never leaves GPUs idle for long (a host whose GPU idles 90 min
# is cancelled with everything on it).
gpu_watchdog() {
  local pids=("$@") idle=() u i
  while sleep 60; do
    mapfile -t u < <(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
    for i in "${!u[@]}"; do
      if [ "${u[$i]// /}" = 0 ]; then idle[i]=$(( ${idle[i]:-0} + 1 )); else idle[i]=0; fi
      if [ "${idle[i]}" -ge "${IDLE_MAX:-10}" ]; then
        echo "GPU_IDLE_ABORT: GPU $i of $CUDA_VISIBLE_DEVICES at 0% for ${idle[i]} min, stopping ${pids[*]}"
        kill "${pids[@]}" 2>/dev/null
        return
      fi
    done
  done
}
