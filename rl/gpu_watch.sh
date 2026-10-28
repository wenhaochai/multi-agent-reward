# Sourced by rl/rollout_job.sh and rl/rl_job.sh. gpu_watch SHELL_PID LOG [READY_CMD LOAD_MIN], in the background: every
# minute it appends each of this job's GPUs (CUDA_VISIBLE_DEVICES) with its utilization and memory to LOG, and sends
# SIGTERM to the job's shell (whose exit trap stops what the job started) when a GPU has been at 0% for 10 minutes with
# nothing loaded on it (under 1 GiB: a dead or missing engine), or for 20 minutes with an engine loaded (an engine whose
# episodes ended early waits for the group's last one; the cluster mails at 30 minutes). It kills nothing itself.
# With READY_CMD (a shell test, e.g. that the first rollout request is in the log): until it succeeds the job is
# loading (the actor's weights, then the engines; GPUs idle by design) and idle minutes are not counted, but the load
# phase may last at most LOAD_MIN minutes.
gpu_watch() {
  local bare=() any=() line i u m ready= minutes=0
  [ -n "${3:-}" ] || ready=1
  while sleep 60; do
    minutes=$((minutes + 1))
    if [ -z "$ready" ]; then
      if eval "$3"; then
        ready=1
        echo "$(date -u +%FT%TZ) ready after $minutes min" >> "$2"
      elif [ "$minutes" -ge "$4" ]; then
        echo "GPU_LOAD_ABORT: not ready after $minutes min" | tee -a "$2"
        kill -TERM "$1"
        return
      fi
    fi
    i=0
    while IFS=', ' read -r idx u m; do
      echo "$(date -u +%FT%TZ) gpu $idx util $u mem_mib $m" >> "$2"
      if [ -n "$ready" ]; then
        if [ "$u" = 0 ]; then any[i]=$(( ${any[i]:-0} + 1 )); else any[i]=0; fi
        if [ "$u" = 0 ] && [ "$m" -lt 1024 ]; then bare[i]=$(( ${bare[i]:-0} + 1 )); else bare[i]=0; fi
      fi
      if [ "${bare[i]:-0}" -ge 10 ] || [ "${any[i]:-0}" -ge 20 ]; then
        echo "GPU_IDLE_ABORT: GPU $idx at 0% for ${any[i]} min (${bare[i]} min with nothing loaded)" | tee -a "$2"
        kill -TERM "$1"
        return
      fi
      i=$((i + 1))
    done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
  done
}
