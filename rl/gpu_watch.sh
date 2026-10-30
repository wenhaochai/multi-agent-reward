# Sourced by rl/rollout_job.sh and rl/rl_job.sh. gpu_watch SHELL_PID LOG [READY_CMD LOAD_MIN], in the background: every
# minute it appends each of this job's GPUs (CUDA_VISIBLE_DEVICES) with its utilization and memory to LOG, and sends
# SIGTERM to the job's shell (whose exit trap stops what the job started) when a GPU has been at 0% for 10 minutes with
# nothing loaded on it (under 1 GiB: a dead or missing engine), or for GPU_IDLE_MIN minutes (default 20) with an engine
# loaded (an engine whose episodes ended early waits for the group's last one; the cluster mails at 30 minutes and
# cancels the host at 90). It kills nothing itself.
# With READY_CMD (a shell test, e.g. that the first rollout request is in the log): until it succeeds the job is
# loading (the actor's weights, then the engines; GPUs idle by design) and idle minutes are not counted, but the load
# phase may last at most LOAD_MIN minutes.
# GPU_ACTIVE_CMD (optional, a shell test; the external-engine trainer, rl/rl_job_ext.sh): idle minutes are counted
# only while it succeeds (a training phase: the trainer's GPUs idle by design while remote engines generate), and both
# counters restart when it fails. GPU_IDLE_MAX_MIN (optional): a GPU at 0% for that many minutes in any phase, the
# load phase included, stops the job too (the outer bound under the host's 90-min cancel).
gpu_watch() {
  local bare=() any=() all=() line i u m ready= minutes=0 active=1
  [ -n "${3:-}" ] || ready=1
  while sleep 60 && kill -0 "$1" 2>/dev/null; do   # ends with the job's shell, even after a KILL
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
    if [ -n "${GPU_ACTIVE_CMD:-}" ]; then
      if eval "$GPU_ACTIVE_CMD"; then active=1; else active=; fi
    fi
    i=0
    while IFS=', ' read -r idx u m; do
      echo "$(date -u +%FT%TZ) gpu $idx util $u mem_mib $m${GPU_ACTIVE_CMD:+ active ${active:-0}}" >> "$2"
      if [ -n "${GPU_IDLE_MAX_MIN:-}" ]; then
        if [ "$u" = 0 ]; then all[i]=$(( ${all[i]:-0} + 1 )); else all[i]=0; fi
      fi
      if [ -n "$ready" ]; then
        if [ -z "$active" ]; then any[i]=0; bare[i]=0
        else
          if [ "$u" = 0 ]; then any[i]=$(( ${any[i]:-0} + 1 )); else any[i]=0; fi
          if [ "$u" = 0 ] && [ "$m" -lt 1024 ]; then bare[i]=$(( ${bare[i]:-0} + 1 )); else bare[i]=0; fi
        fi
      fi
      if [ "${bare[i]:-0}" -ge 10 ] || [ "${any[i]:-0}" -ge "${GPU_IDLE_MIN:-20}" ] || { [ -n "${GPU_IDLE_MAX_MIN:-}" ] && [ "${all[i]:-0}" -ge "$GPU_IDLE_MAX_MIN" ]; }; then
        echo "GPU_IDLE_ABORT: GPU $idx at 0% for ${any[i]} min (${bare[i]} min with nothing loaded; ${all[i]} min in any phase)" | tee -a "$2"
        kill -TERM "$1"
        return
      fi
      i=$((i + 1))
    done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES")
  done
}
