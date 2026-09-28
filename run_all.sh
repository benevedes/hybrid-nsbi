#!/usr/bin/env bash
# Reproduce every result and figure. With --figures, skip training and rebuild the evaluations and figures from the
# trained networks committed in results/. Training jobs are independent and run in parallel, at most MAX_JOBS at a time
# (default 4; each physics job needs ~6 GB of memory). With NGPUS=n, job k runs on GPU k mod n (CUDA_VISIBLE_DEVICES).
# Evaluation and figures run afterwards. Logs go to logs/.
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=${PYTHON:-python}
MAX_JOBS=${MAX_JOBS:-4}
NGPUS=${NGPUS:-0}
DATA=${LATENTCAT_DATA:-data}
mkdir -p logs

PIDS=""
NAMES=""
launch() {  # launch <name> <script> [args...]: background job, throttled to MAX_JOBS
  local name=$1
  shift
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$MAX_JOBS" ]; do sleep 5; done
  if [ "$NGPUS" -gt 0 ]; then
    CUDA_VISIBLE_DEVICES=$(( $(echo $NAMES | wc -w) % NGPUS )) $PY -u "$@" > "logs/$name.log" 2>&1 &
  else
    $PY -u "$@" > "logs/$name.log" 2>&1 &
  fi
  PIDS="$PIDS $!"
  NAMES="$NAMES $name"
  echo "started $name"
}
wait_all() {  # wait for every launched job; fail if any failed
  local failed=0 i=1
  for pid in $PIDS; do
    name=$(echo $NAMES | cut -d' ' -f$i)
    if wait "$pid"; then echo "finished $name"; else echo "FAILED $name (see logs/$name.log)"; failed=1; fi
    i=$((i + 1))
  done
  PIDS=""
  NAMES=""
  return $failed
}

if [ ! -f "$DATA/physics_cache_sbi.npz" ]; then
  $PY -u scripts/physics_cache_csv.py > logs/physics_cache.log 2>&1
fi

FIGURES_ONLY=0
[ "${1:-}" = "--figures" ] && FIGURES_ONLY=1

if [ "$FIGURES_ONLY" = 0 ]; then
# ---------------- training (longest first) ----------------
# the two CARL networks take ~7 h each on an A100 (resumable: rerunning continues from the last finished epoch)
launch physics_carl_sig scripts/physics_train_carl.py --proc sig
launch physics_carl_sbi scripts/physics_train_carl.py --proc sbi
for s in 0 1 2; do
  launch physics_event_s$s scripts/physics_train_event.py --tag event_s$s --seed $((500000 + s))
done
launch gaussian_nsbi scripts/gaussian_train_nsbi.py
for s in 0 1 2; do
  launch gaussian_event_s$s scripts/gaussian_train_event.py --tag event_s$s --seed $s
done
wait_all
fi

# ---------------- evaluation on the fine grids, then figures ----------------
$PY -u scripts/gaussian_eval.py > logs/gaussian_eval.log 2>&1
$PY -u scripts/gaussian_figure.py > logs/gaussian_figure.log 2>&1
$PY -u scripts/physics_eval.py > logs/physics_eval.log 2>&1
$PY -u scripts/physics_eval_carl.py > logs/physics_eval_carl.log 2>&1
$PY -u scripts/physics_figure.py > logs/physics_figure.log 2>&1
$PY -u scripts/gaussian_occupancy.py > logs/gaussian_occupancy.log 2>&1
$PY -u scripts/physics_occupancy.py > logs/physics_occupancy.log 2>&1
$PY -u scripts/gaussian_coverage.py > logs/gaussian_coverage.log 2>&1
$PY -u scripts/physics_coverage.py > logs/physics_coverage.log 2>&1
echo "all done"
