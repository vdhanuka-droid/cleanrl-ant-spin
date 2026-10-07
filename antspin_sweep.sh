#!/usr/bin/env bash
# Reproduce an AntSpin sweep: PPO (3M steps) and SAC (1M steps), paired seeds 1-5, with videos.
# Usage: [ENV_ID=AntSpin-v1] [SESSION=name] ./antspin_sweep.sh [OUT_DIR]   (defaults: AntSpin-v0, sweep, ./antspin_sweep)
# Each run gets its own window in tmux session $SESSION; outputs (runs/, videos/, logs/, diag/) land in OUT_DIR.
# GPU assignment matches the original sweep on an 8-GPU machine; edit SAC_GPUS / PPO_GPUS for yours.
set -euo pipefail
REPO=$(cd "$(dirname "$0")" && pwd)
OUT=$(mkdir -p "${1:-antspin_sweep}" && cd "${1:-antspin_sweep}" && pwd)
ENV_ID=${ENV_ID:-AntSpin-v0}
SESSION=${SESSION:-sweep}
SAC_GPUS=(0 1 2 3 4)
PPO_GPUS=(5 5 6 6 7)
cd "$OUT"
mkdir -p logs diag
git -C "$REPO" rev-parse HEAD > COMMIT
echo "$ENV_ID" > ENV_ID

ENVS="export PYTHONPATH=$REPO OMP_NUM_THREADS=4"
FILTER="grep -av 'GLContext\|__del__\|self.free\|_context\|Exception ignored'"  # drop harmless MuJoCo GL cleanup noise

launch() {  # name gpu algo seed steps
    local name=$1 gpu=$2 algo=$3 seed=$4 steps=$5
    local cmd="$ENVS; CUDA_VISIBLE_DEVICES=$gpu ANTSPIN_DIAG_LOG=$OUT/diag/${name}.csv \
uv run --project $REPO python $REPO/cleanrl/${algo}_continuous_action.py \
--env-id $ENV_ID --seed $seed --total-timesteps $steps --capture-video 2>&1 | $FILTER > logs/${name}.log; exec bash"
    if tmux has-session -t "$SESSION" 2>/dev/null; then
        tmux new-window -t "$SESSION" -n "$name" -c "$OUT" "$cmd"
    else
        tmux new-session -d -s "$SESSION" -n "$name" -c "$OUT" "$cmd"
    fi
}

for seed in 1 2 3 4 5; do
    launch "sac_s$seed" "${SAC_GPUS[$((seed - 1))]}" sac "$seed" 1000000
    launch "ppo_s$seed" "${PPO_GPUS[$((seed - 1))]}" ppo "$seed" 3000000
done
tmux list-windows -t "$SESSION"
