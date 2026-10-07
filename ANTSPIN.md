# AntSpin: teaching the MuJoCo Ant to spin upright in place

Custom Gymnasium environment on top of CleanRL ([vwxyzjn/cleanrl@fe8d8a0](https://github.com/vwxyzjn/cleanrl/commit/fe8d8a0)),
trained with PPO and SAC. This file lists the exact commands that reproduce the reported results.

Two environment versions, both in `cleanrl/my_ant_env.py`:

| Env id | Difference |
|---|---|
| `AntSpin-v0` | Spin reward + speed-based drift penalty; ends when tipped past 60° or torso z ∉ [0.2, 1.5] m. True metric: upright rotations per episode. |
| `AntSpin-v1` | v0 + episode ends when the torso is > 1.5 m from its start; (dx, dy) offset from start appended to the observation (27 → 29 dims); true metric counts only upright rotations within 1.5 m. |

## Files

| File | Purpose |
|---|---|
| `cleanrl/my_ant_env.py` | `AntSpinEnv` (subclass of Gymnasium `Ant-v4`) and its registration as `AntSpin-v0` / `AntSpin-v1` |
| `cleanrl/ppo_continuous_action.py`, `cleanrl/sac_continuous_action.py` | CleanRL scripts; 5 added lines each (import + true-metric logging), see below |
| `cleanrl/plot_antspin.py` | Learning-curve plots: per-seed lines, mean, 95% Student-t interval |
| `antspin_sweep.sh` | Launches all 10 training runs |
| `cleanrl_changes.diff` | Diff of the two CleanRL scripts against upstream |

## Setup

Python 3.8–3.10, Linux; tested with gymnasium 0.29.1 and mujoco 2.3.3 (pinned by `uv.lock`).

```bash
git clone <this repo> cleanrl-ant-spin && cd cleanrl-ant-spin
pip install uv            # if uv is not installed
uv sync --extra mujoco    # creates .venv with the pinned dependencies
```

## Reproduce the experiments

All 10 runs (PPO 3M steps, SAC 1M steps, paired seeds 1–5), in parallel in a tmux session:

```bash
./antspin_sweep.sh antspin_sweep      # outputs in ./antspin_sweep/{runs,videos,logs,diag}
```

For AntSpin-v1, set the env id (and a separate tmux session):

```bash
ENV_ID=AntSpin-v1 SESSION=sweep_v2 ./antspin_sweep.sh antspin_sweep_v2
```

Equivalently, each run on its own (seed `S` in 1..5, `ENV` = `AntSpin-v0` or `AntSpin-v1`), from the repo root:

```bash
uv run python cleanrl/ppo_continuous_action.py --env-id ENV --seed S --total-timesteps 3000000 --capture-video
uv run python cleanrl/sac_continuous_action.py --env-id ENV --seed S --total-timesteps 1000000 --capture-video
```

All other hyperparameters are CleanRL's defaults. `ANTSPIN_DIAG_LOG=<file.csv>` (set by the sweep script) additionally
records why each episode ended; it does not change training.

Wall clock on one RTX 2080 Ti per SAC run and 64 CPU cores: PPO ~190–230 steps/s (~4 h), SAC ~41–46 steps/s (~6.5 h),
with video capture on. SAC runs on GPU nondeterministically, so curves match the reported ones statistically, not bit-for-bit.

## Reproduce the plots and final numbers

```bash
uv run --with matplotlib python cleanrl/plot_antspin.py --runs-dir antspin_sweep/runs --out-dir figures
uv run --with matplotlib python cleanrl/plot_antspin.py --env-id AntSpin-v1 --runs-dir antspin_sweep_v2/runs \
    --out-dir figures_v1 --metric-name "upright rotations within 1.5 m per episode"
```

Writes `figures/{episodic_return,true_metric_rotations}.{png,pdf,csv}` and prints final performance
(mean over each seed's last 100k env steps, then mean ± 95% t-interval across seeds).

Aggregation: per seed, episodes are averaged in non-overlapping 25k-step bins keyed by the env step at which the
episode ended (the only smoothing); then per bin, mean ± t(0.975, 4)·s/√5 across the 5 seeds. x-axis = environment steps.

## Changes to CleanRL files

Both training scripts get the same 5 lines and nothing else:

```python
import my_ant_env  # noqa: F401  registers AntSpin-v0
...
if "true_metric_rotations" in info:
    writer.add_scalar("charts/true_metric_rotations", info["true_metric_rotations"], global_step)
    writer.add_scalar("charts/final_distance_from_start", info["distance_from_origin"], global_step)
```

Full diff against upstream CleanRL fe8d8a0: `cleanrl_changes.diff`. The MuJoCo XML is unchanged.
