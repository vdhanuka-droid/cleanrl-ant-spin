"""Learning-curve plots for the AntSpin-v0 PPO vs SAC sweep.

Aggregation (as stated in the report):
  1. Per seed, episodes are grouped into non-overlapping bins of --bin env steps, keyed by the global step at which
     the episode ended, and averaged within each bin. This puts every seed on a common step grid and is the only
     smoothing applied (per seed, before aggregating).
  2. Per bin, across seeds: mean and Student-t 95% interval, mean +/- t_{0.975, n-1} * s / sqrt(n)
     (2.776 for n = 5). Individual seed curves are overlaid as thin lines.

Usage:
  uv run python cleanrl/plot_antspin.py --runs-dir /nobackup/vdhanuka/antspin/sweep/runs --out-dir figures
"""
import argparse
import csv
import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ALGOS = {  # label, color (validated categorical slots 1 and 2: CVD and contrast checks pass)
    "ppo": ("PPO", "#2a78d6"),
    "sac": ("SAC", "#eb6834"),
}
METRICS = {
    "charts/episodic_return": ("episodic_return", "Mean episodic return", "Episodic return"),
    "charts/true_metric_rotations": (
        "true_metric_rotations",
        "Mean true performance: upright rotations per episode",
        "Upright rotations per episode",
    ),
}
# Student-t 0.975 quantiles by degrees of freedom (n - 1), so the 95% interval needs no scipy.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
TEXT_PRIMARY, TEXT_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_runs(runs_dir, env_id):
    """Return {algo: {seed: {tag: (steps, values)}}}."""
    runs = {}
    for path in sorted(glob.glob(os.path.join(runs_dir, f"{env_id}__*"))):
        m = re.match(rf"{re.escape(env_id)}__(\w+?)_continuous_action__(\d+)__\d+$", os.path.basename(path))
        if not m or m.group(1) not in ALGOS:
            continue
        algo, seed = m.group(1), int(m.group(2))
        acc = EventAccumulator(path, size_guidance={"scalars": 0})
        acc.Reload()
        runs.setdefault(algo, {})[seed] = {
            tag: (np.array([e.step for e in acc.Scalars(tag)]), np.array([e.value for e in acc.Scalars(tag)]))
            for tag in METRICS
        }
    return runs


def bin_seed(steps, values, edges):
    """Mean of the episodes ending in each bin; empty bins are filled by linear interpolation."""
    idx = np.digitize(steps, edges) - 1
    out = np.full(len(edges) - 1, np.nan)
    for b in range(len(out)):
        sel = idx == b
        if sel.any():
            out[b] = values[sel].mean()
    ok = ~np.isnan(out)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return np.interp(centers, centers[ok], out[ok])


def aggregate(seed_data, tag, bin_size):
    """Common grid up to the shortest seed's last full bin; returns centers, per-seed matrix, mean, half-width."""
    last = min(d[tag][0].max() for d in seed_data.values())
    edges = np.arange(0, last - last % bin_size + 1, bin_size, dtype=float)
    seeds = sorted(seed_data)
    curves = np.stack([bin_seed(*seed_data[s][tag], edges) for s in seeds])
    n = len(seeds)
    mean = curves.mean(0)
    half = T975[n - 1] * curves.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    return 0.5 * (edges[:-1] + edges[1:]), seeds, curves, mean, half


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:g}M" if x else "0"))


def plot_metric(runs, tag, bin_size, out_dir):
    key, title, ylabel = METRICS[tag]
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=200)
    rows = []
    for algo, (label, color) in ALGOS.items():
        if algo not in runs:
            continue
        x, seeds, curves, mean, half = aggregate(runs[algo], tag, bin_size)
        for c in curves:  # per-seed curves behind the mean
            ax.plot(x, c, color=color, linewidth=0.8, alpha=0.35, zorder=1)
        ax.fill_between(x, mean - half, mean + half, color=color, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(x, mean, color=color, linewidth=2.0, zorder=3, label=f"{label} (mean of {len(seeds)} seeds, 95% t-CI)")
        ax.annotate(label, (x[-1], mean[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9, fontweight="bold", color=TEXT_PRIMARY)
        for i, xc in enumerate(x):
            rows.append([algo, int(xc), mean[i], mean[i] - half[i], mean[i] + half[i]]
                        + [f"seed_{s}={v:.4f}" for s, v in zip(seeds, curves[:, i])])
    style_axes(ax)
    ax.set_title(title, loc="left", fontsize=11, color=TEXT_PRIMARY, pad=10)
    ax.set_xlabel("Environment steps", fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_SECONDARY)
    ax.margins(x=0.01)
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=TEXT_SECONDARY)
    fig.text(0.01, 0.01, f"Thin lines: individual seeds. Each seed averaged in {bin_size // 1000}k-step bins before "
             "aggregating; band = mean ± t(0.975, n−1)·s/√n.", fontsize=7, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.03, 0.97, 1))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{key}.{ext}"))
    plt.close(fig)
    with open(os.path.join(out_dir, f"{key}.csv"), "w", newline="") as f:  # table view of the plotted data
        w = csv.writer(f)
        w.writerow(["algo", "step_bin_center", "mean", "ci_low", "ci_high", "per_seed..."])
        w.writerows(rows)


def final_table(runs, window):
    """Per-seed mean over each run's last `window` env steps, then mean +/- t-CI across seeds."""
    print(f"\nFinal performance: mean over each seed's last {window // 1000}k env steps")
    for tag, (key, _, _) in METRICS.items():
        for algo, (label, _) in ALGOS.items():
            if algo not in runs:
                continue
            vals = []
            for s in sorted(runs[algo]):
                steps, v = runs[algo][s][tag]
                vals.append(v[steps > steps.max() - window].mean())
            vals = np.array(vals)
            half = T975[len(vals) - 1] * vals.std(ddof=1) / np.sqrt(len(vals))
            seeds = ", ".join(f"{v:.1f}" for v in vals)
            print(f"  {key:22s} {label}: {vals.mean():8.2f} ± {half:6.2f}   per seed [{seeds}]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out-dir", default="figures")
    p.add_argument("--env-id", default="AntSpin-v0")
    p.add_argument("--bin", type=int, default=25_000, help="env steps per bin (the only smoothing)")
    p.add_argument("--final-window", type=int, default=100_000)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    runs = load_runs(args.runs_dir, args.env_id)
    for algo in runs:
        print(f"{ALGOS[algo][0]}: seeds {sorted(runs[algo])}")
    for tag in METRICS:
        plot_metric(runs, tag, args.bin, args.out_dir)
    final_table(runs, args.final_window)
    print(f"\nWrote {', '.join(sorted(os.listdir(args.out_dir)))} to {args.out_dir}")


if __name__ == "__main__":
    main()
