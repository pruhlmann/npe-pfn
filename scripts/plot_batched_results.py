#!/usr/bin/env python
"""Plot NPE-PFN batched evaluation results."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
results_dir = Path("results/cluster_results")

# Load pendulum results
with open(results_dir / "batched_results_final.json") as f:
    pendulum_data = json.load(f)

# Load wind_tunnel results
with open(results_dir / "results_wind_tunnel_batched_partial.json") as f:
    wind_tunnel_data = json.load(f)

# Combine results
results = {**pendulum_data, **wind_tunnel_data}


def extract_metrics(task_results):
    """Extract metrics from task results, computing mean and std across seeds."""
    num_cals = sorted([int(k) for k in task_results.keys()])

    metrics = {
        "num_cal": num_cals,
        "c2st_mean": [],
        "c2st_std": [],
        "wasserstein_mean": [],
        "wasserstein_std": [],
        "mmd_mean": [],
        "mmd_std": [],
    }

    for num_cal in num_cals:
        seed_results = task_results[str(num_cal)]
        c2st_vals = [v["joint_c2st"] for v in seed_results.values()]
        wass_vals = [v["wasserstein"] for v in seed_results.values()]
        mmd_vals = [v["mmd"] for v in seed_results.values()]

        metrics["c2st_mean"].append(np.mean(c2st_vals))
        metrics["c2st_std"].append(np.std(c2st_vals))
        metrics["wasserstein_mean"].append(np.mean(wass_vals))
        metrics["wasserstein_std"].append(np.std(wass_vals))
        metrics["mmd_mean"].append(np.mean(mmd_vals))
        metrics["mmd_std"].append(np.std(mmd_vals))

    return metrics


# Extract metrics for each task
task_metrics = {}
for task in results:
    task_metrics[task] = extract_metrics(results[task])

# Create plots
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

colors = {"pendulum": "tab:blue", "wind_tunnel": "tab:orange", "light_tunnel": "tab:green"}
markers = {"pendulum": "o", "wind_tunnel": "s", "light_tunnel": "^"}

# Plot C2ST
ax = axes[0]
for task, metrics in task_metrics.items():
    ax.errorbar(
        metrics["num_cal"],
        metrics["c2st_mean"],
        yerr=metrics["c2st_std"],
        label=task.replace("_", " ").title(),
        marker=markers.get(task, "o"),
        color=colors.get(task, None),
        capsize=3,
        linewidth=2,
        markersize=8,
    )
ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="Random (0.5)")
ax.set_xscale("log")
ax.set_xlabel("Number of Calibration Samples")
ax.set_ylabel("Joint C2ST")
ax.set_title("Joint C2ST vs Calibration Size")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0.4, 1.0)

# Plot Wasserstein
ax = axes[1]
for task, metrics in task_metrics.items():
    ax.errorbar(
        metrics["num_cal"],
        metrics["wasserstein_mean"],
        yerr=metrics["wasserstein_std"],
        label=task.replace("_", " ").title(),
        marker=markers.get(task, "o"),
        color=colors.get(task, None),
        capsize=3,
        linewidth=2,
        markersize=8,
    )
ax.set_xscale("log")
ax.set_xlabel("Number of Calibration Samples")
ax.set_ylabel("Wasserstein Distance")
ax.set_title("Wasserstein Distance vs Calibration Size")
ax.legend()
ax.grid(True, alpha=0.3)

# Plot MMD
ax = axes[2]
for task, metrics in task_metrics.items():
    ax.errorbar(
        metrics["num_cal"],
        metrics["mmd_mean"],
        yerr=metrics["mmd_std"],
        label=task.replace("_", " ").title(),
        marker=markers.get(task, "o"),
        color=colors.get(task, None),
        capsize=3,
        linewidth=2,
        markersize=8,
    )
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Number of Calibration Samples")
ax.set_ylabel("MMD")
ax.set_title("MMD vs Calibration Size")
ax.legend()
ax.grid(True, alpha=0.3)

plt.suptitle("NPE-PFN Batched Evaluation on RoPEFM Benchmarks", fontsize=14, fontweight="bold")
plt.tight_layout()

# Save figure as PDF
output_path = results_dir / "batched_eval_results.pdf"
plt.savefig(output_path, bbox_inches="tight")
print(f"Saved plot to {output_path}")

# Print summary table
print("\n" + "=" * 80)
print("Summary Table: Mean (Std) across 5 seeds")
print("=" * 80)
for task, metrics in task_metrics.items():
    print(f"\n{task.upper()}")
    print("-" * 60)
    print(f"{'Num Cal':>10} | {'C2ST':>15} | {'Wasserstein':>15} | {'MMD':>15}")
    print("-" * 60)
    for i, num_cal in enumerate(metrics["num_cal"]):
        c2st = f"{metrics['c2st_mean'][i]:.3f} ({metrics['c2st_std'][i]:.3f})"
        wass = f"{metrics['wasserstein_mean'][i]:.3f} ({metrics['wasserstein_std'][i]:.3f})"
        mmd = f"{metrics['mmd_mean'][i]:.4f} ({metrics['mmd_std'][i]:.4f})"
        print(f"{num_cal:>10} | {c2st:>15} | {wass:>15} | {mmd:>15}")
