#!/usr/bin/env python
"""Plot NPE-PFN comparison with baselines from rebuttal metrics."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load NPE-PFN results
npe_pfn_dir = Path("results/cluster_results")
with open(npe_pfn_dir / "batched_results_final.json") as f:
    npe_pfn_pendulum = json.load(f)
with open(npe_pfn_dir / "results_wind_tunnel_batched_partial.json") as f:
    npe_pfn_wind = json.load(f)

# Load baseline metrics
with open("/tmp/rebuttal/metrics/metrics.json") as f:
    baseline_metrics = json.load(f)

# Load fm_post_transform from metrics(1).json (has 5 seeds)
with open("metrics(1).json") as f:
    fm_metrics = json.load(f)

# Override fm_post_transform with data from metrics(1).json
for task in ["pendulum", "wind_tunnel"]:
    if task in fm_metrics and "methods" in fm_metrics[task]:
        if "fm_post_transform" in fm_metrics[task]["methods"]:
            if task not in baseline_metrics:
                baseline_metrics[task] = {"methods": {}, "baselines": {}}
            if "methods" not in baseline_metrics[task]:
                baseline_metrics[task]["methods"] = {}
            baseline_metrics[task]["methods"]["fm_post_transform"] = fm_metrics[task]["methods"]["fm_post_transform"]


def extract_npe_pfn_metrics(task_results):
    """Extract NPE-PFN metrics, computing mean and std across seeds."""
    num_cals = sorted([int(k) for k in task_results.keys()])
    metrics = {
        "num_cal": num_cals,
        "c2st_mean": [], "c2st_std": [],
        "wasserstein_mean": [], "wasserstein_std": [],
        "mmd_mean": [], "mmd_std": [],
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


def extract_baseline_metrics(method_results):
    """Extract baseline metrics, computing mean and std across seeds."""
    num_cals = sorted([int(k) for k in method_results.keys()])
    metrics = {
        "num_cal": num_cals,
        "c2st_mean": [], "c2st_std": [],
        "wasserstein_mean": [], "wasserstein_std": [],
        "mmd_mean": [], "mmd_std": [],
    }
    for num_cal in num_cals:
        c2st_vals = method_results[str(num_cal)]["joint_c2st"]
        wass_vals = method_results[str(num_cal)]["wasserstein"]
        mmd_vals = method_results[str(num_cal)]["mmd"]
        metrics["c2st_mean"].append(np.mean(c2st_vals))
        metrics["c2st_std"].append(np.std(c2st_vals))
        metrics["wasserstein_mean"].append(np.mean(wass_vals))
        metrics["wasserstein_std"].append(np.std(wass_vals))
        metrics["mmd_mean"].append(np.mean(mmd_vals))
        metrics["mmd_std"].append(np.std(mmd_vals))
    return metrics


# Method display names and colors
METHOD_NAMES = {
    "npe_pfn": "NPE-PFN (Ours)",
    "fm_post_transform": "FMCPE",
    "dpe": "DPE",
    "mf_npe": "MF-NPE",
}

COLORS = {
    "npe_pfn": "tab:red",
    "fm_post_transform": "tab:blue",
    "dpe": "tab:orange",
    "mf_npe": "tab:purple",
}

MARKERS = {
    "npe_pfn": "o",
    "fm_post_transform": "s",
    "dpe": "D",
    "mf_npe": "v",
}

# Only include these methods
SELECTED_METHODS = {"fm_post_transform", "dpe", "mf_npe"}


def plot_task(ax, task_name, npe_pfn_results, baseline_data, title, metric="c2st", ylabel="Joint C2ST", ylim=None, log_y=False):
    """Plot metric comparison for a single task."""
    mean_key = f"{metric}_mean"
    std_key = f"{metric}_std"

    # Plot NPE-PFN
    npe_metrics = extract_npe_pfn_metrics(npe_pfn_results)
    ax.errorbar(
        npe_metrics["num_cal"],
        npe_metrics[mean_key],
        yerr=npe_metrics[std_key],
        label=METHOD_NAMES["npe_pfn"],
        marker=MARKERS["npe_pfn"],
        color=COLORS["npe_pfn"],
        capsize=3,
        linewidth=2,
        markersize=8,
        zorder=10,
    )

    # Plot methods
    if "methods" in baseline_data and baseline_data["methods"]:
        for method, method_results in baseline_data["methods"].items():
            if method in METHOD_NAMES and method in SELECTED_METHODS:
                metrics = extract_baseline_metrics(method_results)
                ax.errorbar(
                    metrics["num_cal"],
                    metrics[mean_key],
                    yerr=metrics[std_key],
                    label=METHOD_NAMES.get(method, method),
                    marker=MARKERS.get(method, "x"),
                    color=COLORS.get(method, None),
                    capsize=3,
                    linewidth=1.5,
                    markersize=6,
                    alpha=0.8,
                )

    # Plot baselines
    if "baselines" in baseline_data and baseline_data["baselines"]:
        for method, method_results in baseline_data["baselines"].items():
            if method in METHOD_NAMES and method in SELECTED_METHODS:
                metrics = extract_baseline_metrics(method_results)
                ax.errorbar(
                    metrics["num_cal"],
                    metrics[mean_key],
                    yerr=metrics[std_key],
                    label=METHOD_NAMES.get(method, method),
                    marker=MARKERS.get(method, "x"),
                    color=COLORS.get(method, None),
                    capsize=3,
                    linewidth=1.5,
                    markersize=6,
                    linestyle="--",
                    alpha=0.8,
                )

    # Formatting
    if metric == "c2st":
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.7, label="Optimal (0.5)")
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Number of Calibration Samples", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xticks([10, 50, 200, 1000])
    ax.set_xticklabels(["10", "50", "200", "1000"])


# Create Pendulum plots (3 metrics)
fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4.5))
pendulum_baseline = baseline_metrics.get("pendulum", {"methods": {}, "baselines": {}})

plot_task(axes1[0], "pendulum", npe_pfn_pendulum["pendulum"], pendulum_baseline,
          "Pendulum - Joint C2ST", metric="c2st", ylabel="Joint C2ST", ylim=(0.4, 1.05))
plot_task(axes1[1], "pendulum", npe_pfn_pendulum["pendulum"], pendulum_baseline,
          "Pendulum - Wasserstein", metric="wasserstein", ylabel="Wasserstein Distance")
plot_task(axes1[2], "pendulum", npe_pfn_pendulum["pendulum"], pendulum_baseline,
          "Pendulum - MMD", metric="mmd", ylabel="MMD", log_y=True)

plt.tight_layout()
fig1.savefig(npe_pfn_dir / "comparison_pendulum.pdf", bbox_inches="tight")
print(f"Saved: {npe_pfn_dir / 'comparison_pendulum.pdf'}")

# Create Wind Tunnel plots (3 metrics)
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5))
wind_baseline = baseline_metrics.get("wind_tunnel", {"methods": {}, "baselines": {}})

plot_task(axes2[0], "wind_tunnel", npe_pfn_wind["wind_tunnel"], wind_baseline,
          "Wind Tunnel - Joint C2ST", metric="c2st", ylabel="Joint C2ST", ylim=(0.4, 1.05))
plot_task(axes2[1], "wind_tunnel", npe_pfn_wind["wind_tunnel"], wind_baseline,
          "Wind Tunnel - Wasserstein", metric="wasserstein", ylabel="Wasserstein Distance")
plot_task(axes2[2], "wind_tunnel", npe_pfn_wind["wind_tunnel"], wind_baseline,
          "Wind Tunnel - MMD", metric="mmd", ylabel="MMD", log_y=True)

plt.tight_layout()
fig2.savefig(npe_pfn_dir / "comparison_wind_tunnel.pdf", bbox_inches="tight")
print(f"Saved: {npe_pfn_dir / 'comparison_wind_tunnel.pdf'}")

# Print summary
print("\n" + "=" * 60)
print("NPE-PFN vs Baselines Summary (Joint C2ST)")
print("=" * 60)

for task, npe_data, baseline_data in [
    ("Pendulum", npe_pfn_pendulum["pendulum"], baseline_metrics.get("pendulum", {})),
    ("Wind Tunnel", npe_pfn_wind["wind_tunnel"], baseline_metrics.get("wind_tunnel", {})),
]:
    print(f"\n{task}:")
    npe_metrics = extract_npe_pfn_metrics(npe_data)
    for i, num_cal in enumerate(npe_metrics["num_cal"]):
        print(f"  num_cal={num_cal}: NPE-PFN = {npe_metrics['c2st_mean'][i]:.3f} +/- {npe_metrics['c2st_std'][i]:.3f}")
