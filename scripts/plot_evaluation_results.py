#!/usr/bin/env python3
"""
Plot NPE-PFN evaluation results as boxplots.

Creates a figure with 3 rows (one per metric) and 1 column.
Each row shows boxplots for different calibration set sizes.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Use a clean style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
})

def load_results(results_path: Path) -> dict:
    """Load results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def plot_boxplots(results: dict, task: str, save_path: Path):
    """
    Plot boxplots for evaluation metrics.

    Args:
        results: Dictionary with results from evaluation
        task: Task name (e.g., 'high_dim_gaussian')
        save_path: Path to save the figure
    """
    task_results = results[task]

    # Get calibration sizes (sorted numerically)
    calib_sizes = sorted([int(k) for k in task_results.keys()])

    # Metrics to plot (in order: C2ST, Wasserstein, MMD)
    metrics = [
        ('conditional_c2st', 'C2ST'),
        ('conditional_wasserstein', 'Wasserstein'),
        ('conditional_mmd', 'MMD')
    ]

    # Create figure with 3 rows, 1 column
    fig, axes = plt.subplots(3, 1, figsize=(6, 8), constrained_layout=True)

    for ax, (metric_key, metric_name) in zip(axes, metrics):
        # Collect data for each calibration size
        data_per_calib = []

        for calib_size in calib_sizes:
            calib_str = str(calib_size)
            # Get values across all seeds
            seed_values = []
            for seed, seed_data in task_results[calib_str].items():
                # Extract per-observation metric values
                per_obs_metrics = seed_data.get('per_observation_metrics', [])
                for obs_data in per_obs_metrics:
                    if metric_key in obs_data:
                        seed_values.append(obs_data[metric_key])

            data_per_calib.append(seed_values)

        # Create boxplot
        positions = np.arange(len(calib_sizes))
        bp = ax.boxplot(
            data_per_calib,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=True,
            boxprops=dict(linewidth=1.5),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            medianprops=dict(color='black', linewidth=2),
            flierprops=dict(marker='o', markersize=4, alpha=0.5)
        )

        # Color the boxes
        color = 'tab:blue'
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Set labels and title
        ax.set_ylabel(metric_name)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(c) for c in calib_sizes])

        # Add grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

    # Set common x-label on bottom plot
    axes[-1].set_xlabel('Calibration Set Size')

    # Add title
    task_display = task.replace('_', ' ').title()
    fig.suptitle(f'NPE-PFN Evaluation: {task_display}', fontsize=14, fontweight='bold')

    # Save figure
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to {save_path}")

    # Also save as PDF
    pdf_path = save_path.with_suffix('.pdf')
    fig.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved figure to {pdf_path}")

    plt.close(fig)


def main():
    # Paths
    results_dir = Path(__file__).parent.parent / 'results' / 'npe_pfn_evaluation'
    results_file = results_dir / 'npe_pfn_results_final.json'

    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return

    # Load results
    results = load_results(results_file)
    print(f"Loaded results for tasks: {list(results.keys())}")

    # Plot for each task
    for task in results.keys():
        save_path = results_dir / f'{task}_boxplots.png'
        plot_boxplots(results, task, save_path)


if __name__ == '__main__':
    main()
