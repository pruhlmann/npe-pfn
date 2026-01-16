#!/usr/bin/env python3
"""
Test script to verify that priors are correctly defined for transformed data.
"""

import sys
from pathlib import Path

import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from evaluate_ropefm import create_prior


def test_prior(task_name: str, expected_dim: int):
    """Test that prior can be created and sampled from."""
    print(f"\n{'='*60}")
    print(f"Testing prior for task: {task_name}")
    print(f"{'='*60}")

    # Create prior
    prior = create_prior(task_name)
    print(f"Prior created: {prior}")

    # Sample from prior
    num_samples = 10
    samples = prior.sample((num_samples,))
    print(f"Sample shape: {samples.shape}")
    print(f"Expected shape: ({num_samples}, {expected_dim})")

    # Verify shape
    assert samples.shape == (num_samples, expected_dim), \
        f"Shape mismatch: got {samples.shape}, expected ({num_samples}, {expected_dim})"

    # Print sample statistics
    print(f"Sample mean: {samples.mean(dim=0)}")
    print(f"Sample std: {samples.std(dim=0)}")
    print(f"Sample min: {samples.min(dim=0)[0]}")
    print(f"Sample max: {samples.max(dim=0)[0]}")

    # Compute log prob
    log_prob = prior.log_prob(samples)
    print(f"Log prob shape: {log_prob.shape}")
    print(f"Log prob mean: {log_prob.mean()}")

    print(f"✓ Test passed for {task_name}")


if __name__ == "__main__":
    # Test each task
    test_prior("pendulum", expected_dim=2)
    test_prior("high_dim_gaussian", expected_dim=3)
    test_prior("wind_tunnel", expected_dim=1)

    print(f"\n{'='*60}")
    print("All tests passed!")
    print(f"{'='*60}\n")
