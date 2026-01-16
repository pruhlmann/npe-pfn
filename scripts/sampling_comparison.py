#!/usr/bin/env python3
"""
Sampling Time Comparison: N samples for 1 obs vs 1 sample for N obs

This script compares the sampling time for:
1. Strategy A: Sample N points for 1 observation
2. Strategy B: Sample 1 point for each of N observations

We use a simple Gaussian prior with a linear model.
"""

import sys
import time
import argparse
from pathlib import Path

import torch
import numpy as np
from pyro import distributions as dist

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from npe_pfn import TabPFN_Based_NPE_PFN


def create_linear_simulator(theta_dim, obs_dim, noise_std=0.1, seed=42):
    """Create a linear simulator: y = A @ theta + b + noise"""
    torch.manual_seed(seed)
    A = torch.randn(obs_dim, theta_dim)
    b = torch.randn(obs_dim)

    def simulator(theta):
        y = theta @ A.T + b + noise_std * torch.randn(theta.shape[0], obs_dim)
        return y

    return simulator


def test_strategy_A(posterior, y_test_single, N, device):
    """Strategy A: Sample N points for 1 observation"""
    # Warm-up
    _ = posterior.sample(sample_shape=torch.Size([1]), x=y_test_single)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Time sampling N points for 1 observation
    start_time = time.time()
    samples_A = posterior.sample(sample_shape=torch.Size([N]), x=y_test_single)

    if device.type == "cuda":
        torch.cuda.synchronize()

    time_A = time.time() - start_time

    return samples_A, time_A


def test_strategy_B(posterior, y_test_multiple, device):
    """Strategy B: Sample 1 point for N observations"""
    N = len(y_test_multiple)

    # Warm-up
    _ = posterior.sample(sample_shape=torch.Size([1]), x=y_test_multiple[0:1])

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Time sampling 1 point for each of N observations
    start_time = time.time()
    samples_B = []
    for y_single in y_test_multiple:
        sample = posterior.sample(sample_shape=torch.Size([1]), x=y_single.unsqueeze(0))
        samples_B.append(sample.squeeze())
    samples_B = torch.stack(samples_B)

    if device.type == "cuda":
        torch.cuda.synchronize()

    time_B = time.time() - start_time

    return samples_B, time_B


def test_scaling(posterior, prior, simulator, device, N_values):
    """Test different values of N to see scaling"""
    times_A_list = []
    times_B_list = []

    print("\n" + "="*60)
    print("SCALING ANALYSIS")
    print("="*60)

    # Generate single observation for Strategy A
    theta_test_single = prior.sample((1,))
    y_test_single = simulator(theta_test_single)  # Keep on CPU

    for n in N_values:
        print(f"\nTesting N={n}...")

        # Strategy A
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        _ = posterior.sample(sample_shape=torch.Size([n]), x=y_test_single)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_a = time.time() - start
        times_A_list.append(t_a)

        # Strategy B
        y_test_n = simulator(prior.sample((n,)))  # Keep on CPU
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for y_single in y_test_n:
            _ = posterior.sample(sample_shape=torch.Size([1]), x=y_single.unsqueeze(0))
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_b = time.time() - start
        times_B_list.append(t_b)

        ratio = t_b / t_a if t_a > 0 else 0
        print(f"  Strategy A: {t_a:.4f}s")
        print(f"  Strategy B: {t_b:.4f}s")
        print(f"  Ratio (B/A): {ratio:.2f}x")

    return times_A_list, times_B_list


def main():
    parser = argparse.ArgumentParser(
        description='Compare sampling strategies for NPE-PFN'
    )
    parser.add_argument(
        '--theta_dim',
        type=int,
        default=5,
        help='Dimension of theta (default: 5)'
    )
    parser.add_argument(
        '--obs_dim',
        type=int,
        default=10,
        help='Dimension of observations (default: 10)'
    )
    parser.add_argument(
        '--num_calibration',
        type=int,
        default=100,
        help='Number of calibration samples (default: 100)'
    )
    parser.add_argument(
        '--N',
        type=int,
        default=100,
        help='Number of samples/observations for main comparison (default: 100)'
    )
    parser.add_argument(
        '--gpu_num',
        type=int,
        default=None,
        help='GPU number to use (default: auto-detect)'
    )
    parser.add_argument(
        '--scaling',
        action='store_true',
        help='Run scaling analysis with multiple N values'
    )

    args = parser.parse_args()

    # Setup device
    if args.gpu_num is not None:
        if torch.cuda.is_available() and args.gpu_num < torch.cuda.device_count():
            device = torch.device(f'cuda:{args.gpu_num}')
        else:
            print(f"Warning: GPU {args.gpu_num} not available, using CPU")
            device = torch.device('cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print("="*60)
    print("SAMPLING TIME COMPARISON")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device}")
    print(f"Theta dim: {args.theta_dim}")
    print(f"Obs dim: {args.obs_dim}")
    print(f"Calibration samples: {args.num_calibration}")
    print(f"N (samples/obs): {args.N}")
    print("="*60)

    # Define prior: Gaussian
    prior_loc = torch.zeros(args.theta_dim)
    prior_cov = torch.eye(args.theta_dim)
    prior = dist.MultivariateNormal(loc=prior_loc, covariance_matrix=prior_cov)

    # Define linear model
    simulator = create_linear_simulator(args.theta_dim, args.obs_dim)

    # Generate calibration data
    print("\nGenerating calibration data...")
    theta_cal = prior.sample((args.num_calibration,))
    y_cal = simulator(theta_cal)

    # Keep calibration data on CPU (TabPFN requires CPU tensors)
    print(f"Calibration data: theta_cal {theta_cal.shape}, y_cal {y_cal.shape}")

    # Initialize NPE-PFN
    print("\nInitializing NPE-PFN...")
    posterior = TabPFN_Based_NPE_PFN(prior=prior)
    posterior.append_simulations(theta_cal, y_cal)
    print("NPE-PFN initialized and calibration data appended")

    # Strategy A: Sample N points for 1 observation
    print("\n" + "="*60)
    print(f"STRATEGY A: Sample {args.N} points for 1 observation")
    print("="*60)

    theta_test_single = prior.sample((1,))
    y_test_single = simulator(theta_test_single)  # Keep on CPU

    samples_A, time_A = test_strategy_A(posterior, y_test_single, args.N, device)

    print(f"Time taken: {time_A:.4f} seconds")
    print(f"Samples shape: {samples_A.shape}")
    print(f"Time per sample: {time_A / args.N * 1000:.2f} ms")

    # Strategy B: Sample 1 point for N observations
    print("\n" + "="*60)
    print(f"STRATEGY B: Sample 1 point for each of {args.N} observations")
    print("="*60)

    theta_test_multiple = prior.sample((args.N,))
    y_test_multiple = simulator(theta_test_multiple)  # Keep on CPU

    samples_B, time_B = test_strategy_B(posterior, y_test_multiple, device)

    print(f"Time taken: {time_B:.4f} seconds")
    print(f"Samples shape: {samples_B.shape}")
    print(f"Time per observation: {time_B / args.N * 1000:.2f} ms")

    # Comparison
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"\nStrategy A: {args.N} samples for 1 observation")
    print(f"  Total time: {time_A:.4f} seconds")
    print(f"  Time per sample: {time_A / args.N * 1000:.2f} ms")

    print(f"\nStrategy B: 1 sample for {args.N} observations")
    print(f"  Total time: {time_B:.4f} seconds")
    print(f"  Time per observation: {time_B / args.N * 1000:.2f} ms")

    speedup = time_B / time_A if time_A > 0 else 0
    print(f"\nSpeedup factor: {speedup:.2f}x")
    if time_A < time_B:
        print(f"✓ Strategy A is {speedup:.2f}x FASTER")
    else:
        print(f"✓ Strategy B is {1/speedup:.2f}x FASTER")

    print("="*60)

    # Scaling analysis
    if args.scaling:
        N_values = [10, 20, 50, 100, 200]
        times_A_list, times_B_list = test_scaling(
            posterior, prior, simulator, device, N_values
        )

        print("\n" + "="*60)
        print("SCALING SUMMARY")
        print("="*60)
        print(f"{'N':>5} | {'Strategy A':>12} | {'Strategy B':>12} | {'Ratio (B/A)':>12}")
        print("-"*60)
        for n, t_a, t_b in zip(N_values, times_A_list, times_B_list):
            ratio = t_b / t_a if t_a > 0 else 0
            print(f"{n:>5} | {t_a:>10.4f}s | {t_b:>10.4f}s | {ratio:>10.2f}x")
        print("="*60)


if __name__ == '__main__':
    main()
