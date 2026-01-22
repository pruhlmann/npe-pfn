#!/usr/bin/env python3
"""
NPE-PFN Batched Evaluation Script for RoPEFM Benchmarks

This script evaluates NPE-PFN using `sample_batched` for efficient joint metric
computation on pendulum, wind_tunnel, and light_tunnel tasks.

Key difference from evaluate_ropefm.py:
- Uses sample_batched() for true batched sampling (1 sample per observation)
- Computes JOINT metrics (not conditional metrics)
- Supports light_tunnel task with random projection embedding for images

IMPORTANT: Data Transformation
-------------------------------
The ropefm code applies LogitBoxTransform to theta samples before saving for tasks
with compact prior support (pendulum, wind_tunnel, light_tunnel). This transform
maps [a,b] → R using: theta_transformed = logit((theta - a) / (b - a))

Therefore, the priors defined in create_prior() must match the TRANSFORMED space.
"""

import argparse
import json
import logging
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Callable

import numpy as np
import ot  # POT library for optimal transport
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro import distributions as dist
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

# Add parent directory to path to import npe_pfn
sys.path.append(str(Path(__file__).parent.parent))
from npe_pfn import TabPFN_Based_NPE_PFN

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# RANDOM PROJECTION EMBEDDING FOR HIGH-DIMENSIONAL OBSERVATIONS
# ============================================================================

class RandomProjection(nn.Module):
    """Random projection for dimensionality reduction of high-dim observations.

    Projects high-dimensional inputs (e.g., flattened images) to a lower
    dimensional space using a fixed random matrix. Useful for light_tunnel
    task where observations are (3, 64, 64) = 12288 dimensional images.
    """

    def __init__(self, in_dim: int = 12288, out_dim: int = 128, seed: int = 42):
        super().__init__()
        # Generate reproducible random projection matrix
        generator = torch.Generator().manual_seed(seed)
        proj_matrix = torch.randn(in_dim, out_dim, generator=generator) / np.sqrt(out_dim)
        self.register_buffer('proj', proj_matrix)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input to lower dimensional space.

        Args:
            x: Input tensor of shape [batch, ...] where product of ... equals in_dim

        Returns:
            Projected tensor of shape [batch, out_dim]
        """
        x_flat = x.reshape(x.shape[0], -1)
        return x_flat @ self.proj


def prepare_observations(
    y: torch.Tensor,
    task_name: str,
    projection: Optional[RandomProjection] = None
) -> torch.Tensor:
    """Prepare observations for NPE-PFN, applying embedding if needed.

    Args:
        y: Raw observations [batch, ...]
        task_name: Name of the task
        projection: Optional RandomProjection for high-dim tasks

    Returns:
        Processed observations [batch, obs_dim]
    """
    if task_name == 'light_tunnel':
        if projection is None:
            raise ValueError("light_tunnel task requires a RandomProjection")
        # Apply random projection to flatten and reduce dimensions
        return projection(y)
    return y


# ============================================================================
# METRIC FUNCTIONS (copied from evaluate_ropefm.py)
# ============================================================================

class DefaultMLP(nn.Module):
    """Default MLP: hidden layers of size 8*dim each, ReLU, output 2 logits."""

    def __init__(self, input_dim: int, hidden_mult: int = 8):
        super().__init__()
        h = hidden_mult * input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, h // 2),
            nn.ReLU(),
            nn.Linear(h // 2, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, h // 2),
            nn.ReLU(),
            nn.Linear(h // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


class SmallMLP(nn.Module):
    """Smaller MLP: hidden layers of size 4*dim."""

    def __init__(self, input_dim: int, hidden_mult: int = 4):
        super().__init__()
        h = hidden_mult * input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, 2),
        )

    def forward(self, x):
        return self.net(x)


class LinearClassifier(nn.Module):
    """Simple linear classifier (no hidden layer)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.fc(x)


_MODEL_REGISTRY = {
    "mlp": DefaultMLP,
    "mlp_small": SmallMLP,
    "linear": LinearClassifier,
}


def classifier_two_samples_test_torch(
    X: torch.Tensor,
    Y: torch.Tensor,
    seed: int = 1,
    n_folds: int = 5,
    scoring: str = "accuracy",
    z_score: bool = True,
    noise_scale: Optional[float] = None,
    cv: str = "StratifiedKFold",
    model: Union[str, Callable] = "mlp",
    model_kwargs: Optional[Dict] = None,
    training_kwargs: Optional[Dict] = None,
) -> float:
    """PyTorch version of classifier two-sample test."""
    if model_kwargs is None:
        model_kwargs = {}
    if training_kwargs is None:
        training_kwargs = {}
    epochs = int(training_kwargs.get("epochs", 100))
    batch_size = int(training_kwargs.get("batch_size", 128))
    lr = float(training_kwargs.get("lr", 1e-3))
    weight_decay = float(training_kwargs.get("weight_decay", 0.0))
    verbose = bool(training_kwargs.get("verbose", False))
    device = training_kwargs.get("device", None)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    if z_score:
        X_mean = torch.mean(X, dim=0)
        X_std = torch.std(X, dim=0)
        X_std = torch.where(X_std == 0, torch.ones_like(X_std), X_std)
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std

    if noise_scale is not None:
        X = X + noise_scale * torch.randn_like(X)
        Y = Y + noise_scale * torch.randn_like(Y)

    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()

    ndim = X_np.shape[1]

    data = np.concatenate((X_np, Y_np), axis=0)
    labels = np.concatenate(
        (np.zeros(X_np.shape[0], dtype=int), np.ones(Y_np.shape[0], dtype=int)), axis=0
    )

    if cv == "KFold":
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    elif cv == "StratifiedKFold":
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    else:
        raise ValueError(f"Unknown cross-validation strategy: {cv}")

    fold_scores = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(data, labels)):
        X_train = torch.from_numpy(data[train_idx]).float()
        y_train = torch.from_numpy(labels[train_idx]).long()
        X_test = torch.from_numpy(data[test_idx]).float()
        y_test = torch.from_numpy(labels[test_idx]).long()

        train_ds = TensorDataset(X_train, y_train)
        test_ds = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        if isinstance(model, str):
            if model not in _MODEL_REGISTRY:
                raise ValueError(
                    f"Unknown model string '{model}'. Known: {list(_MODEL_REGISTRY.keys())}"
                )
            model_cls = _MODEL_REGISTRY[model]
            net = model_cls(ndim, **model_kwargs) if model != "linear" else model_cls(ndim)
        elif callable(model):
            try:
                net = model(ndim, **model_kwargs)
            except TypeError:
                net = model()
        else:
            raise ValueError("model must be a string key or a callable/class building an nn.Module")

        net = net.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        net.train()
        tbar = tqdm(range(epochs), desc=f"Fold {fold_idx} Training", disable=not verbose)
        for epoch in tbar:
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                logits = net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * xb.size(0)
            current_acc = float((logits.argmax(dim=1) == yb).float().mean())
            tbar.set_postfix(
                loss=f"{epoch_loss / len(train_ds):.4f}", acc=f"{100 * current_acc:.2f} %"
            )

        net.eval()
        all_preds = []
        all_true = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                logits = net(xb)
                preds = logits.argmax(dim=1).detach().cpu().numpy()
                all_preds.append(preds)
                all_true.append(yb.numpy())
        all_preds = np.concatenate(all_preds, axis=0)
        all_true = np.concatenate(all_true, axis=0)

        if scoring == "accuracy":
            acc = float((all_preds == all_true).mean())
            fold_scores.append(acc)
        else:
            raise ValueError(f"Unknown scoring: {scoring}")

    mean_score = float(np.mean(fold_scores))
    return mean_score


def MMD(x, y, kernel):
    """Empirical maximum mean discrepancy."""
    device = x.device
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    dxx = rx.t() + rx - 2.0 * xx
    dyy = ry.t() + ry - 2.0 * yy
    dxy = rx.t() + ry - 2.0 * zz

    XX, YY, XY = (
        torch.zeros(xx.shape).to(device),
        torch.zeros(xx.shape).to(device),
        torch.zeros(xx.shape).to(device),
    )

    if kernel == "multiscale":
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx) ** -1
            YY += a**2 * (a**2 + dyy) ** -1
            XY += a**2 * (a**2 + dxy) ** -1

    if kernel == "rbf":
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2.0 * XY)


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_calibration_data(
    task_name: str, data_path: str = "./data"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load calibration data for a task."""
    file_path = Path(data_path) / task_name / "calibrations.pt"

    logger.info(f"Loading calibration data from {file_path}")
    data = torch.load(file_path, weights_only=False)

    theta = data['theta']
    x = data['x']
    y = data['y']

    logger.info(f"Loaded calibration data: theta {theta.shape}, x {x.shape}, y {y.shape}")
    return theta, x, y


def load_test_data(
    task_name: str, data_path: str = "./data"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load test data for a task."""
    file_path = Path(data_path) / task_name / "test_data.pt"

    logger.info(f"Loading test data from {file_path}")
    data = torch.load(file_path, weights_only=False)

    theta_test = data['theta']
    x_test = data['x']
    y_test = data['y']

    logger.info(f"Loaded test data: theta {theta_test.shape}, x {x_test.shape}, y {y_test.shape}")
    return theta_test, x_test, y_test


def subsample_calibration(
    theta: torch.Tensor,
    y: torch.Tensor,
    num_samples: int,
    seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Subsample calibration data with a given seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    indices = torch.randperm(len(theta))[:num_samples]
    return theta[indices], y[indices]


# ============================================================================
# PRIOR CREATION
# ============================================================================

def create_prior(task_name: str) -> dist.Distribution:
    """
    Create prior distribution matching the EXACT transformed space from ropefm simulators.

    The ropefm simulators define priors and transforms:
    - pendulum: Uniform prior + LogitBoxTransform → Logistic(0,1)
    - wind_tunnel: Uniform prior + LogitBoxTransform → Logistic(0,1)
    - light_tunnel: Uniform([0,255]^3 × [-180,180]^2) + LogitBoxTransform → Logistic(0,1)

    Returns the prior in the transformed space.
    """
    if task_name == 'pendulum':
        # From ropefm/simulator/pendulum.py
        # Original: Uniform([0,3] × [0.5,10]) - 2D
        # Transformed: Logistic(0, 1) for each dimension
        return dist.Independent(dist.Logistic(torch.zeros(2), torch.ones(2)), 1)

    elif task_name == 'wind_tunnel':
        # From ropefm/simulator/wind_tunnel.py
        # Original: Uniform([0, 45]) - 1D
        # Transformed: Logistic(0, 1)
        return dist.Independent(dist.Logistic(torch.zeros(1), torch.ones(1)), 1)

    elif task_name == 'light_tunnel':
        # From ropefm/simulator/light_tunnel.py
        # Original: Uniform([0,255]^3 × [-180,180]^2) - 5D (RGB + 2 angles)
        # Transformed: Logistic(0, 1) for each dimension
        return dist.Independent(dist.Logistic(torch.zeros(5), torch.ones(5)), 1)

    elif task_name == 'high_dim_gaussian':
        # From ropefm/simulator/pure_gaussian.py
        # Uses IdentityTransform (no transformation)
        torch.manual_seed(0)
        theta_dim = 3
        prior_var_scale = 5.0

        prior_loc = torch.rand(theta_dim) * 10 - 5
        cov_theta_sqrt = torch.normal(0.0, prior_var_scale, (theta_dim, theta_dim))
        prior_cov = cov_theta_sqrt @ cov_theta_sqrt.T + torch.eye(theta_dim)

        return dist.MultivariateNormal(loc=prior_loc, covariance_matrix=prior_cov)

    else:
        raise ValueError(f"Unknown task: {task_name}")


# ============================================================================
# BATCHED POSTERIOR SAMPLING
# ============================================================================

def sample_posteriors_batched(
    posterior: TabPFN_Based_NPE_PFN,
    y_test: torch.Tensor,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Generate 1 posterior sample for each test observation using sample_batched.

    This is the key difference from evaluate_ropefm.py which uses a loop.
    sample_batched processes all observations in a single call.

    Args:
        posterior: TabPFN_Based_NPE_PFN instance
        y_test: [num_test, obs_dim] - observations (on CPU for TabPFN)
        device: torch.device - device object (unused, TabPFN requires CPU)

    Returns:
        theta_pred: [num_test, theta_dim]
    """
    logger.info(f"Using sample_batched for {len(y_test)} observations...")

    # sample_batched returns [num_obs, num_samples, theta_dim]
    # We want 1 sample per observation, so sample_shape = torch.Size([1])
    theta_pred = posterior.sample_batched(
        x=y_test,  # [num_test, obs_dim]
        sample_shape=torch.Size([1]),  # 1 sample per observation
        show_progress_bars=True,
    )

    # Squeeze the sample dimension: [num_test, 1, theta_dim] -> [num_test, theta_dim]
    theta_pred = theta_pred.squeeze(1)

    logger.info(f"Generated {theta_pred.shape[0]} posterior samples via sample_batched")
    return theta_pred


# ============================================================================
# JOINT METRIC COMPUTATION
# ============================================================================

def compute_joint_metrics(
    theta_true: torch.Tensor,
    theta_pred: torch.Tensor,
    y_obs: torch.Tensor,
    seed: int = 42,
    device: torch.device = torch.device('cpu')
) -> Dict[str, float]:
    """
    Compute joint C2ST, joint Wasserstein, joint MMD.

    Creates joint distribution [theta, y] and compares:
    - True joint: [theta_true, y_obs]
    - Pred joint: [theta_pred, y_obs]
    """
    theta_true_cpu = theta_true.detach().cpu()
    theta_pred_cpu = theta_pred.detach().cpu()
    y_obs_cpu = y_obs.detach().cpu()

    # Flatten y_obs if it's multi-dimensional (e.g., images)
    if y_obs_cpu.ndim > 2:
        y_obs_cpu = y_obs_cpu.reshape(y_obs_cpu.shape[0], -1)

    # Create joint: [theta, y]
    true_joint = torch.cat([theta_true_cpu, y_obs_cpu], dim=1)
    pred_joint = torch.cat([theta_pred_cpu, y_obs_cpu], dim=1)

    logger.info(f"Computing joint metrics: true_joint {true_joint.shape}, pred_joint {pred_joint.shape}")

    # Joint C2ST
    logger.info("Computing joint C2ST...")
    joint_c2st = classifier_two_samples_test_torch(
        true_joint, pred_joint,
        seed=seed, n_folds=5, z_score=True, model='mlp',
        training_kwargs={'epochs': 100, 'batch_size': 128, 'verbose': False, 'device': str(device)}
    )

    # Joint Wasserstein
    logger.info("Computing joint Wasserstein...")
    a = torch.ones(true_joint.shape[0]) / true_joint.shape[0]
    b = torch.ones(pred_joint.shape[0]) / pred_joint.shape[0]
    M = ot.dist(true_joint.numpy(), pred_joint.numpy())
    joint_wasserstein = float(np.sqrt(ot.emd2(a.numpy(), b.numpy(), M)))

    # Joint MMD
    logger.info("Computing joint MMD...")
    true_joint_device = true_joint.to(device)
    pred_joint_device = pred_joint.to(device)
    joint_mmd = MMD(true_joint_device, pred_joint_device, kernel='rbf').item()

    return {
        'joint_c2st': joint_c2st,
        'joint_wasserstein': joint_wasserstein,
        'joint_mmd': joint_mmd
    }


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def evaluate_task_batched(
    task_name: str,
    num_cal_list: list,
    seed_list: list,
    data_path: str,
    output_dir: Path,
    max_test_samples: int = None,
    projection_dim: int = 128,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """
    Evaluate NPE-PFN for a single task using sample_batched.

    Uses true batched sampling for joint metric computation.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting BATCHED evaluation for task: {task_name}")
    logger.info(f"{'='*80}\n")

    # Load data
    theta_cal, x_cal, y_cal = load_calibration_data(task_name, data_path)
    theta_test, x_test, y_test = load_test_data(task_name, data_path)

    # Setup random projection for high-dim tasks
    projection = None
    if task_name == 'light_tunnel':
        # y has shape [N, 3, 64, 64] = [N, 12288] flattened
        in_dim = int(np.prod(y_cal.shape[1:]))
        projection = RandomProjection(in_dim=in_dim, out_dim=projection_dim, seed=42)
        logger.info(f"Created RandomProjection: {in_dim} -> {projection_dim}")

        # Apply projection to observations
        y_cal = prepare_observations(y_cal, task_name, projection)
        y_test = prepare_observations(y_test, task_name, projection)
        logger.info(f"Projected observations: cal {y_cal.shape}, test {y_test.shape}")

    # Optionally limit test samples
    if max_test_samples is not None and max_test_samples < len(theta_test):
        logger.info(f"Limiting test samples to {max_test_samples} (from {len(theta_test)})")
        theta_test = theta_test[:max_test_samples]
        x_test = x_test[:max_test_samples]
        y_test = y_test[:max_test_samples]

    # Create prior
    prior = create_prior(task_name)
    logger.info(f"Created prior for {task_name}")

    # Initialize results
    task_results = {}

    # Evaluate for each calibration size
    for num_cal in num_cal_list:
        logger.info(f"\n--- Calibration size: {num_cal} ---")
        task_results[str(num_cal)] = {}

        # Evaluate for each seed
        for seed in seed_list:
            logger.info(f"\nSeed: {seed}")

            # Subsample calibration data
            theta_sub, y_sub = subsample_calibration(theta_cal, y_cal, num_cal, seed)
            logger.info(f"Subsampled calibration data: {theta_sub.shape}, {y_sub.shape}")

            # Initialize NPE-PFN
            posterior = TabPFN_Based_NPE_PFN(prior=prior)
            posterior.append_simulations(theta_sub, y_sub)
            logger.info("Appended simulations to posterior")

            # Sample from posterior using batched sampling
            logger.info(f"Sampling posteriors for {len(y_test)} test observations (batched)...")
            theta_pred = sample_posteriors_batched(posterior, y_test, device=device)
            logger.info(f"Generated posterior samples: {theta_pred.shape}")

            # For joint metrics, we need the original y_test (before projection for light_tunnel)
            # But for metric computation we use the embedded y
            # Reload original y_test for metric computation if needed
            if task_name == 'light_tunnel':
                # Reload original test data for joint metrics (we want original images)
                _, _, y_test_original = load_test_data(task_name, data_path)
                if max_test_samples is not None and max_test_samples < len(y_test_original):
                    y_test_original = y_test_original[:max_test_samples]
                y_for_metrics = y_test_original
            else:
                y_for_metrics = y_test

            # Compute metrics
            logger.info("Computing joint metrics...")
            metrics = compute_joint_metrics(theta_test, theta_pred, y_for_metrics, seed=seed, device=device)

            # Store results
            task_results[str(num_cal)][str(seed)] = {
                **metrics,
                'timestamp': datetime.now().isoformat()
            }

            logger.info(f"Results: C2ST={metrics['joint_c2st']:.4f}, "
                       f"Wasserstein={metrics['joint_wasserstein']:.4f}, "
                       f"MMD={metrics['joint_mmd']:.6f}")

            # Save intermediate results
            save_results(
                {task_name: task_results},
                output_dir / f"results_{task_name}_batched_partial.json"
            )

    return task_results


# ============================================================================
# RESULTS SAVING
# ============================================================================

def save_results(results: Dict, output_path: Path):
    """Save results as JSON and pickle."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {output_path}")

    pkl_path = output_path.with_suffix('.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"Saved results (pickle) to {pkl_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate NPE-PFN on RoPEFM benchmarks using sample_batched'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data',
        help='Path to data directory containing task subdirectories'
    )
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=['pendulum', 'wind_tunnel', 'light_tunnel'],
        help='Tasks to evaluate'
    )
    parser.add_argument(
        '--num_cal',
        nargs='+',
        type=int,
        default=[10, 50, 200, 1000],
        help='Calibration sizes to evaluate'
    )
    parser.add_argument(
        '--seeds',
        nargs='+',
        type=int,
        default=[0, 1, 2, 3, 4],
        help='Seeds for subsampling'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results/batched_eval',
        help='Output directory for results'
    )
    parser.add_argument(
        '--max_test_samples',
        type=int,
        default=None,
        help='Maximum number of test samples to use (for debugging/testing)'
    )
    parser.add_argument(
        '--projection_dim',
        type=int,
        default=128,
        help='Dimension for random projection embedding (for light_tunnel task)'
    )
    parser.add_argument(
        '--gpu_num',
        type=int,
        default=None,
        help='GPU number to use for metric computation'
    )

    args = parser.parse_args()

    # Initialize device
    if args.gpu_num is not None:
        if not torch.cuda.is_available():
            logger.warning(f"GPU {args.gpu_num} requested but CUDA not available. Using CPU.")
            device = torch.device('cpu')
        elif args.gpu_num >= torch.cuda.device_count():
            logger.warning(f"GPU {args.gpu_num} requested but only {torch.cuda.device_count()} GPUs available. Using GPU 0.")
            device = torch.device('cuda:0')
        else:
            device = torch.device(f'cuda:{args.gpu_num}')
            logger.info(f"Using GPU {args.gpu_num}: {torch.cuda.get_device_name(args.gpu_num)}")
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            logger.info(f"Auto-detected GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            logger.info("No GPU detected, using CPU")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup file logging
    log_file = output_dir / 'batched_evaluation.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))
    logger.addHandler(file_handler)

    logger.info("="*80)
    logger.info("NPE-PFN BATCHED Evaluation on RoPEFM Benchmarks")
    logger.info("="*80)
    logger.info(f"Tasks: {args.tasks}")
    logger.info(f"Calibration sizes: {args.num_cal}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Projection dim (for light_tunnel): {args.projection_dim}")
    logger.info(f"Device: {device}")
    logger.info(f"Output directory: {output_dir}")

    # Initialize results
    all_results = {}

    # Evaluate each task
    for task in args.tasks:
        try:
            task_results = evaluate_task_batched(
                task_name=task,
                num_cal_list=args.num_cal,
                seed_list=args.seeds,
                data_path=args.data_path,
                output_dir=output_dir,
                max_test_samples=args.max_test_samples,
                projection_dim=args.projection_dim,
                device=device
            )
            all_results[task] = task_results
        except Exception as e:
            logger.error(f"Error evaluating task {task}: {e}", exc_info=True)
            continue

    # Save final results
    save_results(all_results, output_dir / 'batched_results_final.json')

    logger.info("\n" + "="*80)
    logger.info("Batched evaluation complete!")
    logger.info(f"Results saved to {output_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
