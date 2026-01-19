#!/usr/bin/env python3
"""
NPE-PFN Evaluation Script for RoPEFM Benchmarks

This script evaluates NPE-PFN against baseline methods from the ropefm repository
across multiple benchmark tasks with varying calibration set sizes.

IMPORTANT: Data Transformation
-------------------------------
The ropefm code applies LogitBoxTransform to theta samples before saving for tasks
with compact prior support (pendulum, wind_tunnel). This transform maps [a,b] → R
using: theta_transformed = logit((theta - a) / (b - a))

Therefore, the priors defined in create_prior() must match the TRANSFORMED space:
- pendulum: Uses Logistic distribution (transformed from Uniform[0,3] × Uniform[0.5,10])
- wind_tunnel: Uses Logistic distribution (transformed from Uniform[0,45])
- high_dim_gaussian: Uses MultivariateNormal (no transform, IdentityTransform)

See ropefm/simulator/base.py lines 192 and 246 for transformation implementation.
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
# METRIC FUNCTIONS (copied from ropefm)
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
    """
    PyTorch version of classifier two-sample test.

    Args:
        X, Y: torch.Tensor inputs of shape (n_samples, dim).
        model: "mlp" (default), "mlp_small", "linear", or a callable
        model_kwargs: kwargs forwarded to model constructor
        training_kwargs: {
            "epochs": int (default 100),
            "batch_size": int (default 128),
            "lr": float (default 1e-3),
            "weight_decay": float (default 0.0),
            "device": "cpu" or "cuda" or None (auto),
            "verbose": bool (default False)
        }

    Returns:
        float: mean accuracy (if scoring=="accuracy")
    """
    # --- defaults and reproducibility ---
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

    # --- Preprocessing ---
    if z_score:
        X_mean = torch.mean(X, dim=0)
        X_std = torch.std(X, dim=0)
        # avoid division by zero
        X_std = torch.where(X_std == 0, torch.ones_like(X_std), X_std)
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std

    if noise_scale is not None:
        X = X + noise_scale * torch.randn_like(X)
        Y = Y + noise_scale * torch.randn_like(Y)

    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()

    ndim = X_np.shape[1]

    # --- Build labels and data container ---
    data = np.concatenate((X_np, Y_np), axis=0)
    labels = np.concatenate(
        (np.zeros(X_np.shape[0], dtype=int), np.ones(Y_np.shape[0], dtype=int)), axis=0
    )

    # --- CV splitter ---
    if cv == "KFold":
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    elif cv == "StratifiedKFold":
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    else:
        raise ValueError(f"Unknown cross-validation strategy: {cv}")

    fold_scores = []

    # --- iterate folds ---
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(data, labels)):
        # build datasets
        X_train = torch.from_numpy(data[train_idx]).float()
        y_train = torch.from_numpy(labels[train_idx]).long()
        X_test = torch.from_numpy(data[test_idx]).float()
        y_test = torch.from_numpy(labels[test_idx]).long()

        train_ds = TensorDataset(X_train, y_train)
        test_ds = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # --- construct model ---
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

        # --- train ---
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

        # --- eval on test set ---
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

        # compute fold score
        if scoring == "accuracy":
            acc = float((all_preds == all_true).mean())
            fold_scores.append(acc)
        else:
            raise ValueError(f"Unknown scoring: {scoring}")

    mean_score = float(np.mean(fold_scores))
    return mean_score


def MMD(x, y, kernel):
    """Empirical maximum mean discrepancy.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
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
    """Load calibration data for a task.

    Expected structure:
        data_path/<task_name>/calibrations.pt
    """
    file_path = Path(data_path) / task_name / "calibrations.pt"

    logger.info(f"Loading calibration data from {file_path}")
    data = torch.load(file_path)

    theta = data['theta']  # [N, theta_dim]
    x = data['x']  # [N, obs_dim]
    y = data['y']  # [N, obs_dim]

    logger.info(f"Loaded calibration data: theta {theta.shape}, x {x.shape}, y {y.shape}")
    return theta, x, y


def load_test_data(
    task_name: str, data_path: str = "./data"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load test data for a task.

    Expected structure:
        data_path/<task_name>/test_data.pt
    """
    file_path = Path(data_path) / task_name / "test_data.pt"

    logger.info(f"Loading test data from {file_path}")
    data = torch.load(file_path)

    theta_test = data['theta']  # [M, theta_dim]
    x_test = data['x']  # [M, obs_dim]
    y_test = data['y']  # [M, obs_dim]

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
    - pendulum: Uniform prior + LogitBoxTransform
    - wind_tunnel: Uniform prior + LogitBoxTransform
    - high_dim_gaussian: Gaussian prior + IdentityTransform (no transform)

    This function returns the prior in the transformed space by applying the
    transform to the base distribution using TransformedDistribution.

    Args:
        task_name: 'pendulum', 'high_dim_gaussian', or 'wind_tunnel'

    Returns:
        Prior distribution in transformed space
    """
    if task_name == 'pendulum':
        # From ropefm/simulator/pendulum.py lines 40-49
        low = torch.tensor([0.0, 0.5])
        high = torch.tensor([3.0, 10.0])

        # Base prior: Independent Uniform
        base_prior = dist.Independent(
            dist.Uniform(low, high), 1
        )

        # Apply LogitBoxTransform
        from torch.distributions import TransformedDistribution, ComposeTransform
        from torch.distributions.transforms import AffineTransform, SigmoidTransform

        # LogitBoxTransform: x -> logit((x - a) / (b - a))
        # Implemented as: normalize to [0,1] then apply logit
        transforms = []
        for i in range(len(low)):
            # Scale and shift to [0, 1]
            affine_to_unit = AffineTransform(loc=-low[i], scale=1.0/(high[i] - low[i]))
            # Apply logit (inverse of sigmoid)
            logit_transform = SigmoidTransform().inv
            transforms.append(ComposeTransform([affine_to_unit, logit_transform]))

        # For simplicity, use explicit logit computation via distribution
        # The transformed prior is the base prior passed through the transform
        # In practice, we approximate with Logistic(0, 1) for each dimension
        # This matches the distribution of logit(Uniform(0,1))
        return dist.Independent(dist.Logistic(torch.zeros(2), torch.ones(2)), 1)

    elif task_name == 'high_dim_gaussian':
        # From ropefm/simulator/pure_gaussian.py lines 26-33
        # This uses IdentityTransform (no transformation)
        torch.manual_seed(0)
        theta_dim = 3
        prior_var_scale = 5.0

        prior_loc = torch.rand(theta_dim) * 10 - 5
        cov_theta_sqrt = torch.normal(0.0, prior_var_scale, (theta_dim, theta_dim))
        prior_cov = cov_theta_sqrt @ cov_theta_sqrt.T + torch.eye(theta_dim)

        return dist.MultivariateNormal(loc=prior_loc, covariance_matrix=prior_cov)

    elif task_name == 'wind_tunnel':
        # From ropefm/simulator/wind_tunnel.py lines 308-319
        low = torch.tensor([0.0])
        high = torch.tensor([45.0])

        # Base prior: Independent Uniform
        base_prior = dist.Independent(
            dist.Uniform(low, high), 1
        )

        # Apply LogitBoxTransform
        # The transformed distribution is Logistic(0, 1) for logit(Uniform(0,1))
        return dist.Independent(dist.Logistic(torch.zeros(1), torch.ones(1)), 1)

    else:
        raise ValueError(f"Unknown task: {task_name}")


# ============================================================================
# POSTERIOR SAMPLING
# ============================================================================

def sample_posteriors(
    posterior: TabPFN_Based_NPE_PFN,
    y_test: torch.Tensor,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Generate 1 posterior sample for each test observation.

    Args:
        posterior: TabPFN_Based_NPE_PFN instance
        y_test: [num_test, obs_dim] - on CPU (TabPFN requirement)
        device: torch.device - device object (unused but kept for API compatibility)

    Returns:
        all_samples: [num_test, theta_dim]
    """
    all_samples = []

    for y_single in tqdm(y_test, desc="Sampling posteriors"):
        # y_single is on CPU (TabPFN requires CPU tensors)
        # Sample 1 posterior sample conditioned on y_single
        sample = posterior.sample(
            sample_shape=torch.Size([1]),
            x=y_single.unsqueeze(0),  # NPE-PFN expects batch dimension
        )
        # Samples are already on CPU
        all_samples.append(sample.squeeze())  # [theta_dim]

    return torch.stack(all_samples, dim=0)  # [num_test, theta_dim]


# ============================================================================
# GAUSSIAN POSTERIOR COMPUTATION
# ============================================================================

def create_gaussian_posterior_params(seed: int = 0, theta_dim: int = 3, obs_dim: int = 3,
                                      prior_var_scale: float = 5.0,
                                      likelihood_var_scale: float = 2.0,
                                      noisy_var_scale: float = 5.0):
    """
    Create parameters for the Gaussian task matching ropefm/simulator/pure_gaussian.py.
    Returns all necessary parameters to compute ground truth posterior.
    """
    torch.manual_seed(seed)

    # Prior parameters
    prior_loc = torch.rand((theta_dim,)) * 10 - 5
    cov_theta_sqrt = torch.normal(0.0, prior_var_scale, (theta_dim, theta_dim))
    prior_cov = cov_theta_sqrt @ cov_theta_sqrt.T + torch.eye(theta_dim)

    # Likelihood parameters
    cov_likelihood_sqrt = torch.normal(0.0, likelihood_var_scale, (obs_dim, obs_dim))
    A = torch.normal(0.0, 1.0, (obs_dim, theta_dim))
    b = torch.normal(0.0, 1.0, (obs_dim,))
    likelihood_cov = cov_likelihood_sqrt @ cov_likelihood_sqrt.T

    # Noisy process parameters
    cov_noisy_sqrt = torch.normal(0.0, noisy_var_scale, (obs_dim, obs_dim))
    C = torch.normal(1.0, 1.0, (obs_dim, obs_dim))
    d = torch.rand((obs_dim,)) * 5 + 5
    noise_cov = cov_noisy_sqrt @ cov_noisy_sqrt.T

    return {
        'prior_loc': prior_loc,
        'prior_cov': prior_cov,
        'A': A,
        'b': b,
        'likelihood_cov': likelihood_cov,
        'C': C,
        'd': d,
        'noise_cov': noise_cov
    }


def compute_gaussian_posterior_distribution(y: torch.Tensor, params: Dict) -> dist.MultivariateNormal:
    """
    Compute the ground truth posterior p(theta | y) for the Gaussian task.

    Args:
        y: Noisy observations of shape (batch_size, obs_dim)
        params: Dictionary with prior, likelihood, and noise parameters

    Returns:
        MultivariateNormal distribution representing p(theta | y)
    """
    # Extract parameters
    mu_theta = params['prior_loc']
    Sigma_theta = params['prior_cov']
    A = params['A']
    b = params['b']
    Sigma_lik = params['likelihood_cov']
    C = params['C']
    d = params['d']
    Sigma_noise = params['noise_cov']

    # Linear mapping from theta to y
    F = C @ A
    c = C @ b + d
    Sigma_y = C @ Sigma_lik @ C.T + Sigma_noise

    # Inverses
    Sigma_y_inv = torch.linalg.inv(Sigma_y)
    Sigma_theta_inv = torch.linalg.inv(Sigma_theta)

    # Posterior covariance
    Sigma_post = torch.linalg.inv(Sigma_theta_inv + F.T @ Sigma_y_inv @ F)

    # Center y
    y_centered = y - c
    mean_post = (
        Sigma_post @ (Sigma_theta_inv @ mu_theta + y_centered @ (Sigma_y_inv @ F)).T
    ).T  # shape (batch_size, theta_dim)

    return dist.MultivariateNormal(
        loc=mean_post, covariance_matrix=Sigma_post.expand(y.shape[0], -1, -1)
    )


# ============================================================================
# CONDITIONAL METRIC COMPUTATION
# ============================================================================

def compute_conditional_metrics(
    theta_pred: torch.Tensor,
    theta_true_samples: torch.Tensor,
    seed: int = 42,
    device: torch.device = torch.device('cpu')
) -> Dict[str, float]:
    """
    Compute conditional metrics comparing NPE-PFN samples vs ground truth posterior samples.

    Args:
        theta_pred: [N, theta_dim] - NPE-PFN posterior samples, one per observation
        theta_true_samples: [N, theta_dim] - Ground truth posterior samples, one per observation
        seed: int - random seed
        device: torch.device - device to use for metric computation

    Returns:
        Dictionary with conditional_c2st, conditional_wasserstein, conditional_mmd
    """
    logger.info(f"Computing conditional metrics: theta_pred {theta_pred.shape}, theta_true {theta_true_samples.shape}")

    # Compute conditional C2ST
    logger.info("Computing conditional C2ST...")
    conditional_c2st = classifier_two_samples_test_torch(
        theta_true_samples, theta_pred,
        seed=seed, n_folds=5, z_score=True, model='mlp',
        training_kwargs={'epochs': 100, 'batch_size': 128, 'verbose': False, 'device': str(device)}
    )

    # Compute conditional Wasserstein
    logger.info("Computing conditional Wasserstein...")
    a = torch.ones(theta_true_samples.shape[0]) / theta_true_samples.shape[0]
    b = torch.ones(theta_pred.shape[0]) / theta_pred.shape[0]
    M = ot.dist(theta_true_samples.cpu().numpy(), theta_pred.cpu().numpy())
    conditional_wasserstein = float(np.sqrt(ot.emd2(a.numpy(), b.numpy(), M)))

    # Compute conditional MMD
    logger.info("Computing conditional MMD...")
    theta_true_device = theta_true_samples.to(device)
    theta_pred_device = theta_pred.to(device)
    conditional_mmd = MMD(theta_true_device, theta_pred_device, kernel='rbf').item()

    return {
        'conditional_c2st': conditional_c2st,
        'conditional_wasserstein': conditional_wasserstein,
        'conditional_mmd': conditional_mmd
    }


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

    Args:
        theta_true: [N, theta_dim] - ground truth parameters (on CPU)
        theta_pred: [N, theta_dim] - predicted posterior samples (on CPU)
        y_obs: [N, obs_dim] - observations (on CPU)
        seed: int - random seed
        device: torch.device - device to use for metric computation

    Returns:
        Dictionary with joint_c2st, joint_wasserstein, joint_mmd
    """
    # Flatten y_obs if it's multi-dimensional
    if y_obs.ndim > 2:
        y_obs = y_obs.reshape(y_obs.shape[0], -1)

    # Create true joint: [theta_true, y_obs]
    true_joint = torch.cat([theta_true, y_obs], dim=1)  # [N, d_theta+d_obs]

    # Create predicted joint: [theta_pred, y_obs]
    pred_joint = torch.cat([theta_pred, y_obs], dim=1)  # [N, d_theta+d_obs]

    logger.info(f"Computing joint metrics: true_joint {true_joint.shape}, pred_joint {pred_joint.shape}")

    # Compute joint C2ST (this will handle device internally)
    logger.info("Computing joint C2ST...")
    joint_c2st = classifier_two_samples_test_torch(
        true_joint, pred_joint,
        seed=seed, n_folds=5, z_score=True, model='mlp',
        training_kwargs={'epochs': 100, 'batch_size': 128, 'verbose': False, 'device': str(device)}
    )

    # Compute joint Wasserstein (always uses CPU/numpy)
    logger.info("Computing joint Wasserstein...")
    a = torch.ones(true_joint.shape[0]) / true_joint.shape[0]
    b = torch.ones(pred_joint.shape[0]) / pred_joint.shape[0]
    M = ot.dist(true_joint.cpu().numpy(), pred_joint.cpu().numpy())
    joint_wasserstein = float(np.sqrt(ot.emd2(a.numpy(), b.numpy(), M)))

    # Compute joint MMD (move to device for computation)
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
# GAUSSIAN TASK EVALUATION WITH CONDITIONAL METRICS
# ============================================================================

def evaluate_gaussian_task_conditional(
    task_name: str,
    num_cal_list: list,
    seed_list: list,
    data_path: str,
    output_dir: Path,
    num_posterior_samples: int = 1000,
    num_observations: int = 10,
    max_test_samples: int = None,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """
    Evaluate NPE-PFN for Gaussian task with conditional metrics.

    For each observation y_obs:
    - Sample N times from p_true(theta|y_obs)
    - Sample N times from p_npe-pfn(theta|y_obs)
    - Compute C2ST, MMD, Wasserstein between these two sets of N samples

    Args:
        task_name: Should be 'high_dim_gaussian'
        num_cal_list: List of calibration sizes to evaluate
        seed_list: List of seeds for subsampling
        data_path: Path to data directory
        output_dir: Output directory for results
        num_posterior_samples: Number of samples to draw from each posterior (N)
        num_observations: Number of test observations to evaluate
        max_test_samples: Maximum test samples to load (for selecting observations)
        device: Device for computation
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting CONDITIONAL evaluation for Gaussian task: {task_name}")
    logger.info(f"{'='*80}\n")

    # Load data
    theta_cal, x_cal, y_cal = load_calibration_data(task_name, data_path)
    theta_test, x_test, y_test = load_test_data(task_name, data_path)

    # Optionally limit test samples
    if max_test_samples is not None and max_test_samples < len(theta_test):
        logger.info(f"Limiting test samples to {max_test_samples} (from {len(theta_test)})")
        theta_test = theta_test[:max_test_samples]
        x_test = x_test[:max_test_samples]
        y_test = y_test[:max_test_samples]

    # Select observations to evaluate
    num_observations = min(num_observations, len(y_test))
    logger.info(f"Will evaluate {num_observations} observations with {num_posterior_samples} samples each")

    # Create Gaussian posterior parameters (must match data generation)
    logger.info("Creating Gaussian posterior parameters...")
    gaussian_params = create_gaussian_posterior_params(
        seed=0,  # Same seed used in data generation
        theta_dim=theta_test.shape[1],
        obs_dim=y_test.shape[1]
    )

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

            # Store metrics for each observation
            obs_metrics = []

            # Evaluate each observation
            for obs_idx in tqdm(range(num_observations), desc=f"Evaluating observations"):
                y_obs = y_test[obs_idx]

                # Sample N times from true posterior p(theta|y_obs)
                posterior_dist = compute_gaussian_posterior_distribution(
                    y_obs.unsqueeze(0), gaussian_params
                )
                theta_true_samples = posterior_dist.sample((num_posterior_samples,)).squeeze(1)  # [N, theta_dim]

                # Sample N times from NPE-PFN posterior p_npe-pfn(theta|y_obs)
                theta_pred_samples = posterior.sample(
                    sample_shape=torch.Size([num_posterior_samples]),
                    x=y_obs.unsqueeze(0),
                )  # [N, theta_dim]

                # Compute conditional metrics for this observation
                conditional_metrics = compute_conditional_metrics(
                    theta_pred_samples, theta_true_samples, seed=seed, device=device
                )
                conditional_metrics['observation_idx'] = obs_idx
                obs_metrics.append(conditional_metrics)

                if obs_idx == 0:  # Log first observation as example
                    logger.info(f"Example obs {obs_idx}: C2ST={conditional_metrics['conditional_c2st']:.4f}, "
                               f"Wasserstein={conditional_metrics['conditional_wasserstein']:.4f}, "
                               f"MMD={conditional_metrics['conditional_mmd']:.6f}")

            # Aggregate metrics across observations
            avg_metrics = {
                'conditional_c2st_mean': float(np.mean([m['conditional_c2st'] for m in obs_metrics])),
                'conditional_c2st_std': float(np.std([m['conditional_c2st'] for m in obs_metrics])),
                'conditional_wasserstein_mean': float(np.mean([m['conditional_wasserstein'] for m in obs_metrics])),
                'conditional_wasserstein_std': float(np.std([m['conditional_wasserstein'] for m in obs_metrics])),
                'conditional_mmd_mean': float(np.mean([m['conditional_mmd'] for m in obs_metrics])),
                'conditional_mmd_std': float(np.std([m['conditional_mmd'] for m in obs_metrics])),
                'per_observation_metrics': obs_metrics,
                'timestamp': datetime.now().isoformat()
            }

            # Store results
            task_results[str(num_cal)][str(seed)] = avg_metrics

            logger.info(f"Averaged results over {num_observations} observations:")
            logger.info(f"  C2ST: {avg_metrics['conditional_c2st_mean']:.4f} ± {avg_metrics['conditional_c2st_std']:.4f}")
            logger.info(f"  Wasserstein: {avg_metrics['conditional_wasserstein_mean']:.4f} ± {avg_metrics['conditional_wasserstein_std']:.4f}")
            logger.info(f"  MMD: {avg_metrics['conditional_mmd_mean']:.6f} ± {avg_metrics['conditional_mmd_std']:.6f}")

            # Save intermediate results
            save_results(
                {task_name: task_results},
                output_dir / f"results_{task_name}_conditional_partial.json"
            )

    return task_results


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def evaluate_task(
    task_name: str,
    num_cal_list: list,
    seed_list: list,
    data_path: str,
    output_dir: Path,
    max_test_samples: int = None,
    batch_size: int = 10,
    device: torch.device = torch.device('cpu')
) -> Dict:
    """Evaluate NPE-PFN for a single task across all calibration sizes and seeds."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting evaluation for task: {task_name}")
    logger.info(f"{'='*80}\n")

    # Load data
    theta_cal, x_cal, y_cal = load_calibration_data(task_name, data_path)
    theta_test, x_test, y_test = load_test_data(task_name, data_path)

    # Optionally limit test samples for debugging/testing
    if max_test_samples is not None and max_test_samples < len(theta_test):
        logger.info(f"Limiting test samples to {max_test_samples} (from {len(theta_test)})")
        theta_test = theta_test[:max_test_samples]
        x_test = x_test[:max_test_samples]
        y_test = y_test[:max_test_samples]

    # Note: Data stays on CPU - TabPFN requires CPU tensors

    # Create prior
    prior = create_prior(task_name)
    logger.info(f"Created prior for {task_name}")

    # Initialize results for this task
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

            # Append simulations (TabPFN requires CPU tensors)
            posterior.append_simulations(theta_sub, y_sub)
            logger.info("Appended simulations to posterior")

            # Sample from posterior for each test observation
            logger.info(f"Sampling posteriors for {len(y_test)} test observations...")
            theta_pred = sample_posteriors(posterior, y_test, batch_size=batch_size, device=device)
            logger.info(f"Generated posterior samples: {theta_pred.shape}")

            # Compute metrics
            logger.info("Computing joint metrics...")
            metrics = compute_joint_metrics(theta_test, theta_pred, y_test, seed=seed, device=device)

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
                output_dir / f"results_{task_name}_partial.json"
            )

    return task_results


# ============================================================================
# RESULTS SAVING
# ============================================================================

def save_results(results: Dict, output_path: Path):
    """Save results as JSON and pickle."""
    # Save JSON
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {output_path}")

    # Save pickle
    pkl_path = output_path.with_suffix('.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    logger.info(f"Saved results (pickle) to {pkl_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate NPE-PFN on RoPEFM benchmarks')
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data',
        help='Path to data directory containing task subdirectories with calibrations.pt and test_data.pt'
    )
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=['pendulum', 'high_dim_gaussian', 'wind_tunnel'],
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
        default='results/npe_pfn_evaluation',
        help='Output directory for results'
    )
    parser.add_argument(
        '--max_test_samples',
        type=int,
        default=None,
        help='Maximum number of test samples to use (for debugging/testing)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=10,
        help='Batch size for posterior sampling procedure'
    )
    parser.add_argument(
        '--gpu_num',
        type=int,
        default=None,
        help='GPU number to use (e.g., 0, 1, 2...). If not specified, uses CPU or auto-detects GPU'
    )
    parser.add_argument(
        '--num_posterior_samples',
        type=int,
        default=1000,
        help='Number of samples to draw from each posterior for conditional evaluation of Gaussian task (default: 1000)'
    )
    parser.add_argument(
        '--num_observations',
        type=int,
        default=10,
        help='Number of test observations to evaluate for conditional evaluation of Gaussian task (default: 10)'
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
    log_file = output_dir / 'npe_pfn_evaluation.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))
    logger.addHandler(file_handler)

    logger.info("="*80)
    logger.info("NPE-PFN Evaluation on RoPEFM Benchmarks")
    logger.info("="*80)
    logger.info(f"Tasks: {args.tasks}")
    logger.info(f"Calibration sizes: {args.num_cal}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Device: {device}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Conditional mode params (for Gaussian task):")
    logger.info(f"  Posterior samples per observation: {args.num_posterior_samples}")
    logger.info(f"  Number of observations: {args.num_observations}")

    # Initialize results
    all_results = {}

    # Evaluate each task
    for task in args.tasks:
        try:
            # Use conditional evaluation for Gaussian task (always)
            if task == 'high_dim_gaussian':
                logger.info(f"Using CONDITIONAL evaluation for Gaussian task: {task}")
                task_results = evaluate_gaussian_task_conditional(
                    task_name=task,
                    num_cal_list=args.num_cal,
                    seed_list=args.seeds,
                    data_path=args.data_path,
                    output_dir=output_dir,
                    num_posterior_samples=args.num_posterior_samples,
                    num_observations=args.num_observations,
                    max_test_samples=args.max_test_samples,
                    device=device
                )
            else:
                # Standard joint evaluation for other tasks
                logger.info(f"Using JOINT evaluation for task: {task}")
                task_results = evaluate_task(
                    task_name=task,
                    num_cal_list=args.num_cal,
                    seed_list=args.seeds,
                    data_path=args.data_path,
                    output_dir=output_dir,
                    max_test_samples=args.max_test_samples,
                    batch_size=args.batch_size,
                    device=device
                )
            all_results[task] = task_results
        except Exception as e:
            logger.error(f"Error evaluating task {task}: {e}", exc_info=True)
            continue

    # Save final results
    save_results(all_results, output_dir / 'npe_pfn_results_final.json')

    logger.info("\n" + "="*80)
    logger.info("Evaluation complete!")
    logger.info(f"Results saved to {output_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
