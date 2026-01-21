# Implementation of `sample_batched` for NPE-PFN

This document explains the implementation of true batched sampling for multiple observations in NPE-PFN.

## Overview

The `sample_batched` method allows sampling from the posterior for multiple observations simultaneously, avoiding the overhead of calling `sample()` in a for-loop. This provides significant speedups (5x+ in benchmarks) by leveraging TabPFN's ability to predict for multiple test points in a single forward pass.

## Architecture

The implementation consists of two methods:

1. **`_sample_batched`** - Core autoregressive sampling for multiple observations
2. **`sample_batched`** - Public API with input validation and rejection sampling

```
sample_batched(x)  →  _sample_batched(x)  →  TabPFN predictions
       ↓                     ↓
   Validation          Autoregressive loop
   Rejection           (runs ONCE for all obs)
```

---

## Key Insight: Batching Through Interleaving

The critical insight is that TabPFN's autoregressive sampling loop can process all observations together if we properly interleave the data.

### Original `_sample` (single observation)

```python
# For 1 observation, N samples:
samples_batch = x.repeat(N, 1)  # Shape: [N, dim_x]

for param_idx in range(dim_theta):
    model.fit(context_features, context_target)
    pred = model.predict(samples_batch)  # Predict for N points
    param_samples = pred.sample()
    samples_batch = concat(samples_batch, param_samples)
```

### New `_sample_batched` (multiple observations)

```python
# For M observations, N samples each:
samples_batch = x.repeat_interleave(N, dim=0)  # Shape: [M*N, dim_x]
# Layout: [obs1, obs1, ..., obs2, obs2, ..., obsM, obsM, ...]
#          \___ N times ___/  \___ N times ___/

for param_idx in range(dim_theta):
    model.fit(context_features, context_target)  # Same context for all
    pred = model.predict(samples_batch)  # Predict for M*N points at once!
    param_samples = pred.sample()
    samples_batch = concat(samples_batch, param_samples)

# Reshape: [M*N, dim_theta] → [M, N, dim_theta]
theta_samples = samples_batch[:, dim_x:].reshape(M, N, dim_theta)
```

---

## Detailed Implementation

### 1. `_sample_batched` - Core Method

Located at `npe_pfn/npe_pfn.py:171-251`

#### Step 1: Prepare interleaved observations

```python
num_obs = x.shape[0]      # M observations
dim_x = x.shape[1]

# Interleave: each observation repeated num_samples_per_obs times
samples_batch = x.repeat_interleave(num_samples_per_obs, dim=0)
# Shape: [M * N, dim_x]
```

**Example:** For 3 observations and 2 samples each:
```
x = [[obs1], [obs2], [obs3]]  # Shape: [3, dim_x]

samples_batch = [[obs1],    # For sample 1 of obs1
                 [obs1],    # For sample 2 of obs1
                 [obs2],    # For sample 1 of obs2
                 [obs2],    # For sample 2 of obs2
                 [obs3],    # For sample 1 of obs3
                 [obs3]]    # For sample 2 of obs3
# Shape: [6, dim_x]
```

#### Step 2: Shared context (no filtering)

```python
theta_context = self._theta_train  # All training parameters
x_context = self._x_train          # All training observations
joint_data = torch.cat([x_context, theta_context], dim=1)
```

**Important:** Batched sampling uses all training data as shared context. This differs from `TabPFN_Based_NPE_PFN` which filters context per observation. The trade-off:
- ✅ Enables true batching (same context for all)
- ⚠️ Context limited to TabPFN's capacity (~10K samples)

#### Step 3: Autoregressive loop (runs once!)

```python
for param_idx in range(dim_theta):
    # Features: x + already sampled theta dimensions
    features_end = dim_x + param_idx
    target_idx = dim_x + param_idx

    # Fit on joint training data
    self._model.fit(
        joint_data[:, :features_end],   # [x, θ₁, ..., θₖ₋₁]
        joint_data[:, target_idx]        # θₖ
    )

    # Predict for ALL observations at once
    pred_dist = self._model.predict(samples_batch, output_type="full")
    param_samples = pred_dist["criterion"].sample(pred_dist["logits"])

    # Append new dimension to all samples
    samples_batch = torch.cat([samples_batch, param_samples[:, None]], dim=1)
```

**This is the key optimization:** The loop runs `dim_theta` times total, not `dim_theta × num_obs` times.

#### Step 4: Handle log probabilities (with device management)

```python
if with_log_prob:
    # TabPFN may return samples on CPU while model is on GPU
    device = pred_dist["logits"].device
    param_samples_device = param_samples.to(device)

    dim_log_prob = -pred_dist["criterion"](
        pred_dist["logits"], param_samples_device
    )

    # Handle -inf values for numerical stability
    dim_log_prob = torch.where(
        dim_log_prob == float("-inf"),
        torch.log(torch.tensor(eps, device=dim_log_prob.device)),
        dim_log_prob,
    )

    log_probs_batch = log_probs_batch.to(dim_log_prob.device)
    log_probs_batch += dim_log_prob
```

#### Step 5: Reshape output

```python
# Extract theta dimensions and reshape
theta_samples = samples_batch[:, dim_x:]  # [M*N, dim_theta]
theta_samples = theta_samples.reshape(num_obs, num_samples_per_obs, dim_theta)
# Final shape: [M, N, dim_theta]
```

---

### 2. `sample_batched` - Public API

Located at `npe_pfn/npe_pfn.py:310-410`

#### Input validation

```python
if self.embedding_net:
    x = x.reshape(-1, *self.x_shape)
    x = self.embedding_net(x)
x = self._validate_x(x)

num_obs = x.shape[0]
num_samples = torch.Size(sample_shape).numel()
```

#### Without prior: Direct sampling

```python
if self.prior is None:
    samples, log_probs = self._sample_batched(
        x, num_samples, with_log_prob=with_log_prob, eps=eps
    )
    return samples if not with_log_prob else (samples, log_probs)
```

#### With prior: Rejection sampling

When a prior is specified, samples must be within the prior's support. This requires rejection sampling:

```python
# Oversample to account for rejections
num_to_sample = int(num_samples * oversample_factor)  # Default 1.5x

samples_needed = torch.full((num_obs,), num_samples, dtype=torch.long)
collected_samples = [[] for _ in range(num_obs)]

max_iter = 10
for iteration in range(max_iter):
    if samples_needed.sum() == 0:
        break

    # Sample in batch (all observations together)
    raw_samples, raw_log_probs = self._sample_batched(
        x, num_to_sample, with_log_prob=with_log_prob, eps=eps
    )

    # Filter per observation
    for obs_idx in range(num_obs):
        if samples_needed[obs_idx] == 0:
            continue

        obs_samples = raw_samples[obs_idx]
        valid_mask = self._within_support(obs_samples)
        valid_samples = obs_samples[valid_mask]

        n_take = min(len(valid_samples), samples_needed[obs_idx].item())
        collected_samples[obs_idx].append(valid_samples[:n_take])
        samples_needed[obs_idx] -= n_take

# Stack results
final_samples = torch.stack([
    torch.cat(s, dim=0)[:num_samples] for s in collected_samples
])
```

---

## Complexity Analysis

| Method | TabPFN fits | TabPFN predictions |
|--------|------------|-------------------|
| For-loop (`sample` × M) | `M × dim_theta` | `M × dim_theta × N` |
| Batched (`sample_batched`) | `dim_theta` | `dim_theta × (M × N)` |

The batched method reduces TabPFN fits from `M × dim_theta` to just `dim_theta`, which is the primary source of speedup.

---

## Limitations & Trade-offs

1. **No per-observation filtering**: Uses shared context for all observations. For `TabPFN_Based_NPE_PFN`, this means filtering is disabled during batched sampling.

2. **Context size limit**: TabPFN has a context limit (~10K samples). Large training sets may need truncation.

3. **Memory**: Processes `M × N` samples simultaneously. For very large batches, may need chunking.

4. **Rejection sampling overhead**: With restrictive priors, multiple iterations may be needed.

---

## Usage Example

```python
from npe_pfn.npe_pfn import NPE_PFN_Core

# Setup
model = NPE_PFN_Core(prior=prior)
model.append_simulations(theta_train, x_train)

# Multiple observations
x_test = torch.randn(10, obs_dim)  # 10 observations

# Batched sampling (fast)
samples = model.sample_batched(
    x=x_test,
    sample_shape=torch.Size([100])  # 100 samples per observation
)
# Output shape: [10, 100, theta_dim]

# With log probabilities
samples, log_probs = model.sample_batched(
    x=x_test,
    sample_shape=torch.Size([100]),
    with_log_prob=True
)
# samples: [10, 100, theta_dim]
# log_probs: [10, 100]
```

---

## Files Modified

| File | Changes |
|------|---------|
| `npe_pfn/npe_pfn.py` | Added `_sample_batched` (lines 171-251), replaced `sample_batched` stub (lines 310-410) |
| `tests/test_npe_pfn.py` | Added `test_sample_batched` and `test_sample_batched_single_obs_matches_sample` |
| `notebooks/benchmark_sample_batched.ipynb` | Benchmark notebook comparing performance |
