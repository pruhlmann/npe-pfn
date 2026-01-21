# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NPE-PFN implements Neural Posterior Estimation using TabPFNv2 as a tabular foundation model for simulation-based Bayesian inference. Key feature: **no training required** - TabPFN enables instant posterior estimation from simulation data.

## Commands

```bash
# Install (editable)
pip install -e .

# Run all tests
pytest --log-cli-level=INFO tests

# Run fast CPU-only tests (no GPU required)
pytest -m fast --log-cli-level=INFO tests

# Run single test file
pytest --log-cli-level=INFO tests/test_npe_pfn.py

# Run specific test
pytest --log-cli-level=INFO tests/test_npe_pfn.py::test_sampling_and_log_prob_base

# Lint
ruff check .

# Format
ruff format .
```

## Architecture

### Core Classes (npe_pfn/)

- **`TabPFN_Based_NPE_PFN`** (npe_pfn.py) - Main inference class. Accepts prior + simulations via `append_simulations()`, then samples from posterior with `sample()`. Supports optional filtering strategies for large datasets via `filter_type` parameter.

- **`TabPFN_Based_Uncond_Estimator`** (npe_pfn.py) - Unconditional estimator variant that doesn't require conditioning on observations.

- **`run_tsnpe_pfn`** (tsnpe_pfn.py) - Sequential/multi-round variant (TSNPE-PFN). Iteratively refines the posterior by focusing simulations on promising regions. Key parameters: `num_rounds`, `num_simulations`, `sampling_method` ("rejection" or "sir").

- **`PosteriorSupport`** (support_posterior.py) - Estimates posterior support region for filtering. Factory function `get_filtering_method()` returns filtering strategies: `no_filtering`, `latest_filtering`, `random_filtering`, `standardized_euclidean_filtering`.

- **`NPE_PFN_RestrictedPrior`** (restricted_prior.py) - Restricted prior wrapper using TabPFN classifier for accept/reject decisions. Extends `sbi.utils.RestrictedPrior`.

### Log Probability Modes

`log_prob()` supports two modes:
- `"autoregressive"` - Sequential dimension-wise evaluation
- `"ratio_based"` - Density ratio estimation via classification

### Integration & Dependencies

- Built on the `sbi` package (v0.23.0). Uses `simulate_for_sbi()` for simulation orchestration.
- Uses TabPFN v2.0+ (PriorLabs-TabPFN) as the underlying tabular foundation model.
- Prior must be a torch distribution.
- Requires torch ~2.6.0, Python 3.10-3.13.

## Typical Workflow

### Basic NPE-PFN
```python
from npe_pfn import TabPFN_Based_NPE_PFN

posterior = TabPFN_Based_NPE_PFN(prior=prior)
posterior.append_simulations(thetas, xs)
samples = posterior.sample((num_samples,), x=observation)
```

### Sequential TSNPE-PFN
```python
from npe_pfn import run_tsnpe_pfn

posterior = run_tsnpe_pfn(
    simulator=simulator,
    prior=prior,
    observation=x_o,
    num_simulations=10_000,
    num_rounds=10,
    sampling_method="rejection"  # or "sir"
)
```

See `demo.ipynb` for complete examples.

## Testing

Tests use pytest markers:
- `@pytest.mark.fast` - Quick CPU-only tests suitable for CI
- Unmarked tests may require GPU or longer runtime
- Some tests use `@pytest.mark.xfail` for known limitations
