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

# Run fast CPU-only tests
pytest -m fast --log-cli-level=INFO tests

# Lint
ruff check .
```

## Architecture

### Core Classes (npe_pfn/)

- **`TabPFN_Based_NPE_PFN`** - Main inference class. Accepts prior + simulations via `append_simulations()`, then samples from posterior with `sample()`. Supports optional filtering strategies for large datasets.

- **`run_tsnpe_pfn`** - Sequential/multi-round variant (TSNPE-PFN). Iteratively refines the posterior by focusing simulations on promising regions.

- **`PosteriorSupport`** (support_posterior.py) - Estimates posterior support region for filtering. Factory function `get_filtering_method()` returns filtering strategies: `no_filtering`, `latest_filtering`, `random_filtering`, `standardized_euclidean_filtering`.

### Log Probability Modes

`log_prob()` supports two modes:
- `"autoregressive"` - Sequential dimension-wise evaluation
- `"ratio_based"` - Density ratio estimation via classification

### Integration

Built on the `sbi` package. Uses `simulate_for_sbi()` for simulation orchestration. Prior must be a torch distribution.

## Typical Workflow

```python
from npe_pfn import TabPFN_Based_NPE_PFN

posterior = TabPFN_Based_NPE_PFN(prior=prior)
posterior.append_simulations(thetas, xs)
samples = posterior.sample((num_samples,), x=observation)
```

See `demo.ipynb` for complete examples.
