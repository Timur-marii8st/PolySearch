"""Sequential policies receiving only candidate features and revealed labels."""
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.linear_model import Ridge

POLICIES = ("random", "maximin", "ridge_greedy", "gp_ei", "ridge_residual_gp_ei")
# Fixed prior scales (fraction of conversion), not estimated on hidden labels.
SIGNAL_SD = 0.05
NOISE_SD = 0.02
LENGTH_SCALE = 0.35
RIDGE_ALPHA = 1.0


def geometry(X):
    X = np.asarray(X, dtype=float)
    span = np.ptp(X, axis=0)
    return (X - X.min(axis=0)) / np.where(span > 0, span, 1.0)


def expected_improvement(mean, std, incumbent):
    delta = np.asarray(mean) - float(incumbent)
    std = np.asarray(std)
    safe = np.maximum(std, 1e-12)
    z = delta / safe
    return np.maximum(0.0, np.where(std > 1e-12,
                                    delta * norm.cdf(z) + safe * norm.pdf(z),
                                    np.maximum(delta, 0.0)))


def choose(policy, X, observed_indices, observed_y, rng):
    """Return an unmeasured index. No hidden outcomes are passed to this API."""
    X = np.asarray(X, dtype=float)
    observed = np.asarray(observed_indices, dtype=int)
    y = np.asarray(observed_y, dtype=float)
    if X.ndim != 2 or not np.isfinite(X).all() or len(y) != len(observed) or not len(y):
        raise ValueError("Invalid observed history")
    if (len(set(observed)) != len(observed) or np.any(observed < 0)
            or np.any(observed >= len(X)) or not np.isfinite(y).all()):
        raise ValueError("Invalid observed indices or outcomes")
    remaining = np.setdiff1d(np.arange(len(X)), observed)
    if not len(remaining):
        raise ValueError("Pool exhausted")
    if policy not in POLICIES:
        raise ValueError("Unknown policy")
    if policy == "random":
        return int(rng.choice(remaining))
    Z = geometry(X)  # Entire geometry is known before any query; no labels.
    train, candidates = Z[observed], Z[remaining]
    if policy == "maximin":
        score = np.linalg.norm(candidates[:, None, :] - train[None, :, :], axis=2).min(axis=1)
    elif policy == "ridge_greedy":
        trend = Ridge(alpha=RIDGE_ALPHA).fit(train, y)
        score = trend.predict(candidates)
    else:
        if policy == "ridge_residual_gp_ei":
            trend = Ridge(alpha=RIDGE_ALPHA).fit(train, y)
            residuals = y - trend.predict(train)
            baseline = trend.predict(candidates)
        else:
            baseline = np.full(len(candidates), np.mean(y))
            residuals = y - np.mean(y)
        kernel = ConstantKernel(SIGNAL_SD**2, constant_value_bounds="fixed") * Matern(
            length_scale=np.full(X.shape[1], LENGTH_SCALE),
            length_scale_bounds="fixed", nu=2.5)
        model = GaussianProcessRegressor(kernel=kernel, alpha=NOISE_SD**2 + 1e-10,
                                         optimizer=None, normalize_y=False)
        model.fit(train, residuals)
        mean, sd = model.predict(candidates, return_std=True)
        # Latent posterior variance, not predictive observation noise.
        # For the residual GP the fitted Ridge trend is treated as fixed.
        score = expected_improvement(baseline + mean, sd, np.max(y))
    if not np.all(np.isfinite(score)):
        raise ValueError("Non-finite policy score")
    # Random seeded tie-breaker, independent of hidden outcomes.
    tied = np.flatnonzero(np.isclose(score, score.max(), rtol=1e-10, atol=1e-12))
    return int(remaining[int(rng.choice(tied))])
