"""Regression candidates for the MMA kinetic dataset.

No estimator uses reaction IDs, source metadata, folds, or measured outcomes as features.
"""
import numpy as np
from scipy.optimize import least_squares
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def group_weights(groups):
    """Each condition contributes equally to weighted least squares."""
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return len(groups) / (len(counts) * counts[inverse])


class KineticRegressor(RegressorMixin, BaseEstimator):
    """A low-capacity, physically motivated apparent first-order baseline.

    X(t) = 1 - exp(-k_eff * t).
    log(k_eff) = b0 + bT * 1000*(1/Tref - 1/Tkelvin)
                    + bR * log(1 + CT/CM/0.05).
    The apparent rate is positive; this is not a mechanistic FRP simulator.
    The temperature coefficient is constrained nonnegative; CT/CM coefficient
    is unconstrained within a broad numerical bound. Time is in minutes.
    """
    def __init__(self, regularization=0.01):
        self.regularization = regularization

    @staticmethod
    def _design(X):
        X = check_array(X)
        if X.shape[1] != 3 or np.any(X[:, 0] <= -273.15) or np.any(X[:, 1:] < 0):
            raise ValueError("Expected temperature (C), CT/CM >= 0, residence >= 0")
        T = X[:, 0] + 273.15
        z = np.column_stack([np.ones(len(X)), 1000 * (1 / 363.15 - 1 / T),
                             np.log1p(X[:, 1] / 0.05)])
        return z, X[:, 2]

    @staticmethod
    def _evaluate(z, t, beta):
        k = np.exp(np.clip(z @ beta, -40, 40))
        return -np.expm1(-k * t)

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        z, t = self._design(X)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if w.shape != y.shape or np.any(w <= 0) or not np.all(np.isfinite(w)):
            raise ValueError("Invalid sample weights")
        sw = np.sqrt(w / w.sum())
        penalty = np.sqrt(self.regularization)

        def residual(beta):
            error = sw * (self._evaluate(z, t, beta) - y)
            return np.r_[error, penalty * beta[1:] / np.array([10.0, 2.0])]

        result = least_squares(residual, [-4.5, 4.0, 0.0],
                               bounds=([-20, 0, -10], [5, 30, 10]), max_nfev=500)
        if not result.success:
            raise RuntimeError("Kinetic baseline optimization failed: " + result.message)
        self.coef_ = result.x
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        z, t = self._design(X)
        return self._evaluate(z, t, self.coef_)


def make_gp(seed):
    # Scale all inputs using statistics of this fit only.
    kernel = (ConstantKernel(1.0, (0.01, 100.0))
              * Matern(np.ones(3), (0.08, 30.0), nu=2.5)
              + WhiteKernel(0.01, (1e-5, 0.5)))
    return make_pipeline(StandardScaler(), GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, alpha=1e-8, random_state=seed,
        n_restarts_optimizer=0))


def make_estimator(name, params, seed):
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(**params))
    if name == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**params, loss_function="RMSE", random_seed=seed,
                                 thread_count=1, verbose=False, allow_writing_files=False)
    if name == "gp":
        return make_gp(seed)
    if name == "kinetic":
        return KineticRegressor(**params)
    raise ValueError("Unknown model: " + name)


def fit_estimator(name, params, X, y, groups, seed):
    estimator = make_estimator(name, params, seed)
    weights = group_weights(groups)
    if name == "ridge":
        estimator.fit(X, y, ridge__sample_weight=weights)
    elif name in {"catboost", "kinetic"}:
        estimator.fit(X, y, sample_weight=weights)
    else:
        # Exact sklearn GP uses the row likelihood, not group weights.
        estimator.fit(X, y)
    return estimator


def predict(estimator, X):
    return np.clip(estimator.predict(X), 0.0, 1.0)


def predict_gp(estimator, X):
    X = estimator.named_steps["standardscaler"].transform(X)
    mean, std = estimator.named_steps["gaussianprocessregressor"].predict(X, return_std=True)
    # WhiteKernel is part of predictive observation variance.
    return mean, np.maximum(std, 1e-9)
