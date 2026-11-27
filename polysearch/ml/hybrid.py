"""Physics-inspired surrogates for conversion curves, separate from the generic API.

One apparent log-rate per condition group limits the pseudo-replication of
correlated spectral rows. This is an empirical first-order approximation,
not a mechanistic or validated safety model.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .estimators import KineticRegressor, group_weights


def physical_features(X):
    X = np.asarray(X, dtype=float)
    if (X.ndim != 2 or X.shape[1] != 3 or not np.isfinite(X).all()
            or np.any(X[:, 0] <= -273.15) or np.any(X[:, 1:] < 0)):
        raise ValueError("Expected finite temperature C, CT/CM >= 0, residence >= 0")
    return np.column_stack([1000 * (1/363.15 - 1/(X[:, 0]+273.15)),
                            np.log1p(X[:, 1]/0.05)])


def conversion(log_rate, t):
    return -np.expm1(-np.exp(np.clip(log_rate, -30, 20)) * t)


def condition_rates(X, y, groups):
    X, y, groups = np.asarray(X, float), np.asarray(y, float), np.asarray(groups)
    physical_features(X)
    if y.shape != (len(X),) or groups.shape != y.shape or not len(y):
        raise ValueError("Invalid grouped curve data")
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
        raise ValueError("Conversion must be in [0,1]")
    conditions, rates = [], []
    for group in np.unique(groups):
        keep = groups == group
        xg, yg = X[keep], y[keep]
        if len(np.unique(xg[:, :2], axis=0)) != 1 or not np.any(xg[:, 2] > 0):
            raise ValueError("Each group must describe one condition with positive residence time")
        result = minimize_scalar(lambda log_k: np.mean((conversion(log_k, xg[:, 2])-yg)**2),
                                 bounds=(-16, 2), method="bounded",
                                 options={"xatol": 1e-10})
        if not result.success:
            raise RuntimeError("Apparent rate fit failed")
        conditions.append([*xg[0, :2], 0.0])
        rates.append(float(result.x))
    return np.array(conditions), np.array(rates)


def residual_gp(dim, length, signal, noise):
    if min(length, signal, noise) <= 0:
        raise ValueError("Positive GP prior scales required")
    kernel = ConstantKernel(signal**2, constant_value_bounds="fixed") * Matern(
        length_scale=np.full(dim, length), length_scale_bounds="fixed", nu=2.5)
    return GaussianProcessRegressor(kernel=kernel, alpha=noise**2 + 1e-10,
                                    optimizer=None, normalize_y=False)


class RateSurrogate:
    """Ridge on physical log-rate features, optionally GP on group-level residuals.

    Predictions are medians under the fitted latent log-rate model, not posterior
    means of conversion. Latent uncertainty excludes measurement noise and trend
    estimation uncertainty; downstream group calibration is assessed separately.
    """
    def __init__(self, alpha=1.0, use_gp=True, length=1.0, signal=0.5, noise=0.25):
        self.alpha, self.use_gp = alpha, use_gp
        self.length, self.signal, self.noise = length, signal, noise

    def fit(self, X, y, groups):
        conditions, rates = condition_rates(X, y, groups)
        if len(rates) < 2:
            raise ValueError("Need at least two condition groups")
        Z = physical_features(conditions)
        self.scaler_ = StandardScaler().fit(Z)
        Z = self.scaler_.transform(Z)
        self.trend_ = Ridge(alpha=self.alpha).fit(Z, rates)
        self.group_count_ = len(rates)
        self.gp_ = None
        if self.use_gp:
            self.gp_ = residual_gp(2, self.length, self.signal, self.noise)
            self.gp_.fit(Z, rates-self.trend_.predict(Z))
        return self

    def log_rate(self, X):
        Z = self.scaler_.transform(physical_features(X))
        mean = self.trend_.predict(Z)
        sd = np.zeros(len(Z))
        if self.gp_ is not None:
            correction, sd = self.gp_.predict(Z, return_std=True)
            mean = mean + correction
        return mean, sd

    def predict(self, X):
        mean, _ = self.log_rate(X)
        return conversion(mean, np.asarray(X)[:, 2])

    def distribution(self, X):
        mean, sd = self.log_rate(X)
        t = np.asarray(X)[:, 2]
        center = conversion(mean, t)
        scale = np.maximum((conversion(mean+sd, t)-conversion(mean-sd, t))/2, 1e-6)
        q = norm.ppf(.95)
        return center, scale, conversion(mean-q*sd, t), conversion(mean+q*sd, t)


class KineticResidualGP:
    """First-order conversion trend plus GP residual in transformed feature space.

    The correction is damped to zero at t=0. Predictions are clipped to [0,1];
    monotonicity in residence time is not guaranteed (measured in the benchmark).
    The GP conditions on correlated rows, and treats the fitted trend as fixed.
    """
    def __init__(self, regularization=.01, length=1., signal=.03, noise=.015, shrink=.5):
        self.regularization, self.length = regularization, length
        self.signal, self.noise, self.shrink = signal, noise, shrink

    @staticmethod
    def _design(X):
        return np.column_stack([physical_features(X), np.log1p(np.asarray(X)[:,2])])

    @staticmethod
    def _weight(X):
        t = np.asarray(X)[:, 2]
        return t / (1+t)

    def fit(self, X, y, groups):
        X, y = np.asarray(X, float), np.asarray(y, float)
        Z = self._design(X)
        self.trend_ = KineticRegressor(self.regularization).fit(X, y, group_weights(np.asarray(groups)))
        self.scaler_ = StandardScaler().fit(Z)
        self.gp_ = residual_gp(3, self.length, self.signal, self.noise)
        self.gp_.fit(self.scaler_.transform(Z), y-self.trend_.predict(X))
        return self

    def distribution(self, X):
        Z = self.scaler_.transform(self._design(X))
        mu, sd = self.gp_.predict(Z, return_std=True)
        w = self._weight(X)*self.shrink
        raw = self.trend_.predict(X)+w*mu
        # Nominal model-only width, includes assumed residual noise.
        scale = np.maximum(w*np.sqrt(sd**2+self.noise**2), 1e-6)
        q = norm.ppf(.95)
        return np.clip(raw,0,1), scale, np.clip(raw-q*scale,0,1), np.clip(raw+q*scale,0,1)

    def predict(self, X):
        return self.distribution(X)[0]
