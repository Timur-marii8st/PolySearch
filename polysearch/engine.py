"""Independent GPs, constrained candidate search and marginal calibration."""
import math
import warnings

import numpy as np
from scipy.optimize import linprog
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .models import ProjectConfig, Trial, ResultInput, validate_result


class Space:
    def __init__(self, config: ProjectConfig):
        self.names = [f.name for f in config.factors]
        self.lower = np.array([f.lower for f in config.factors])
        self.width = np.array([f.upper - f.lower for f in config.factors])
        self.d = len(self.names)
        a = np.array([[c.coefficients.get(n, 0.0) for n in self.names]
                      for c in config.constraints], dtype=float).reshape(-1, self.d)
        self.a = a * self.width
        self.b = np.array([c.upper for c in config.constraints]) - a @ self.lower
        if not np.isfinite(self.a).all() or not np.isfinite(self.b).all():
            raise ValueError("Constraint scale is not finite")
        self.anchor = self.solve(np.zeros(self.d))

    def solve(self, objective):
        result = linprog(objective, A_ub=self.a if len(self.a) else None,
                         b_ub=self.b if len(self.a) else None,
                         bounds=[(0, 1)] * self.d, method="highs")
        if not result.success:
            raise ValueError("No feasible point or numerical constraint failure")
        return np.clip(result.x, 0, 1)

    def encode(self, parameters):
        return (np.array([parameters[n] for n in self.names]) - self.lower) / self.width

    def decode(self, point):
        return dict(zip(self.names, (self.lower + self.width * point).tolist()))

    def feasible(self, points):
        return ((points >= 0).all(axis=1) & (points <= 1).all(axis=1)
                & (points @ self.a.T <= self.b + 1e-9).all(axis=1))

    def pool(self, seen, rng, size, centers):
        points = [qmc.LatinHypercube(self.d, seed=int(rng.integers(2**31))).random(size)]
        if len(centers):
            points.append(np.clip(centers[rng.integers(len(centers), size=size // 2)]
                                  + rng.normal(0, 0.12, (size // 2, self.d)), 0, 1))
        candidates = np.vstack(points)
        candidates = candidates[self.feasible(candidates)]
        if len(candidates) < size // 4:
            vertices = np.vstack([self.anchor] + [
                self.solve(rng.normal(size=self.d)) for _ in range(max(16, 4 * self.d))
            ])
            weights = rng.dirichlet(np.ones(len(vertices)), size=size)
            candidates = np.vstack([candidates, vertices, weights @ vertices])
        candidates = candidates[self.feasible(candidates)]
        _, idx = np.unique(np.round(candidates, 10), axis=0, return_index=True)
        candidates = candidates[np.sort(idx)]
        for old in seen:
            candidates = candidates[np.max(np.abs(candidates - old), axis=1) > 1e-7]
        if not len(candidates):
            raise ValueError("No new feasible candidate in the sampled pool")
        if len(candidates) > size:
            candidates = candidates[rng.choice(len(candidates), size, replace=False)]
        return candidates


def loss(config, y):
    lo = np.array([t.lower if t.lower is not None else -np.inf for t in config.targets])
    hi = np.array([t.upper if t.upper is not None else np.inf for t in config.targets])
    scales = np.array([t.scale for t in config.targets])
    result = np.mean((np.maximum(lo - y, 0) + np.maximum(y - hi, 0)) / scales, axis=-1)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite loss; check target scales")
    return result


def calibrate(mean, std, observed, confidence):
    rank = math.ceil((len(observed) + 1) * confidence)
    if rank > len(observed):
        return None
    residuals = np.abs(observed - mean) / np.maximum(std, 1e-12)
    return float(np.sort(residuals)[rank - 1])


def recommend(config: ProjectConfig, trials: list[Trial], batch_size=1,
              pool_size=1536, mc_samples=128):
    if not 1 <= batch_size <= 3 or pool_size < 16 or mc_samples < 1:
        raise ValueError("Invalid search budget")
    space = Space(config)
    for trial in trials:
        validate_result(config, trial)
    rng = np.random.default_rng(config.seed)
    train = [t for t in trials if t.status == "completed" and t.role == "train"]
    cal = [t for t in trials if t.status == "completed" and t.role == "calibration"]
    seen = [space.encode(t.parameters) for t in trials]
    x = np.array([space.encode(t.parameters) for t in train]).reshape(-1, space.d)
    y = np.array([[t.outcomes[o.name] for o in config.targets] for t in train])
    distinct = len(np.unique(np.round(x, 10), axis=0))
    centers = x[np.argsort(loss(config, y))[:8]] if len(train) else x
    points = space.pool(seen, rng, pool_size, centers)
    # Validate the decoded parameters using exactly the public contract.
    valid = []
    for point in points:
        try:
            validate_result(config, ResultInput(parameters=space.decode(point), status="failed"))
            valid.append(point)
        except ValueError:
            continue
    if not valid:
        raise ValueError("No candidates passed final validation")
    points = np.array(valid)
    predictions = []
    report = {"training_trial_ids": [t.id for t in train], "model": "initial_design",
              "probability_calibrated": False, "warnings": []}
    if distinct < 3:
        # Maximin design: avoid pretending to have predictive estimates.
        score = np.min(np.linalg.norm(points[:, None] - np.array(seen)[None, :], axis=2),
                       axis=1) if seen else np.ones(len(points))
        mode = "initial_design"
    else:
        means, stds, quantiles = [], [], []
        cal = [t for t in cal if not np.any(np.max(np.abs(x - space.encode(t.parameters)),
                                                  axis=1) <= 1e-7)]
        cx = np.array([space.encode(t.parameters) for t in cal]).reshape(-1, space.d)
        for j, target in enumerate(config.targets):
            kernel = (ConstantKernel(1, (0.01, 100))
                      * Matern(np.full(space.d, 0.35), (0.04, 5), nu=2.5)
                      + WhiteKernel(0.0025, (1e-6, 0.3)))
            model = GaussianProcessRegressor(kernel=kernel, alpha=1e-8, normalize_y=True,
                                              random_state=config.seed)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(x, y[:, j])
            report["warnings"].extend(str(w.message) for w in caught)
            mu, sd = model.predict(points, return_std=True)
            sd = np.maximum(sd, 1e-12)
            means.append(mu)
            stds.append(sd)
            quantile = None
            if len(cx):
                cm, cs = model.predict(cx, return_std=True)
                quantile = calibrate(cm, cs, np.array([t.outcomes[target.name] for t in cal]),
                                     config.confidence)
            quantiles.append(quantile)
        mean = np.column_stack(means)
        std = np.column_stack(stds)
        samples = mean[:, None, :] + std[:, None, :] * rng.normal(
            size=(1, mc_samples, len(config.targets)))
        losses = loss(config, samples)
        incumbent = float(loss(config, y).min())
        probability = np.mean(losses == 0, axis=1)
        if incumbent > 0:
            score = np.maximum(incumbent - losses, 0).mean(axis=1)
            mode = "monte_carlo_expected_improvement"
        else:
            score = probability
            mode = "model_probability_of_target_hit"
        from scipy.stats import norm
        normal_quantile = float(norm.ppf((1 + config.confidence) / 2))
        for i in range(len(points)):
            props = {}
            for j, t in enumerate(config.targets):
                q = quantiles[j]
                radius = (normal_quantile if q is None else q) * std[i, j]
                props[t.name] = {
                    "mean": float(mean[i, j]), "std": float(std[i, j]),
                    "lower": float(mean[i, j] - radius), "upper": float(mean[i, j] + radius),
                    "interval_type": "gp_uncalibrated" if q is None else "split_conformal_marginal",
                }
            predictions.append({"properties": props, "model_probability": float(probability[i])})
        report.update(model="independent_gp", incumbent_loss=incumbent, calibration_count=len(cal),
                      confidence=config.confidence)
    selected, output = [], []
    remaining = np.ones(len(points), dtype=bool)
    for _ in range(min(batch_size, len(points))):
        adjusted = score.copy()
        if selected:
            distance = np.min(np.linalg.norm(points[:, None] - points[selected][None, :],
                                             axis=2), axis=1)
            if np.max(np.abs(score)) < 1e-15:
                adjusted = distance
            else:
                adjusted *= np.minimum(distance / 0.15, 1)
        adjusted[~remaining] = -np.inf
        i = int(np.argmax(adjusted))
        selected.append(i)
        remaining[i] = False
        output.append({"parameters": space.decode(points[i]), "score": float(score[i]),
                       "acquisition": mode, "prediction": predictions[i] if predictions else None,
                       "training_trial_ids": report["training_trial_ids"]})
    return {"recommendations": output, "report": report}
