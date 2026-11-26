"""Synthetic mathematical benchmark; not a chemistry model."""
from .models import ProjectConfig, Factor, Target, ResultInput

def configuration(seed=42):
    return ProjectConfig(
        factors=[Factor(name=n, lower=0, upper=1) for n in ("factor_a", "factor_b", "factor_c")],
        targets=[Target(name="mfr", lower=14.5, upper=15.5, scale=1),
                 Target(name="modulus", lower=1200, upper=1350, scale=150),
                 Target(name="impact", lower=10, scale=2)],
        mode="demo", seed=seed,
    )

def evaluate(parameters):
    a, b, c = (parameters[k] for k in ("factor_a", "factor_b", "factor_c"))
    return ResultInput(parameters=parameters, outcomes={
        "mfr": 15 + 6*(a-0.65) - 3*(b-0.35) + 2*(c-0.55)**2,
        "modulus": 1275 + 380*(b-0.35) + 100*(a-0.65)**2,
        "impact": 12 - 20*(c-0.55)**2 - 6*(a-0.65)**2,
    })
