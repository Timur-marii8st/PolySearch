"""Validated numerical experiment contracts."""
import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator

Name = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")]
RESERVED = {"status", "role", "note", "id", "recommendation_id", "source"}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Factor(StrictModel):
    name: Name
    lower: FiniteFloat
    upper: FiniteFloat
    unit: str = ""

    @model_validator(mode="after")
    def bounds(self):
        if not self.lower < self.upper or self.upper - self.lower == float("inf"):
            raise ValueError("Factor must have a finite positive range")
        return self


class Target(StrictModel):
    name: Name
    lower: FiniteFloat | None = None
    upper: FiniteFloat | None = None
    scale: Annotated[FiniteFloat, Field(gt=0)]
    unit: str = ""

    @model_validator(mode="after")
    def bounds(self):
        if self.lower is None and self.upper is None:
            raise ValueError("A target needs at least one bound")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("Invalid target interval")
        return self


class Constraint(StrictModel):
    coefficients: dict[Name, FiniteFloat]
    upper: FiniteFloat


class ProjectConfig(StrictModel):
    factors: list[Factor] = Field(min_length=1, max_length=12)
    targets: list[Target] = Field(min_length=1, max_length=8)
    constraints: list[Constraint] = Field(default_factory=list, max_length=30)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    confidence: float = Field(default=0.9, gt=0, lt=1)
    mode: Literal["real", "demo"] = "real"

    @model_validator(mode="after")
    def names(self):
        factors = {f.name for f in self.factors}
        names = [f.name for f in self.factors] + [t.name for t in self.targets]
        if len(set(names)) != len(names) or set(names) & RESERVED:
            raise ValueError("Factor and target names must be unique and non-reserved")
        for constraint in self.constraints:
            if not constraint.coefficients or not set(constraint.coefficients) <= factors:
                raise ValueError("Unknown or empty constraint coefficients")
        return self


class ResultInput(StrictModel):
    parameters: dict[Name, FiniteFloat]
    outcomes: dict[Name, FiniteFloat] = Field(default_factory=dict)
    status: Literal["completed", "failed"] = "completed"
    role: Literal["train", "calibration"] = "train"
    recommendation_id: str | None = None
    note: str = Field(default="", max_length=2000)


class Trial(ResultInput):
    id: str
    source: Literal["manual", "csv", "synthetic"]
    created_at: str


def validate_result(config: ProjectConfig, result: ResultInput) -> None:
    if set(result.parameters) != {f.name for f in config.factors}:
        raise ValueError("Parameters must match the configured factors")
    for f in config.factors:
        if not f.lower <= result.parameters[f.name] <= f.upper:
            raise ValueError(f"Factor outside bounds: {f.name}")
    for c in config.constraints:
        value = sum(w * result.parameters[k] for k, w in c.coefficients.items())
        if not math.isfinite(value) or value > c.upper + 1e-9:
            raise ValueError("Linear constraint violated")
    targets = {t.name for t in config.targets}
    if not set(result.outcomes) <= targets:
        raise ValueError("Unknown target")
    if result.status == "completed" and set(result.outcomes) != targets:
        raise ValueError("Completed trials require all targets")
    if result.status == "failed" and result.role == "calibration":
        raise ValueError("Failed trials cannot be calibration data")
