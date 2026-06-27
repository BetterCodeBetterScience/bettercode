"""A tour of dataclass features, built around a small scientific example.

Demonstrates: the methods generated for free; type annotations that define
fields but are not enforced at runtime; safe handling of mutable defaults;
validation and a derived field in __post_init__; frozen (immutable, hashable)
instances; the asdict/replace helpers; ordering and slots; and type-aware
equality.
"""

from dataclasses import asdict, dataclass, field, replace
from pydantic import BaseModel, ConfigDict, Field, EmailStr
from typing import Optional, List


# === 1. The basics: declare fields, get __init__/__repr__/__eq__ for free ====
@dataclass
class Measurement:
    subject_id: str
    value: float
    unit: str = "mV"  # a default
    tags: list[str] = field(default_factory=list)  # mutable default, done right

    def is_outlier(self, threshold: float) -> bool:  # still an ordinary class
        return abs(self.value) > threshold


# === 2. Validation + a derived field on a frozen (immutable) record ==========
@dataclass(frozen=True)
class ExperimentConfig:
    n_subjects: int
    n_conditions: int
    seed: int = 42
    n_total: int = field(init=False)  # derived; not supplied by the caller

    def __post_init__(self) -> None:
        if self.n_subjects <= 0:
            raise ValueError(f"n_subjects must be positive, got {self.n_subjects}")
        # frozen blocks normal assignment, so the derived field is set this way:
        object.__setattr__(self, "n_total", self.n_subjects * self.n_conditions)


# === 3. Ordering and slots ===================================================
@dataclass(order=True, slots=True)
class Sample:
    quality: float  # comparisons compare fields top-to-bottom, so quality leads
    name: str


# === 4. Type-aware equality: same fields, different type, not equal ==========
@dataclass
class Point2D:
    x: float
    y: float


@dataclass
class Vector2D:
    x: float
    y: float


class DatabaseConfig(BaseModel):
    host: str
    port: int = Field(ge=1, le=65535, default=5432)

    model_config = ConfigDict(str_strip_whitespace=True, populate_by_name=True)


class AppConfig(BaseModel):
    app_name: str
    debug_mode: bool = False
    database: DatabaseConfig
    timeout_seconds: int = Field(ge=1, le=300, default=30)

    model_config = ConfigDict(
        extra="forbid",  # Reject extra fields
        str_strip_whitespace=True,
    )


class UserProfile(BaseModel):
    model_config = ConfigDict(
        frozen=True,  # instance is immutable
        str_strip_whitespace=True,  # Cleans "  admin  " -> "admin"
        extra="forbid",  # Fails if JSON has unknown keys
    )

    username: str
    email: EmailStr  # Built-in email validation


def _demo() -> None:
    # --- generated methods ---
    m1 = Measurement("S01", 0.42, tags=["clean"])
    m2 = Measurement("S01", 0.42, tags=["clean"])
    print("repr:                  ", m1)  # labeled __repr__ for free
    print("value equality:        ", m1 == m2)  # field-by-field __eq__ for free
    print("method works:          ", m1.is_outlier(0.3))

    # --- annotations are NOT enforced at runtime (use pydantic if you need that) ---
    sketchy = Measurement("S02", value="not a number")  # runs without error
    print("unenforced annotation: ", repr(sketchy.value))

    # --- a mutable default is per-instance, never shared ---
    a, b = Measurement("S03", 1.0), Measurement("S04", 2.0)
    a.tags.append("flagged")
    print("independent defaults:  ", a.tags, b.tags)  # b.tags stays empty

    # --- frozen: validation, derived field, immutability, hashability ---
    config = ExperimentConfig(n_subjects=40, n_conditions=3)
    print("derived field n_total: ", config.n_total)  # 120
    try:
        config.seed = 0  # frozen -> blocked
    except Exception as exc:
        print("immutable:             ", type(exc).__name__)
    try:
        ExperimentConfig(n_subjects=-1, n_conditions=3)
    except ValueError as exc:
        print("validated:             ", exc)
    runs = {config: "baseline"}  # frozen -> hashable, usable as a dict key
    print("usable as dict key:    ", runs[config])

    # --- helpers: asdict (serialize) and replace (functional-style update) ---
    print("asdict:                ", asdict(config))
    bigger = replace(config, n_subjects=80)  # new instance; original untouched
    print(
        "replace:               ",
        f"new n_total={bigger.n_total}, original n_subjects={config.n_subjects}",
    )

    # --- order=True enables sorting; slots=True blocks stray attributes ---
    samples = [Sample(0.9, "b"), Sample(0.2, "a"), Sample(0.7, "c")]
    print("sorted by quality:     ", [s.name for s in sorted(samples)])
    try:
        samples[0].quallity = 1.0  # typo: an undeclared attribute
    except AttributeError as exc:
        print("slots caught typo:     ", type(exc).__name__)

    # --- type-aware equality ---
    print("Point2D == Vector2D:   ", Point2D(1, 2) == Vector2D(1, 2))

    # pydantic example
    from pydantic import BaseModel, Field
    from typing import List

    class Measurement(BaseModel):
        subject_id: str
        value: float = Field(ge=-1000, le=1000)
        unit: str = "mV"
        tags: List[str] = []

    # Valid
    m1 = Measurement(subject_id="A001", value=50.0)
    print(m1.value)  # 50.0

    # Invalid - raises PydanticValidationError
    try:
        m2 = Measurement(subject_id="A001", value=1500.0)
    except Exception as e:
        print(e)  # PydanticValidationError: value must be <= 1000

    # config example


if __name__ == "__main__":
    _demo()
