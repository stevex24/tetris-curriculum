from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from random import Random
from statistics import mean, stdev
from typing import Any, Iterable

from .agent import LearningAgent
from .training import _history_board, board_rows, control_material, train

DIAGNOSTIC_DIMENSIONS = ("hole_management", "height_management", "surface_management")
CALIBRATION_SEED = 500_005
DIAGNOSTIC_POPULATION_SEED = 500_105


@dataclass(frozen=True)
class LearnerProfile:
    """Observable play summary; policy parameters are intentionally not representable."""

    ability_elo: float
    mean_holes: float
    mean_max_height: float
    mean_bumpiness: float
    line_clear_rate: float
    lines_cleared: int
    placements_observed: int
    normalized_weakness: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Calibration:
    seed: int
    population: str
    placements_per_history: int
    dimensions: tuple[str, ...]
    means: dict[str, float]
    sample_sds: dict[str, float]
    history_ids: tuple[str, ...]


@dataclass(frozen=True)
class Diagnosis:
    primary: str
    secondary: str
    margin: float
    confidence: str
    mixed: bool
    normalized_weakness: dict[str, float]


def _steps(records: Iterable[dict[str, Any]], recent_steps: int = 40) -> list[dict[str, Any]]:
    return [step for record in records for step in record.get("steps", [])][-recent_steps:]


def measure_profile(records: Iterable[dict[str, Any]], ability_elo: float,
                    recent_steps: int = 40) -> LearnerProfile:
    rows = _steps(records, recent_steps)
    if not rows:
        return LearnerProfile(ability_elo, 0.0, 0.0, 0.0, 0.0, 0, 0)
    values = lambda key: [float(row["features"][key]) for row in rows]
    lines = int(sum(values("lines_cleared")))
    return LearnerProfile(ability_elo=ability_elo, mean_holes=mean(values("holes")),
                          mean_max_height=mean(values("max_height")),
                          mean_bumpiness=mean(values("bumpiness")),
                          line_clear_rate=lines / len(rows), lines_cleared=lines,
                          placements_observed=len(rows))


def _raw_dimensions(profile: LearnerProfile) -> dict[str, float]:
    return {"hole_management": profile.mean_holes,
            "height_management": profile.mean_max_height,
            "surface_management": profile.mean_bumpiness}


def fit_calibration(profiles: Iterable[tuple[str, LearnerProfile]], seed: int,
                    population: str, placements_per_history: int) -> Calibration:
    rows = list(profiles)
    if len(rows) < 2 or len({identifier for identifier, _ in rows}) != len(rows):
        raise ValueError("calibration needs at least two uniquely identified histories")
    columns = {key: [_raw_dimensions(profile)[key] for _, profile in rows]
               for key in DIAGNOSTIC_DIMENSIONS}
    sds = {key: stdev(values) for key, values in columns.items()}
    if any(value <= 0 for value in sds.values()):
        raise ValueError("calibration population has no variation in a diagnostic dimension")
    return Calibration(seed, population, placements_per_history, DIAGNOSTIC_DIMENSIONS,
                       {key: mean(values) for key, values in columns.items()}, sds,
                       tuple(identifier for identifier, _ in rows))


def normalize_profile(profile: LearnerProfile, calibration: Calibration) -> LearnerProfile:
    """A score of +1 means one calibration-population SD worse than its mean."""
    raw = _raw_dimensions(profile)
    scores = {key: (raw[key] - calibration.means[key]) / calibration.sample_sds[key]
              for key in calibration.dimensions}
    return LearnerProfile(**{**asdict(profile), "normalized_weakness": scores})


def diagnose(profile: LearnerProfile, mixed_margin: float = 0.35) -> Diagnosis:
    if profile.placements_observed == 0 or set(profile.normalized_weakness) != set(DIAGNOSTIC_DIMENSIONS):
        raise ValueError("diagnosis requires observed placements and all normalized dimensions")
    ranked = sorted(DIAGNOSTIC_DIMENSIONS,
                    key=lambda key: profile.normalized_weakness[key], reverse=True)
    margin = profile.normalized_weakness[ranked[0]] - profile.normalized_weakness[ranked[1]]
    mixed = margin < mixed_margin
    confidence = "low/mixed" if mixed else ("moderate" if margin < 0.75 else "high")
    return Diagnosis(ranked[0], ranked[1], margin, confidence, mixed,
                     dict(profile.normalized_weakness))


def tutorial_for(diagnosis: Diagnosis, ability_elo: float, seed: int) -> dict[str, Any]:
    mapping = {"hole_management": "hole_avoidance", "height_management": "stack_height",
               "surface_management": "bumpiness"}
    tutorial = mapping[diagnosis.primary]
    difficulty = "introductory" if ability_elo < 1400 else ("intermediate" if ability_elo < 1600 else "advanced")
    return {"tutorial_type": tutorial, "difficulty": difficulty,
            "board": board_rows(_history_board(tutorial, difficulty, seed)),
            "selection_basis": diagnosis.primary,
            "rationale": {"hole_management": "Practice exposes choices that create or avoid cavities.",
                          "height_management": "Practice makes stack reduction and vertical risk salient.",
                          "surface_management": "Practice contrasts smooth and sharply uneven landing surfaces."}[diagnosis.primary]}


def derived_seed(master_seed: int, population: str, index: int, domain: str) -> int:
    digest = hashlib.sha256(f"hour5:{master_seed}:{population}:{index}:{domain}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7fffffff


def natural_profiles(master_seed: int, population: str, count: int,
                     placements: int = 40) -> list[tuple[str, LearnerProfile, list[dict[str, Any]]]]:
    results = []
    for index in range(count):
        agent_seed = derived_seed(master_seed, population, index, "agent")
        history_seed = derived_seed(master_seed, population, index, "history")
        agent = LearningAgent(f"{population}-{index:03d}", [0.20, 0.10, 0.15, 0.25],
                              0.04, 0.25, agent_seed)
        record = train(agent, control_material(), 1500.0, placements, history_seed)
        history = [{"history_id": agent.name, "steps": record["placements"]}]
        results.append((agent.name, measure_profile(history, 1500.0, placements), history))
    return results


def synthetic_history(kind: str, seed: int, placements: int = 40) -> list[dict[str, Any]]:
    """Predeclared construct probes; labels control generation but are absent from step data."""
    centers = {
        "balanced": (31.7, 11.9, 18.2),
        "hole_management": (43.0, 11.9, 18.2),
        "height_management": (31.7, 15.5, 18.2),
        "surface_management": (31.7, 11.9, 28.0),
        "mixed_hole_height": (40.0, 13.45, 18.2),
    }
    if kind not in centers:
        raise ValueError(f"unknown synthetic history kind: {kind}")
    rng, center = Random(seed), centers[kind]
    steps = []
    for _ in range(placements):
        holes = max(0.0, center[0] + rng.choice((-2, -1, 0, 1, 2)))
        height = max(1.0, center[1] + rng.choice((-0.5, 0.0, 0.5)))
        bump = max(0.0, center[2] + rng.choice((-2, -1, 0, 1, 2)))
        steps.append({"features": {"holes": holes, "max_height": height,
                                    "bumpiness": bump, "lines_cleared": 0.0}})
    return [{"history_id": f"construct-{kind}-{seed}", "steps": steps}]


def pearson_matrix(profiles: Iterable[LearnerProfile]) -> dict[str, dict[str, float]]:
    rows = list(profiles)
    columns = {key: [_raw_dimensions(row)[key] for row in rows] for key in DIAGNOSTIC_DIMENSIONS}
    result = {}
    for a, x in columns.items():
        result[a] = {}
        for b, y in columns.items():
            mx, my = mean(x), mean(y)
            numerator = sum((u - mx) * (v - my) for u, v in zip(x, y))
            denominator = math.sqrt(sum((u - mx) ** 2 for u in x) * sum((v - my) ** 2 for v in y))
            result[a][b] = numerator / denominator if denominator else float("nan")
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
