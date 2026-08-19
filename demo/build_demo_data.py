"""Build one deterministic presentation replay without running an experiment batch."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from random import Random

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tetris_research.agent import LearningAgent
from tetris_research.elo import EloRatings
from tetris_research.hour4 import evaluate
from tetris_research.hour6 import Hour6Config, history_material, load_frozen_calibration
from tetris_research.hour5 import diagnose, measure_profile, normalize_profile
from tetris_research.tetris import HEIGHT, SHAPES, TetrisAdapter
from tetris_research.training import CONDITIONS, CONTROL, RATING_HISTORY, RATING_ONLY, control_material, select_rating_only_material, train

OUT = Path(__file__).with_name("demo_data.json")
REPLICATE = 0
CHALLENGE_INDEX = 0


def rows(state):
    return [[1 if state.rows[y] & (1 << x) else 0 for x in range(10)] for y in range(HEIGHT - 1, -1, -1)]


def frame(state, piece, choice, number, total_lines):
    return {"board": rows(state), "piece": piece, "placement": number,
            "action": list(choice.action), "landing_y": choice.info["landing_y"],
            "holes": int(choice.raw_features["holes"]),
            "max_height": int(choice.raw_features["max_height"]),
            "bumpiness": int(choice.raw_features["bumpiness"]), "lines": total_lines,
            "cleared_now": int(choice.info["lines_cleared"])}


def replay(agent, seed, start_state=None, limit=120, learn=False):
    adapter, rng = TetrisAdapter(), Random(seed)
    state = start_state or adapter.initial_state()
    result = [{"board": rows(state), "piece": None, "placement": 0, "holes": 0,
               "max_height": 0, "bumpiness": 0, "lines": 0, "cleared_now": 0}]
    lines = 0
    while len(result) - 1 < limit:
        piece = adapter.sample_environment(rng)
        choices = adapter.legal_actions(state, piece)
        if not choices:
            break
        choice = agent.choose(choices, learn=learn)
        state = choice.next_state
        lines += int(choice.info["lines_cleared"])
        result.append(frame(state, piece, choice, len(result), lines))
    return result


def main():
    profile_rows = list(csv.DictReader((ROOT / "artifacts/hour7/results/diagnosis_profiles.csv").open()))
    eligible = [r for r in profile_rows if r["confidence"] == "high"]
    selected = eligible[0]
    assert int(selected["replicate"]) == REPLICATE

    saved = json.loads((ROOT / "artifacts/hour7/results/challenge_results.jsonl").read_text().splitlines()[REPLICATE])
    seeds = saved["seeds"]
    config = Hour6Config(master_seed=700_007)
    calibration = load_frozen_calibration(ROOT / "artifacts/hour5/demo/calibration_parameters.json")
    baseline = LearningAgent("baseline", list(config.initial_weights), config.learning_rate,
                             config.temperature, seeds["baseline_agent"])
    baseline_record = train(baseline, control_material(), config.initial_elo,
                            config.baseline_history_placements, seeds["baseline_history"])
    history = [{"history_id": "hour7-000", "seed": seeds["baseline_history"],
                "steps": baseline_record["placements"]}]
    profile = normalize_profile(measure_profile(history, config.initial_elo, 40), calibration)
    diagnosis = diagnose(profile, config.mixed_margin)
    agents = {condition: baseline.clone(condition) for condition in CONDITIONS}
    pre = {condition: evaluate(agent, seeds["evaluation"], 120) for condition, agent in agents.items()}
    ratings = EloRatings(config.initial_elo, config.elo_k)
    for i in range(6):
        for a, b in ((CONTROL, RATING_ONLY), (CONTROL, RATING_HISTORY), (RATING_ONLY, RATING_HISTORY)):
            x, y = pre[a]["challenges"][i]["lines_cleared"], pre[b]["challenges"][i]["lines_cleared"]
            ratings.update(a, b, 1.0 if x > y else (0.0 if x < y else 0.5))
    materials = {CONTROL: control_material(),
                 RATING_ONLY: select_rating_only_material(ratings.rating(RATING_ONLY), seeds["tutorial_selection"]),
                 RATING_HISTORY: history_material(diagnosis, ratings.rating(RATING_HISTORY), seeds["tutorial_selection"])}

    # Replays consume disposable clones, preserving the exact training RNG state.
    before_frames = replay(agents[RATING_HISTORY].clone("before_replay"), seeds["evaluation"][CHALLENGE_INDEX])
    tutorial_frames = replay(agents[RATING_HISTORY].clone("tutorial_replay"), seeds["training_stream"],
                             materials[RATING_HISTORY].state, limit=16, learn=True)
    training = {condition: train(agents[condition], materials[condition], ratings.rating(condition),
                                 40, seeds["training_stream"]) for condition in CONDITIONS}
    after_frames = replay(agents[RATING_HISTORY].clone("after_replay"), seeds["evaluation"][CHALLENGE_INDEX])

    expected_pre = saved["pre"][RATING_HISTORY]["challenges"][CHALLENGE_INDEX]
    expected_post = saved["post"][RATING_HISTORY]["challenges"][CHALLENGE_INDEX]
    assert len(before_frames) - 1 == expected_pre["successful_placements"]
    assert len(after_frames) - 1 == expected_post["successful_placements"]
    assert selected["primary_weakness"] == diagnosis.primary
    assert training[RATING_HISTORY]["tutorial_board"]

    summaries = {}
    for hour in (6, 7):
        summaries[str(hour)] = json.loads((ROOT / f"artifacts/hour{hour}/results/statistical_summary.json").read_text())["paired_comparisons"]
    data = {
        "selection": {"study": "Hour 7", "replicate": REPLICATE,
                      "rule": "First high-confidence history-aware learner in saved Hour 7 replicate order; chosen before viewing replay success.",
                      "label": "Single illustrative learner — not the statistical result."},
        "profile": {"ability": float(selected["raw_ability_elo"]),
                    "hole_z": float(selected["z_hole_management"]),
                    "height_z": float(selected["z_height_management"]),
                    "surface_z": float(selected["z_surface_management"]),
                    "primary": selected["primary_weakness"], "secondary": selected["secondary_weakness"],
                    "confidence": selected["confidence"], "mixed": selected["mixed"] == "True"},
        "tutorial": {"type": selected["selected_tutorial"], "difficulty": materials[RATING_HISTORY].difficulty,
                     "rationale": materials[RATING_HISTORY].rationale, "frames": tutorial_frames},
        "challenge_seed": seeds["evaluation"][CHALLENGE_INDEX],
        "before": before_frames, "after": after_frames, "results": summaries,
    }
    OUT.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    print(f"Wrote {OUT} ({len(before_frames)-1} before, {len(after_frames)-1} after placements)")


if __name__ == "__main__":
    main()
