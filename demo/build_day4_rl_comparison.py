"""Build the frozen Day 4 classroom replay from committed checkpoint states."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tetris_research.richer_student import RicherRLStudent  # noqa: E402
from tetris_research.tetris import SHAPES, TetrisAdapter, TetrisState  # noqa: E402


RESULTS = ROOT / "experiments/day4/final_results.json"
PREREGISTRATION = ROOT / "experiments/day4/preregistration.json"
OUTPUT = Path(__file__).with_name("day4_rl_comparison.json")
SEED = 2026082801
EXPOSURES = (0, 750, 3000)


def board(state: TetrisState) -> list[str]:
    """Return display rows from ceiling to floor."""
    return [format(row, "010b") for row in reversed(state.rows)]


def replay(snapshot: dict[str, Any], stream: list[str]) -> dict[str, Any]:
    frozen = copy.deepcopy(snapshot)
    frozen["agent_id"] = f"{snapshot['agent_id']}-day4-demo"
    frozen["learning_enabled"] = False
    student = RicherRLStudent.from_state(frozen)
    initial_agent_state = copy.deepcopy(student.serialize_state())
    game, state, lines = TetrisAdapter(), TetrisState(), 0
    frames = [{"board": board(state), "placements": 0, "lines": 0,
               "piece": None, "action": None, "cleared_now": 0}]

    for piece in stream:
        legal = game.legal_actions(state, piece)
        if not legal:
            break
        decision = student.choose_placement(
            state, piece, legal, learn=False, deterministic=True
        )
        cleared = int(decision.evaluation.info["lines_cleared"])
        lines += cleared
        state = decision.evaluation.next_state
        frames.append({"board": board(state), "placements": len(frames),
                       "lines": lines, "piece": piece,
                       "action": list(decision.evaluation.action),
                       "cleared_now": cleared})

    if student.serialize_state() != initial_agent_state:
        raise AssertionError("replay mutated a frozen policy")
    return {"frames": frames, "placements": len(frames) - 1, "lines": lines,
            "status": "cap" if len(frames) - 1 == len(stream) else "game_over",
            "learning_enabled": student.learning_enabled,
            "source_agent_id": snapshot["agent_id"]}


def build() -> dict[str, Any]:
    saved = json.loads(RESULTS.read_text())
    preregistered = json.loads(PREREGISTRATION.read_text())
    held_out = preregistered["held_out_game_seeds"]
    if SEED not in held_out:
        raise AssertionError("demo seed is not preregistered")
    if SEED != held_out[0]:
        raise AssertionError("demo must use the first preregistered held-out seed")

    maximum = int(saved["configuration"]["evaluation_maximum"])
    rng = random.Random(SEED)
    stream = [rng.choice(tuple(SHAPES)) for _ in range(maximum)]
    stream_hash = hashlib.sha256("".join(stream).encode()).hexdigest()
    recorded = next(row for row in saved["game_rows"] if row["seed"] == SEED)
    if stream_hash != recorded["stream_sha256"]:
        raise AssertionError("held-out piece stream does not match Day 4")

    policies = {}
    for exposure in EXPOSURES:
        key = f"rl@{exposure}"
        policies[str(exposure)] = replay(saved["evaluation_before_states"][key], stream)
        actual = policies[str(exposure)]
        expected = recorded[key]
        if (actual["placements"], actual["lines"]) != (
                expected["placements"], expected["lines_cleared"]):
            raise AssertionError(f"{key} replay does not match committed metrics")

    return {
        "title": "Adaptive Tetris Tutor — Learning from Play",
        "subtitle": "Same student architecture, same held-out piece sequence, different amounts of prior RL training",
        "seed": SEED, "demo_cap": maximum, "exposures": list(EXPOSURES),
        "piece_stream": stream, "piece_stream_sha256": stream_hash,
        "policies": policies,
        "integrity": {
            "snapshot_source": "experiments/day4/final_results.json evaluation_before_states",
            "same_piece_stream": True, "learning_disabled": True,
            "expert_invoked": False, "seed_is_preregistered": True,
            "recorded_metrics_matched": True,
        },
    }


def verify(data: dict[str, Any]) -> None:
    if data["exposures"] != list(EXPOSURES):
        raise AssertionError("incorrect exposure labels")
    if not all(data["policies"][str(x)]["learning_enabled"] is False
               for x in EXPOSURES):
        raise AssertionError("a replay policy has learning enabled")
    integrity = data["integrity"]
    if not (integrity["same_piece_stream"] and integrity["learning_disabled"]
            and not integrity["expert_invoked"] and integrity["seed_is_preregistered"]):
        raise AssertionError("replay integrity check failed")
    expected_hash = hashlib.sha256("".join(data["piece_stream"]).encode()).hexdigest()
    if data["piece_stream_sha256"] != expected_hash:
        raise AssertionError("piece stream hash mismatch")
    for policy in data["policies"].values():
        for index, frame in enumerate(policy["frames"][1:]):
            if frame["piece"] != data["piece_stream"][index]:
                raise AssertionError("policy did not consume the shared indexed stream")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    data = build()
    verify(data)
    if not args.verify_only:
        OUTPUT.write_text(json.dumps(data, separators=(",", ":")) + "\n")
        print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    counts = ", ".join(
        f"{x}: {data['policies'][str(x)]['placements']} placements / "
        f"{data['policies'][str(x)]['lines']} lines" for x in EXPOSURES
    )
    print(f"Verified seed {SEED}; {counts}")


if __name__ == "__main__":
    main()
