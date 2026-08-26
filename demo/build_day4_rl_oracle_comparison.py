"""Build a four-board Day 4 replay with the frozen Day 2 reference oracle."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tetris_research.expert import (  # noqa: E402
    DELLACHERIE_WEIGHTS, DellacherieSearchExpert,
)
from tetris_research.tetris import TetrisAdapter, TetrisState  # noqa: E402

THREE_BOARD_BUILDER = Path(__file__).with_name("build_day4_rl_comparison.py")
SPEC = importlib.util.spec_from_file_location("day4_three_board_builder", THREE_BOARD_BUILDER)
if not SPEC or not SPEC.loader:
    raise RuntimeError("could not load the existing Day 4 replay builder")
three_board = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(three_board)

OUTPUT = Path(__file__).with_name("day4_rl_oracle_comparison.json")
ORACLE_KEY = "oracle"
ORACLE_BEAM_WIDTH = 3
ORACLE_CONTINUATION_DISCOUNT = 0.35


def oracle_replay(stream: list[str]) -> dict[str, Any]:
    oracle = DellacherieSearchExpert(
        beam_width=ORACLE_BEAM_WIDTH,
        continuation_discount=ORACLE_CONTINUATION_DISCOUNT,
    )
    game, state, lines = TetrisAdapter(), TetrisState(), 0
    frames = [{"board": three_board.board(state), "placements": 0, "lines": 0,
               "piece": None, "action": None, "cleared_now": 0}]
    for piece in stream:
        legal = game.legal_actions(state, piece)
        if not legal:
            break
        placement = oracle.preferred_placement(state, piece, legal).placement
        selected = next(choice for choice in legal if tuple(choice.action) ==
                        (placement.rotation, placement.x))
        cleared = int(selected.info["lines_cleared"])
        lines += cleared
        state = selected.next_state
        frames.append({"board": three_board.board(state), "placements": len(frames),
                       "lines": lines, "piece": piece,
                       "action": [placement.rotation, placement.x],
                       "cleared_now": cleared})
    return {"frames": frames, "placements": len(frames) - 1, "lines": lines,
            "status": "cap" if len(frames) - 1 == len(stream) else "game_over",
            "expert_id": oracle.expert_id, "beam_width": oracle.beam_width,
            "continuation_discount": oracle.continuation_discount}


def build() -> dict[str, Any]:
    data = three_board.build()
    learner_states_before = hashlib.sha256(json.dumps(
        data["policies"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    data["policies"][ORACLE_KEY] = oracle_replay(data["piece_stream"])
    learner_states_after = hashlib.sha256(json.dumps(
        {key: data["policies"][key] for key in map(str, three_board.EXPOSURES)},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if learner_states_before != learner_states_after:
        raise AssertionError("oracle replay altered learner replay state")
    data["title"] = "Adaptive Tetris Tutor — Learning from Play"
    data["subtitle"] = "Same test, different amounts of prior training, plus the fixed reference oracle"
    data["policy_order"] = [*map(str, three_board.EXPOSURES), ORACLE_KEY]
    data["integrity"].update({
        "all_start_empty": True,
        "all_capped_at": data["demo_cap"],
        "oracle_implementation": "tetris_research.expert.DellacherieSearchExpert",
        "oracle_version": DellacherieSearchExpert.VERSION,
        "oracle_beam_width": ORACLE_BEAM_WIDTH,
        "oracle_continuation_discount": ORACLE_CONTINUATION_DISCOUNT,
        "oracle_weights_sha256": hashlib.sha256(json.dumps(
            DELLACHERIE_WEIGHTS, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "oracle_altered_learner_state": False,
    })
    return data


def verify(data: dict[str, Any]) -> None:
    three_board.verify(data)
    if data["seed"] != 2026082801 or data["demo_cap"] != 300:
        raise AssertionError("incorrect seed or test cap")
    if data["policy_order"] != ["0", "750", "3000", ORACLE_KEY]:
        raise AssertionError("incorrect policy order")
    empty = ["0000000000"] * 20
    if any(policy["frames"][0]["board"] != empty or
           policy["frames"][0]["placements"] != 0
           for policy in data["policies"].values()):
        raise AssertionError("every board must start empty")
    if any(policy["placements"] > data["demo_cap"]
           for policy in data["policies"].values()):
        raise AssertionError("a board exceeded the test cap")
    for policy in data["policies"].values():
        if len(policy["frames"]) != policy["placements"] + 1:
            raise AssertionError("missing recorded placement frame")
        for index, frame in enumerate(policy["frames"][1:]):
            if frame["piece"] != data["piece_stream"][index]:
                raise AssertionError("policy did not consume the shared indexed stream")
    oracle = data["policies"][ORACLE_KEY]
    if (oracle["expert_id"] != DellacherieSearchExpert.VERSION or
            oracle["beam_width"] != ORACLE_BEAM_WIDTH or
            oracle["continuation_discount"] != ORACLE_CONTINUATION_DISCOUNT):
        raise AssertionError("oracle configuration differs from the Day 2 reference")
    if data["integrity"]["oracle_altered_learner_state"]:
        raise AssertionError("oracle altered learner state")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    data = build()
    verify(data)
    if not args.verify_only:
        OUTPUT.write_text(json.dumps(data, separators=(",", ":")) + "\n")
        print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    print("Verified seed 2026082801; " + ", ".join(
        f"{key}: {data['policies'][key]['placements']} placements / "
        f"{data['policies'][key]['lines']} lines ({data['policies'][key]['status']})"
        for key in data["policy_order"]))


if __name__ == "__main__":
    main()
