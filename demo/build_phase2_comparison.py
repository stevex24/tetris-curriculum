"""Build a replay from the frozen Phase 2 policy snapshots (no learning)."""
from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tetris_research.richer_student import RicherRLStudent  # noqa: E402
from tetris_research.tetris import SHAPES, TetrisAdapter, TetrisState  # noqa: E402

RESULTS = ROOT / "experiments/phase2/final_results.json"
OUTPUT = Path(__file__).with_name("phase2_comparison.json")
REPLICATE = 0


def board(state: TetrisState) -> list[str]:
    return [format(row, "010b") for row in reversed(state.rows)]


def replay(snapshot: dict, stream: list[str], cap: int) -> dict:
    frozen = copy.deepcopy(snapshot)
    frozen["agent_id"] = f"{snapshot['agent_id']}-phase2-demo"
    frozen["learning_enabled"] = False
    student = RicherRLStudent.from_state(frozen)
    before = copy.deepcopy(student.serialize_state())
    game, state, lines = TetrisAdapter(), TetrisState(), 0
    frames = [{"board": board(state), "placements": 0, "lines": 0,
               "piece": None, "action": None, "cleared_now": 0}]
    for piece in stream[:cap]:
        legal = game.legal_actions(state, piece)
        if not legal:
            break
        decision = student.choose_placement(state, piece, legal,
                                            learn=False, deterministic=True)
        cleared = int(decision.evaluation.info["lines_cleared"])
        lines += cleared
        state = decision.evaluation.next_state
        frames.append({"board": board(state), "placements": len(frames),
                       "lines": lines, "piece": piece,
                       "action": list(decision.evaluation.action),
                       "cleared_now": cleared})
    if student.serialize_state() != before:
        raise AssertionError("frozen replay mutated a policy")
    placements = len(frames) - 1
    return {"frames": frames, "placements": placements, "lines": lines,
            "status": "cap" if placements == cap else "game_over",
            "learning_enabled": student.learning_enabled}


def build() -> dict:
    saved = json.loads(RESULTS.read_text())
    config = saved["configuration"]
    seed = int(config["evaluation_game_seeds"][0])
    cap = int(config["evaluation_maximum"])
    replicate = saved["replicates"][REPLICATE]
    recorded = next(row for row in replicate["game_rows"] if row["seed"] == seed)
    rng = random.Random(seed)
    stream = [rng.choice(tuple(SHAPES)) for _ in range(cap)]
    stream_hash = hashlib.sha256("".join(stream).encode()).hexdigest()
    if stream_hash != recorded["stream_sha256"]:
        raise AssertionError("stream does not match frozen evaluation row")
    order = ("ordinary", "imitation", "static_personalized", "responsive_personalized")
    labels = {"ordinary": "Ordinary", "imitation": "Direct Imitation",
              "static_personalized": "Static Personalized",
              "responsive_personalized": "Responsive Personalized"}
    policies = {}
    for name in order:
        replayed = replay(replicate["trained_states"][name], stream, cap)
        expected = recorded[name]
        if (replayed["placements"], replayed["lines"]) != (expected["placements"], expected["lines_cleared"]):
            raise AssertionError(f"{name} replay does not match frozen metrics")
        policies[name] = {"label": labels[name],
                          "rating": replicate["ratings"][name],
                          "rating_gain": replicate["rating_gains"][name],
                          **replayed}
    return {"title": "Phase 2 Adaptive Tetris Tutor",
            "subtitle": "Four frozen policies on one held-out stream",
            "policy_order": list(order), "replicate": REPLICATE,
            "seed": seed, "cap": cap, "piece_stream": stream,
            "piece_stream_sha256": stream_hash, "policies": policies,
            "integrity": {"same_piece_stream": True, "learning_disabled": True,
                          "expert_invoked": False, "seed_rule": "first evaluation_game_seed",
                          "source": "experiments/phase2/final_results.json"}}


if __name__ == "__main__":
    data = build()
    OUTPUT.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}; seed {data['seed']}; stream {data['piece_stream_sha256']}")
