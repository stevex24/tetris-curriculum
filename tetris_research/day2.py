"""Predeclared Day-2 expert validation and matched strength benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .agent import LearningAgent
from .expert import DellacherieSearchExpert, PlacementEvaluation
from .legacy_student import FourFeatureStudentAdapter
from .student import Placement
from .tetris import HEIGHT, SHAPES, WIDTH, TetrisAdapter, TetrisState

ROOT = Path(__file__).resolve().parents[1]
DAY2 = ROOT / "experiments/day2"


def validation_corpus() -> list[dict[str, Any]]:
    """Fixed corpus declared by construction, not selected using policy outcomes."""
    boards = [
        ("empty", "T", []),
        ("ordinary_stacking", "L", [0b1110000111, 0b1100000011, 0b1000000001]),
        ("holes", "I", [0b1111111111, 0b1111011111, 0b1110011111]),
        ("deep_well", "I", [0b1111011111] * 4),
        ("uneven_surface", "S", [0b1111111111, 0b0011111100, 0b0001100000, 0b0000100000]),
        ("high_stack", "J", [0b1110111111] * 10 + [0b0010000000] * 5),
        ("recovery", "T", [0b1111101111, 0b1100100111, 0b1000000011, 0b1001000011]),
        ("single_clear", "I", [0b1111110000]),
        ("tetris_ready", "I", [0b1111111110] * 4),
        ("lookahead_ambiguous", "Z", [0b1110001111, 0b1100000111]),
        ("staircase", "J", [sum(1 << x for x in range(y + 1, WIDTH)) for y in range(6)]),
        ("covered_cells", "O", [0b1111111111, 0b1101111011, 0b1100000011]),
    ]
    corpus = []
    for name, piece, lower_rows in boards:
        rows = tuple(lower_rows) + (0,) * (HEIGHT - len(lower_rows))
        corpus.append({"id": name, "piece": piece, "rows": list(rows)})
    return corpus


def _placement(item: PlacementEvaluation) -> list[int]:
    return [item.placement.rotation, item.placement.x]


def _independent_legal(rows: Sequence[int], piece: str) -> dict[tuple[int, int], tuple[int, ...]]:
    """Canonical drop/replay implementation independent of TetrisAdapter."""
    output = {}
    full = (1 << WIDTH) - 1
    for rotation, cells in enumerate(SHAPES[piece]):
        width = max(dx for dx, _ in cells) + 1
        for x in range(WIDTH - width + 1):
            y = HEIGHT
            def collision(at_y: int) -> bool:
                return any(at_y + dy < 0 or
                           (at_y + dy < HEIGHT and rows[at_y + dy] & (1 << (x + dx)))
                           for dx, dy in cells)
            while y > 0 and not collision(y - 1):
                y -= 1
            if collision(y) or any(y + dy >= HEIGHT for _, dy in cells):
                continue
            result = list(rows)
            for dx, dy in cells:
                result[y + dy] |= 1 << (x + dx)
            cleared = sum(row == full for row in result)
            result = [row for row in result if row != full] + [0] * cleared
            output[(rotation, x)] = tuple(result)
    return output


def validate_expert(expert: Any | None = None) -> dict[str, Any]:
    expert = expert or DellacherieSearchExpert()
    game = TetrisAdapter()
    checks = {name: True for name in ("rules_compatible", "recommended_placements_legal",
              "deterministic_rankings", "board_transitions_match", "rankings_consistent",
              "best_action_zero_regret", "inferior_actions_nonnegative_regret")}
    positions = []
    for item in validation_corpus():
        state, piece = TetrisState(tuple(item["rows"])), item["piece"]
        canonical = game.legal_actions(state, piece)
        independent = _independent_legal(state.rows, piece)
        adapter_map = {(int(a.action[0]), int(a.action[1])): a for a in canonical}
        checks["rules_compatible"] &= set(independent) == set(adapter_map)
        ranking = tuple(expert.rank_placements(state, piece, canonical, deterministic=True))
        rerun = tuple(expert.rank_placements(state, piece, canonical, deterministic=True))
        signature = [(_placement(x), x.value, x.rank) for x in ranking]
        checks["deterministic_rankings"] &= signature == [(_placement(x), x.value, x.rank) for x in rerun]
        placements = [(x.placement.rotation, x.placement.x) for x in ranking]
        checks["recommended_placements_legal"] &= bool(ranking) and all(p in independent for p in placements)
        checks["rankings_consistent"] &= ([x.rank for x in ranking] == list(range(1, len(ranking) + 1))
                                           and [x.value for x in ranking] == sorted((x.value for x in ranking), reverse=True))
        checks["board_transitions_match"] &= all(
            independent[p] == adapter_map[p].next_state.rows for p in placements if p in independent)
        if ranking:
            best_regret = expert.regret(state, piece, canonical, ranking[0].placement).regret
            checks["best_action_zero_regret"] &= abs(best_regret) < 1e-9
            checks["inferior_actions_nonnegative_regret"] &= all(
                expert.regret(state, piece, canonical, x.placement).regret >= -1e-9 for x in ranking)
        legacy = FourFeatureStudentAdapter(LearningAgent("legacy", [.2, .1, .15, .25], seed=0))
        legacy_choice = legacy.choose_placement(state, piece, canonical, deterministic=True).placement
        positions.append({**item, "legal_count": len(canonical), "legacy": [legacy_choice.rotation, legacy_choice.x],
                          "expert": _placement(ranking[0]), "legacy_regret": expert.regret(
                              state, piece, canonical, legacy_choice).regret,
                          "ranking": [{"placement": _placement(x), "value": x.value, "rank": x.rank,
                                       "metadata": x.metadata} for x in ranking]})
    return {"expert": expert.expert_id, "checks": {k: "PASS" if v else "FAIL" for k, v in checks.items()},
            "overall": "PASS" if all(checks.values()) else "FAIL", "positions": positions}


def _metrics(state: TetrisState) -> dict[str, int]:
    raw = TetrisAdapter._features(state, 0)
    heights = []
    for x in range(WIDTH):
        occupied = [y for y, row in enumerate(state.rows) if row & (1 << x)]
        heights.append(max(occupied) + 1 if occupied else 0)
    return {"holes": int(raw["holes"]), "maximum_height": int(raw["max_height"]),
            "aggregate_height": sum(heights)}


def play(stream: Sequence[str], policy: str, expert: DellacherieSearchExpert | None = None) -> dict[str, Any]:
    game, state, lines, trace = TetrisAdapter(), TetrisState(), 0, []
    legacy = FourFeatureStudentAdapter(LearningAgent("legacy", [.2, .1, .15, .25], seed=0))
    expert = expert or DellacherieSearchExpert()
    for index, piece in enumerate(stream):
        legal = game.legal_actions(state, piece)
        if not legal:
            break
        if policy == "legacy":
            decision = legacy.choose_placement(state, piece, legal, deterministic=True)
            placement, selected = decision.placement, decision.evaluation
        else:
            placement = expert.preferred_placement(state, piece, legal).placement
            selected = next(x for x in legal if tuple(x.action) == (placement.rotation, placement.x))
        trace.append({"index": index, "piece": piece, "before": list(state.rows),
                      "placement": [placement.rotation, placement.x], "after": list(selected.next_state.rows)})
        state = selected.next_state
        lines += int(selected.info["lines_cleared"])
    return {"placements": len(trace), "lines_cleared": lines, **_metrics(state), "trace": trace}


def _bootstrap_relative(pairs: list[dict[str, Any]], seed: int, n: int = 10000) -> list[float]:
    rng, count, values = random.Random(seed), len(pairs), []
    for _ in range(n):
        sample = [pairs[rng.randrange(count)] for _ in range(count)]
        legacy = statistics.mean(x["legacy"]["placements"] for x in sample)
        expert = statistics.mean(x["expert"]["placements"] for x in sample)
        values.append(expert / legacy - 1.0)
    return sorted(values)


def benchmark(*, games: int = 50, maximum: int = 500, master_seed: int = 2026082302,
              beam_width: int = 3) -> dict[str, Any]:
    rng, pieces = random.Random(master_seed), tuple(SHAPES)
    streams = [[rng.choice(pieces) for _ in range(maximum)] for _ in range(games)]
    expert = DellacherieSearchExpert(beam_width=beam_width)
    pairs = []
    for game_id, stream in enumerate(streams):
        pairs.append({"game": game_id, "stream_sha256": hashlib.sha256("".join(stream).encode()).hexdigest(),
                      "legacy": play(stream, "legacy", expert), "expert": play(stream, "expert", expert)})
    disagreements, agreements = [], []
    game_adapter = TetrisAdapter()
    for pair in pairs:
        for legacy_step in pair["legacy"]["trace"]:
            state = TetrisState(tuple(legacy_step["before"]))
            legal = game_adapter.legal_actions(state, legacy_step["piece"])
            ranking = expert.rank_placements(state, legacy_step["piece"], legal)
            expert_placement = _placement(ranking[0])
            selected = next(x for x in legal if list(x.action) == expert_placement)
            candidate = (pair["game"], legacy_step, expert_placement,
                         list(selected.next_state.rows), ranking)
            target = disagreements if legacy_step["placement"] != expert_placement else agreements
            if len(disagreements) < 8 or (len(disagreements) >= 8 and len(agreements) < 2):
                target.append(candidate)
            if len(disagreements) >= 8 and len(agreements) >= 2:
                break
        if len(disagreements) >= 8 and len(agreements) >= 2:
            break
    examples = []
    for game_id, legacy_step, expert_placement, expert_after, ranking in disagreements[:8] + agreements[:2]:
        state, piece = TetrisState(tuple(legacy_step["before"])), legacy_step["piece"]
        legal = TetrisAdapter().legal_actions(state, piece)
        legacy_placement = Placement(*legacy_step["placement"])
        examples.append({"selection": "first_disagreement" if legacy_step["placement"] != expert_placement else "agreement_control",
                         "game": game_id, "placement_index": legacy_step["index"], "piece": piece,
                         "board_before": legacy_step["before"], "legacy_placement": legacy_step["placement"],
                         "expert_placement": expert_placement, "legacy_board_after": legacy_step["after"],
                         "expert_board_after": expert_after,
                         "legacy_regret": expert.regret(state, piece, legal, legacy_placement).regret,
                         "ranking": [{"placement": _placement(x), "value": x.value, "rank": x.rank} for x in ranking]})
    relative = _bootstrap_relative(pairs, 2026082303)
    def summary(label: str) -> dict[str, Any]:
        rows = [x[label] for x in pairs]
        return {metric: {"mean": statistics.mean(x[metric] for x in rows),
                         "median": statistics.median(x[metric] for x in rows),
                         "min": min(x[metric] for x in rows), "max": max(x[metric] for x in rows)}
                for metric in ("placements", "lines_cleared", "holes", "maximum_height", "aggregate_height")}
    legacy_mean = summary("legacy")["placements"]["mean"]
    expert_mean = summary("expert")["placements"]["mean"]
    improvement = expert_mean / legacy_mean - 1.0
    ci = [relative[249], relative[9749]]
    passed = improvement >= .50 and ci[0] > .20
    # Keep complete numerical outcomes but omit bulky per-step traces from main results.
    compact_pairs = [{k: v for k, v in pair.items() if k not in ()} for pair in pairs]
    for pair in compact_pairs:
        pair["legacy"] = {k: v for k, v in pair["legacy"].items() if k != "trace"}
        pair["expert"] = {k: v for k, v in pair["expert"].items() if k != "trace"}
    return {"master_seed": master_seed, "matched_games": games, "maximum_placements": maximum,
            "beam_width": beam_width, "legacy": summary("legacy"), "expert": summary("expert"),
            "relative_placement_improvement": improvement, "relative_improvement_95ci": ci,
            "criterion": "point improvement >= 50% and paired-bootstrap 95% CI lower bound > 20%",
            "threshold_satisfied": passed, "pairs": compact_pairs, "representative_examples": examples}


def certificate(validation: Mapping[str, Any], result: Mapping[str, Any] | None) -> str:
    lines = ["EXPERT POLICY VALIDATION", "", f"Expert implementation/version: {validation['expert']}"]
    labels = {"rules_compatible": "Rules compatible", "recommended_placements_legal": "All recommended placements legal",
              "deterministic_rankings": "Deterministic rankings reproducible", "board_transitions_match": "Independent board transitions match",
              "rankings_consistent": "Expert rankings internally consistent"}
    lines += [f"{labels[k]}: {validation['checks'][k]}" for k in labels]
    if result is None:
        lines += ["", "Matched benchmark games: NOT AVAILABLE", "Threshold satisfied: NOT AVAILABLE",
                  "Expert suitable as Day 3 oracle: NO"]
    else:
        lp, ep = result["legacy"]["placements"]["mean"], result["expert"]["placements"]["mean"]
        lo, hi = result["relative_improvement_95ci"]
        lines += ["", f"Matched benchmark games: {result['matched_games']}", f"Legacy mean placements: {lp:.3f}",
                  f"Expert mean placements: {ep:.3f}", f"Relative improvement: {result['relative_placement_improvement']:.1%}",
                  f"Paired-bootstrap 95% CI: [{lo:.1%}, {hi:.1%}]", f"Predetermined strength threshold: {result['criterion']}",
                  f"Threshold satisfied: {'PASS' if result['threshold_satisfied'] else 'FAIL'}",
                  f"Expert suitable as Day 3 oracle: {'YES' if result['threshold_satisfied'] and validation['overall'] == 'PASS' else 'NO'}"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("pilot", "run", "validate"))
    args = parser.parse_args(argv)
    DAY2.mkdir(parents=True, exist_ok=True)
    if args.mode == "pilot":
        start = time.perf_counter(); benchmark(games=3); elapsed = time.perf_counter() - start
        record = {"games": 3, "elapsed_seconds": elapsed, "projected_100_game_seconds": elapsed / 3 * 100,
                  "performance_not_recorded": True}
        (DAY2 / "runtime_pilot.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2)); return 0
    validation = validate_expert()
    (DAY2 / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    (DAY2 / "corpus.json").write_text(json.dumps(validation_corpus(), indent=2) + "\n")
    if args.mode == "validate":
        result_path = DAY2 / "benchmark.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else None
    else:
        result = benchmark(games=50, beam_width=3)
        (DAY2 / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
        (DAY2 / "representative_examples.json").write_text(
            json.dumps(result["representative_examples"], indent=2) + "\n")
    (DAY2 / "certificate.txt").write_text(certificate(validation, result))
    print(certificate(validation, result), end="")
    return 0 if validation["overall"] == "PASS" and (result is None or result["threshold_satisfied"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
