"""Expert-policy contract and deterministic simulator-native reference policy.

The implementation adapts the classic Dellacherie feature family (landing
height, eroded cells, transitions, holes and wells) and adds a restricted
depth-two beam expectimax.  Values are heuristic utility units, not
probabilities or calibrated confidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .game import ActionEvaluation
from .student import Placement
from .tetris import HEIGHT, SHAPES, WIDTH, TetrisAdapter, TetrisState


@dataclass(frozen=True)
class PlacementEvaluation:
    placement: Placement
    value: float
    rank: int
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ActionRegret:
    chosen: Placement
    expert_preferred: Placement
    regret: float


class ExpertPolicy(Protocol):
    @property
    def expert_id(self) -> str: ...

    def rank_placements(self, state: TetrisState, piece: str,
                        legal_placements: Sequence[ActionEvaluation], *,
                        deterministic: bool = True) -> Sequence[PlacementEvaluation]: ...


# Widely reproduced coefficients for the Dellacherie feature family, in the
# conventional orientation (the feature method, rather than coefficient
# provenance, is the documented-method adaptation).
# The final two terms are simulator-specific safety additions, explicitly kept
# separate from the adapted feature family.
DELLACHERIE_WEIGHTS = {
    "landing_height": -4.500158825082766,
    "eroded_piece_cells": 3.4181268101392694,
    "row_transitions": -3.2178882868487753,
    "column_transitions": -9.348695305445199,
    "holes": -7.899265427351652,
    "cumulative_wells": -3.3855972247263626,
}


class DellacherieSearchExpert:
    """Rich deterministic placement oracle over canonical simulator actions.

    Search has no preview information.  Its second ply therefore evaluates the
    uniform expectation over all seven possible next pieces, retaining the best
    placement for each.  Only the best ``beam_width`` first-ply afterstates get
    that continuation; this is a deterministic beam-search approximation.
    """

    VERSION = "dellacherie-depth2-expectimax-v1"

    def __init__(self, *, beam_width: int = 6, continuation_discount: float = 0.35):
        if beam_width < 1:
            raise ValueError("beam_width must be positive")
        self.beam_width = beam_width
        self.continuation_discount = continuation_discount
        self.game = TetrisAdapter()

    @property
    def expert_id(self) -> str:
        return self.VERSION

    @staticmethod
    def _placement(choice: ActionEvaluation) -> Placement:
        return Placement(int(choice.action[0]), int(choice.action[1]))

    @staticmethod
    def features(state: TetrisState, piece: str, choice: ActionEvaluation) -> dict[str, float]:
        rotation, x = map(int, choice.action)
        cells = SHAPES[piece][rotation]
        landing_y = int(choice.info["landing_y"])
        cleared = int(choice.info["lines_cleared"])
        cleared_rows = []
        if cleared:
            before = list(state.rows)
            for dx, dy in cells:
                before[landing_y + dy] |= 1 << (x + dx)
            cleared_rows = [y for y, row in enumerate(before) if row == (1 << WIDTH) - 1]
        eroded = cleared * sum(landing_y + dy in cleared_rows for _, dy in cells)
        landing_height = landing_y + sum(dy for _, dy in cells) / 4.0
        rows = choice.next_state.rows
        heights, holes = [], 0
        for column in range(WIDTH):
            occupied = [y for y, row in enumerate(rows) if row & (1 << column)]
            height = max(occupied) + 1 if occupied else 0
            heights.append(height)
            holes += sum(not (rows[y] & (1 << column)) for y in range(height))
        row_transitions = 0
        for y in range(HEIGHT):
            previous = 1
            for column in range(WIDTH):
                occupied = 1 if rows[y] & (1 << column) else 0
                row_transitions += occupied != previous
                previous = occupied
            row_transitions += previous != 1
        column_transitions = 0
        for column in range(WIDTH):
            previous = 1  # floor is occupied
            for y in range(HEIGHT):
                occupied = 1 if rows[y] & (1 << column) else 0
                column_transitions += occupied != previous
                previous = occupied
        wells = 0
        for column in range(WIDTH):
            depth = 0
            for y in range(HEIGHT):
                occupied = bool(rows[y] & (1 << column))
                left = column == 0 or bool(rows[y] & (1 << (column - 1)))
                right = column == WIDTH - 1 or bool(rows[y] & (1 << (column + 1)))
                if not occupied and left and right:
                    depth += 1
                    wells += depth
                else:
                    depth = 0
        return {
            "landing_height": float(landing_height),
            "eroded_piece_cells": float(eroded),
            "row_transitions": float(row_transitions),
            "column_transitions": float(column_transitions),
            "holes": float(holes),
            "cumulative_wells": float(wells),
            "aggregate_height": float(sum(heights)),
            "maximum_height": float(max(heights)),
        }

    def _static(self, state: TetrisState, piece: str, choice: ActionEvaluation) -> tuple[float, dict[str, float]]:
        features = self.features(state, piece, choice)
        value = sum(DELLACHERIE_WEIGHTS[name] * features[name]
                    for name in DELLACHERIE_WEIGHTS)
        # Simulator-specific safety additions. They matter principally near the
        # hard 20-row top-out boundary absent from unbounded theoretical play.
        value -= 0.12 * features["aggregate_height"]
        value -= 1.5 * max(0.0, features["maximum_height"] - 16.0) ** 2
        return value, features

    def rank_placements(self, state: TetrisState, piece: str,
                        legal_placements: Sequence[ActionEvaluation], *,
                        deterministic: bool = True) -> Sequence[PlacementEvaluation]:
        if not deterministic:
            raise ValueError("reference policy supports deterministic analysis only")
        if not legal_placements:
            return ()
        first = []
        for index, choice in enumerate(legal_placements):
            static, features = self._static(state, piece, choice)
            first.append((index, choice, static, features))
        beam = {index for index, _, _, _ in
                sorted(first, key=lambda item: (-item[2], item[0]))[:self.beam_width]}
        scored = []
        pieces = tuple(SHAPES)
        for index, choice, static, features in first:
            continuation = None
            if index in beam:
                next_values = []
                for next_piece in pieces:
                    replies = self.game.legal_actions(choice.next_state, next_piece)
                    if replies:
                        next_values.append(max(self._static(choice.next_state, next_piece, reply)[0]
                                               for reply in replies))
                    else:
                        next_values.append(-100000.0)
                continuation = sum(next_values) / len(next_values)
            # Non-beam nodes use the static leaf value as their continuation
            # estimate; searched nodes use an explicit expected best reply.
            value = static + self.continuation_discount * (
                continuation if continuation is not None else static)
            scored.append((index, choice, value, static, continuation, features))
        ordered = sorted(scored, key=lambda item: (-item[2], item[0]))
        result = []
        for rank, (_, choice, value, static, continuation, features) in enumerate(ordered, 1):
            result.append(PlacementEvaluation(
                self._placement(choice), value, rank,
                {"search_depth": 2 if continuation is not None else 1,
                 "beam_width": self.beam_width, "static_value": static,
                 "expected_second_ply_value": continuation, "features": features,
                 "score_semantics": "static Dellacherie utility plus 0.35 times uniform-next-piece best-reply utility"}))
        return tuple(result)

    def preferred_placement(self, state: TetrisState, piece: str,
                            legal_placements: Sequence[ActionEvaluation]) -> PlacementEvaluation:
        ranking = self.rank_placements(state, piece, legal_placements)
        if not ranking:
            raise ValueError("preferred_placement requires at least one legal placement")
        return ranking[0]

    def regret(self, state: TetrisState, piece: str,
               legal_placements: Sequence[ActionEvaluation], chosen: Placement) -> ActionRegret:
        ranking = self.rank_placements(state, piece, legal_placements)
        values = {item.placement: item.value for item in ranking}
        if chosen not in values:
            raise ValueError("chosen placement is not canonical/legal")
        return ActionRegret(chosen, ranking[0].placement, ranking[0].value - values[chosen])
