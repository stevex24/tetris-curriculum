"""Day-2 seam only: contracts for an expert/reference placement policy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .game import ActionEvaluation
from .student import Placement
from .tetris import TetrisState


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

