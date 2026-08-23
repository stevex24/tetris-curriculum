"""Game-level boundary between Tetris infrastructure and a student policy."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .game import ActionEvaluation
from .tetris import TetrisState


@dataclass(frozen=True, order=True)
class Placement:
    """A final placement, not a sequence of controller inputs."""

    rotation: int
    x: int
    use_hold: bool = False


@dataclass(frozen=True)
class ActionScore:
    placement: Placement
    score: float


@dataclass(frozen=True)
class PlacementDecision:
    placement: Placement
    evaluation: ActionEvaluation
    preferences: tuple[ActionScore, ...] | None = None


@dataclass(frozen=True)
class AgentExperience:
    state: TetrisState
    piece: str
    decision: PlacementDecision
    reward: float
    next_state: TetrisState
    terminal: bool


class StudentAgent(ABC):
    """Minimal replaceable policy API used by training and evaluation.

    Implementations may be heuristic, search-based, tabular, or neural.  Internal
    weights are deliberately not part of this contract.
    """

    @property
    @abstractmethod
    def agent_id(self) -> str: ...

    @property
    @abstractmethod
    def agent_version(self) -> str: ...

    @property
    @abstractmethod
    def learning_enabled(self) -> bool: ...

    @abstractmethod
    def clone(self, agent_id: str | None = None) -> "StudentAgent": ...

    @abstractmethod
    def choose_placement(
        self,
        state: TetrisState,
        piece: str,
        legal_placements: Sequence[ActionEvaluation],
        *,
        learn: bool = False,
        deterministic: bool = False,
    ) -> PlacementDecision: ...

    def action_scores(
        self, state: TetrisState, piece: str,
        legal_placements: Sequence[ActionEvaluation],
    ) -> tuple[ActionScore, ...] | None:
        """Return policy preferences when meaningful; opaque policies may return None."""
        return None

    def update(self, experience: AgentExperience) -> None:
        """Optional separated update path; unsupported agents fail explicitly."""
        raise NotImplementedError(f"{type(self).__name__} does not support experience updates")

    def finish_episode(self, *, learned: bool) -> None:
        """Optional episode boundary hook."""

    @abstractmethod
    def serialize_state(self) -> Mapping[str, Any]: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "StudentAgent": ...
