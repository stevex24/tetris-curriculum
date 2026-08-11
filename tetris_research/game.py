from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class ActionEvaluation:
    action: Any
    features: tuple[float, ...]
    raw_features: dict[str, float]
    next_state: Any
    reward: float
    info: dict[str, Any]


class GameAdapter(Protocol):
    """The small boundary an agent needs from any turn-based game."""

    feature_names: Sequence[str]

    def initial_state(self) -> Any: ...

    def sample_environment(self, rng: Any) -> Any: ...

    def legal_actions(self, state: Any, environment: Any) -> list[ActionEvaluation]: ...

    def terminal(self, state: Any) -> bool: ...

    def serialize_state(self, state: Any) -> Any: ...
