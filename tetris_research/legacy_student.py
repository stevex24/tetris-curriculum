"""Compatibility adapter for the frozen four-feature learner."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .agent import LearningAgent
from .game import ActionEvaluation
from .student import ActionScore, Placement, PlacementDecision, StudentAgent
from .tetris import TetrisState


class FourFeatureStudentAdapter(StudentAgent):
    """Expose ``LearningAgent`` without changing its policy or update rule."""

    VERSION = "hours-1-9-four-feature-v1"

    def __init__(self, learner: LearningAgent, *, learning_enabled: bool = True):
        self.learner = learner
        self._learning_enabled = learning_enabled

    @property
    def agent_id(self) -> str:
        return self.learner.name

    @property
    def agent_version(self) -> str:
        return self.VERSION

    @property
    def learning_enabled(self) -> bool:
        return self._learning_enabled

    def clone(self, agent_id: str | None = None) -> "FourFeatureStudentAdapter":
        name = self.agent_id if agent_id is None else agent_id
        return FourFeatureStudentAdapter(self.learner.clone(name), learning_enabled=self.learning_enabled)

    @staticmethod
    def _placement(choice: ActionEvaluation) -> Placement:
        rotation, x = choice.action
        return Placement(int(rotation), int(x))

    def action_scores(self, state: TetrisState, piece: str,
                      legal_placements: Sequence[ActionEvaluation]) -> tuple[ActionScore, ...]:
        return tuple(ActionScore(self._placement(choice),
                                 sum(w * f for w, f in zip(self.learner.weights, choice.features)))
                     for choice in legal_placements)

    def choose_placement(self, state: TetrisState, piece: str,
                         legal_placements: Sequence[ActionEvaluation], *, learn: bool = False,
                         deterministic: bool = False) -> PlacementDecision:
        if not legal_placements:
            raise ValueError("choose_placement requires at least one legal placement")
        if learn and not self.learning_enabled:
            raise RuntimeError("learning is disabled for this agent")
        scores = self.action_scores(state, piece, legal_placements)
        if deterministic:
            # max() preserves the canonical legal-action ordering on score ties.
            index = max(range(len(scores)), key=lambda i: scores[i].score)
            choice = legal_placements[index]
        else:
            # This is the exact historical sampling/update path.
            choice = self.learner.choose(list(legal_placements), learn=learn)
        return PlacementDecision(self._placement(choice), choice, scores)

    def finish_episode(self, *, learned: bool) -> None:
        self.learner.finish_game(learned=learned and self.learning_enabled)

    def serialize_state(self) -> Mapping[str, Any]:
        return {
            "format": "four-feature-learning-agent-v1",
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
            "learning_enabled": self.learning_enabled,
            "weights": list(self.learner.weights),
            "learning_rate": self.learner.learning_rate,
            "temperature": self.learner.temperature,
            "seed": self.learner.seed,
            "games_learned": self.learner.games_learned,
            "rng_state": repr(self.learner._rng.getstate()),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.serialize_state(), indent=2) + "\n")

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "FourFeatureStudentAdapter":
        if state.get("format") != "four-feature-learning-agent-v1":
            raise ValueError("unsupported legacy student state")
        learner = LearningAgent(str(state["agent_id"]), list(state["weights"]),
                                float(state["learning_rate"]), float(state["temperature"]),
                                int(state["seed"]), int(state["games_learned"]))
        learner._rng.setstate(ast.literal_eval(str(state["rng_state"])))
        return cls(learner, learning_enabled=bool(state["learning_enabled"]))

    @classmethod
    def load(cls, path: Path) -> "FourFeatureStudentAdapter":
        return cls.from_state(json.loads(path.read_text()))


def as_student_agent(agent: StudentAgent | LearningAgent) -> StudentAgent:
    """Compatibility coercion for untouched Hours 1-9 callers."""
    return agent if isinstance(agent, StudentAgent) else FourFeatureStudentAdapter(agent)

