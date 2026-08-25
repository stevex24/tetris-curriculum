"""Small trainable placement policy for Day 3 behavior cloning."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .game import ActionEvaluation
from .student import ActionScore, Placement, PlacementDecision, StudentAgent
from .tetris import TetrisState


class LinearImitationStudent(StudentAgent):
    """Four-afterstate-feature softmax policy trained by cross-entropy.

    The features come from the simulator's public action representation.  No
    expert features, values, search state, or coefficients are stored here.
    """

    VERSION = "linear-imitation-four-feature-v1"

    def __init__(self, agent_id: str, weights: Sequence[float] | None = None, *,
                 learning_rate: float = 0.35, learning_enabled: bool = True):
        self._agent_id = agent_id
        self.weights = list(weights if weights is not None else (0.0,) * 4)
        if len(self.weights) != 4:
            raise ValueError("linear imitation student requires four weights")
        self.learning_rate = float(learning_rate)
        self._learning_enabled = bool(learning_enabled)
        self.updates = 0

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def agent_version(self) -> str:
        return self.VERSION

    @property
    def learning_enabled(self) -> bool:
        return self._learning_enabled

    @staticmethod
    def _placement(choice: ActionEvaluation) -> Placement:
        return Placement(int(choice.action[0]), int(choice.action[1]))

    def clone(self, agent_id: str | None = None) -> "LinearImitationStudent":
        clone = LinearImitationStudent(agent_id or self.agent_id, self.weights,
                                       learning_rate=self.learning_rate,
                                       learning_enabled=self.learning_enabled)
        clone.updates = self.updates
        return clone

    def action_scores(self, state: TetrisState, piece: str,
                      legal_placements: Sequence[ActionEvaluation]) -> tuple[ActionScore, ...]:
        return tuple(ActionScore(self._placement(choice),
                                 sum(w * f for w, f in zip(self.weights, choice.features)))
                     for choice in legal_placements)

    def choose_placement(self, state: TetrisState, piece: str,
                         legal_placements: Sequence[ActionEvaluation], *, learn: bool = False,
                         deterministic: bool = False) -> PlacementDecision:
        if not legal_placements:
            raise ValueError("choose_placement requires a legal placement")
        if learn:
            raise RuntimeError("imitation updates require an explicit labeled placement")
        scores = self.action_scores(state, piece, legal_placements)
        index = max(range(len(scores)), key=lambda i: scores[i].score)
        choice = legal_placements[index]
        return PlacementDecision(self._placement(choice), choice, scores)

    def learn_from_label(self, state: TetrisState, piece: str,
                         legal_placements: Sequence[ActionEvaluation], label: Placement) -> float:
        """One full-batch action-softmax cross-entropy SGD update."""
        if not self.learning_enabled:
            raise RuntimeError("learning is disabled for this agent")
        indices = [i for i, choice in enumerate(legal_placements)
                   if self._placement(choice) == label]
        if len(indices) != 1:
            raise ValueError("label must identify exactly one legal placement")
        target = indices[0]
        logits = [sum(w * f for w, f in zip(self.weights, choice.features))
                  for choice in legal_placements]
        peak = max(logits)
        exps = [math.exp(value - peak) for value in logits]
        total = sum(exps)
        probabilities = [value / total for value in exps]
        expected = [sum(probability * choice.features[j]
                        for probability, choice in zip(probabilities, legal_placements))
                    for j in range(4)]
        target_features = legal_placements[target].features
        self.weights = [weight + self.learning_rate * (wanted - average)
                        for weight, wanted, average in zip(self.weights, target_features, expected)]
        self.updates += 1
        return -math.log(max(probabilities[target], 1e-300))

    def serialize_state(self) -> Mapping[str, Any]:
        return {"format": self.VERSION, "agent_id": self.agent_id,
                "agent_version": self.agent_version, "learning_enabled": self.learning_enabled,
                "feature_names": ["holes", "max_height", "bumpiness", "lines_cleared"],
                "weights": list(self.weights), "learning_rate": self.learning_rate,
                "updates": self.updates, "expert_parameters": None}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.serialize_state(), indent=2) + "\n")

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "LinearImitationStudent":
        if state.get("format") != cls.VERSION:
            raise ValueError("unsupported imitation student state")
        agent = cls(str(state["agent_id"]), state["weights"],
                    learning_rate=float(state["learning_rate"]),
                    learning_enabled=bool(state["learning_enabled"]))
        agent.updates = int(state["updates"])
        return agent

    @classmethod
    def load(cls, path: Path) -> "LinearImitationStudent":
        return cls.from_state(json.loads(path.read_text()))
