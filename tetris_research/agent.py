from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from random import Random

from .game import ActionEvaluation


@dataclass
class LearningAgent:
    name: str
    weights: list[float]
    learning_rate: float = 0.04
    temperature: float = 0.25
    seed: int = 0
    games_learned: int = 0
    _rng: Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = Random(self.seed)

    def choose(self, choices: list[ActionEvaluation], learn: bool = True) -> ActionEvaluation:
        logits = [sum(w * f for w, f in zip(self.weights, c.features)) / self.temperature for c in choices]
        peak = max(logits)
        exp = [math.exp(v - peak) for v in logits]
        probs = [v / sum(exp) for v in exp]
        draw, cumulative, selected = self._rng.random(), 0.0, len(choices) - 1
        for i, probability in enumerate(probs):
            cumulative += probability
            if draw <= cumulative:
                selected = i
                break
        choice = choices[selected]
        if learn:
            expected = [sum(p * c.features[j] for p, c in zip(probs, choices)) for j in range(len(self.weights))]
            # Online REINFORCE: only experienced states use the ordinary update path.
            advantage = max(-2.0, min(4.0, choice.reward))
            self.weights = [w + self.learning_rate * advantage * (f - e)
                            for w, f, e in zip(self.weights, choice.features, expected)]
        return choice

    def finish_game(self, learned: bool = True) -> None:
        if learned:
            self.games_learned += 1

    def clone(self, name: str) -> "LearningAgent":
        clone = copy.deepcopy(self)
        clone.name = name
        return clone

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"name": self.name, "weights": self.weights, "learning_rate": self.learning_rate,
                "temperature": self.temperature, "seed": self.seed, "games_learned": self.games_learned,
                "rng_state": repr(self._rng.getstate())}
        path.write_text(json.dumps(data, indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "LearningAgent":
        import ast
        data = json.loads(path.read_text())
        rng_state = ast.literal_eval(data.pop("rng_state"))
        agent = cls(**data)
        agent._rng.setstate(rng_state)
        return agent
