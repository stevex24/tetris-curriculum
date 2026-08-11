from __future__ import annotations

import json
from pathlib import Path
from random import Random

from .agent import LearningAgent
from .game import GameAdapter


def play_game(adapter: GameAdapter, agent: LearningAgent, seed: int, history_path: Path,
              max_steps: int = 200, learn: bool = True) -> dict:
    environment_rng = Random(seed)
    state, steps, total_lines = adapter.initial_state(), [], 0
    for index in range(max_steps):
        environment = adapter.sample_environment(environment_rng)
        choices = adapter.legal_actions(state, environment)
        if not choices:
            break
        choice = agent.choose(choices, learn=learn)
        state = choice.next_state
        total_lines += choice.info.get("lines_cleared", 0)
        action = list(choice.action) if isinstance(choice.action, tuple) else choice.action
        steps.append({"step": index, "environment": environment, "action": action,
                      "reward": choice.reward, "features": choice.raw_features,
                      "state_after": adapter.serialize_state(state)})
        if adapter.terminal(state):
            break
    agent.finish_game(learned=learn)
    record = {"agent": agent.name, "seed": seed, "learning_enabled": learn,
              "steps_played": len(steps), "lines_cleared": total_lines,
              "terminal": len(steps) < max_steps, "steps": steps,
              "final_weights": agent.weights.copy()}
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a") as stream:
        stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    return record
