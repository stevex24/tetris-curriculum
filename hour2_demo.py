import json
from pathlib import Path

from tetris_research import LearningAgent
from tetris_research.training import (board_rows, control_material, select_history_material,
                                      select_rating_only_material, train, write_training_log)


def _recent_records(path: Path, agent_id: str, games: int = 2) -> list[dict]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [record for record in records if record.get("agent") == agent_id][-games:]


def _show_board(rows: list[str]) -> None:
    for row in rows:
        print(f"  |{row}|")
    print("  +----------+")


def main() -> None:
    root = Path("artifacts")
    original = LearningAgent.load(root / "agents" / "agent_a.json")
    rating, selection_seed, training_seed, steps = 1500.0, 2402, 2403, 12
    history = _recent_records(root / "games.jsonl", original.name)
    agents = [original.clone(f"{original.name}_{suffix}")
              for suffix in ("control", "rating_only", "rating_history")]
    materials = [control_material(), select_rating_only_material(rating, selection_seed),
                 select_history_material(rating, history, selection_seed)]

    identical = all(agent.weights == agents[0].weights and
                    agent._rng.getstate() == agents[0]._rng.getstate() for agent in agents[1:])
    print("Hour 2 plumbing demonstration")
    print("Identical initial policy weights and RNG state:", identical)
    print("Training placements per condition:", steps)
    records = []
    for agent, material in zip(agents, materials):
        print(f"\n{material.condition}: difficulty={material.difficulty}")
        if material.diagnosis:
            print("  observable diagnosis severities:",
                  {key: round(value, 4) for key, value in material.diagnosis.items()})
            print("  diagnosed weakness:", material.diagnosed_weakness)
        print("  rationale:", material.rationale)
        _show_board(board_rows(material.state))
        record = train(agent, material, rating, steps, training_seed)
        records.append(record)
        print("  weights before:", [round(value, 6) for value in record["policy_weights_before"]])
        print("  weights after: ", [round(value, 6) for value in record["policy_weights_after"]])
        print("  learning path:", record["learning_path"])

    output = root / "hour2_demo_training.jsonl"
    write_training_log(records, output)
    print("\nEqual exposure:", [record["training_steps"] for record in records])
    print("Log:", output)
    print("This tiny run is a plumbing/validity check, not evidence for the hypothesis.")


if __name__ == "__main__":
    main()
