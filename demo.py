from pathlib import Path

from tetris_research import EloRatings, LearningAgent, TetrisAdapter
from tetris_research.experiment import play_game


def main() -> None:
    output = Path("artifacts")
    history = output / "games.jsonl"
    history.parent.mkdir(exist_ok=True)
    history.write_text("")
    adapter = TetrisAdapter()
    agents = [
        LearningAgent("agent_a", [0.20, 0.10, 0.15, 0.25], seed=101),
        LearningAgent("agent_b", [-0.10, 0.25, 0.05, 0.10], seed=202),
    ]
    initial = {a.name: a.weights.copy() for a in agents}

    # Clone proof uses identical parameters AND RNG state on the same piece seed.
    clone = agents[0].clone("agent_a_clone")
    proof_a = play_game(adapter, agents[0].clone("proof_a"), 777, history, max_steps=20, learn=False)
    proof_b = play_game(adapter, clone.clone("proof_b"), 777, history, max_steps=20, learn=False)
    clone_identical = [(s["action"], s["state_after"]) for s in proof_a["steps"]] == [
        (s["action"], s["state_after"]) for s in proof_b["steps"]]

    elo = EloRatings()
    updates = []
    for game_index in range(6):
        seed = 1000 + game_index
        results = [play_game(adapter, a, seed, history, max_steps=120) for a in agents]
        score_a = 1.0 if results[0]["lines_cleared"] > results[1]["lines_cleared"] else (
            0.0 if results[0]["lines_cleared"] < results[1]["lines_cleared"] else 0.5)
        updates.append(elo.update(agents[0].name, agents[1].name, score_a))
        for agent in agents:
            agent.save(output / "agents" / f"{agent.name}.json")

    print("Feature order:", list(adapter.feature_names))
    print("Agent A initial parameters:", [round(v, 6) for v in initial["agent_a"]])
    print("Agent A final parameters:  ", [round(v, 6) for v in agents[0].weights])
    print("Parameters changed:", initial["agent_a"] != agents[0].weights)
    print("Exact clone behaves identically initially:", clone_identical)
    print("Elo updates:")
    for i, update in enumerate(updates, 1):
        print(f"  {i}: A {update['a_before']:.2f} -> {update['a_after']:.2f}; "
              f"B {update['b_before']:.2f} -> {update['b_after']:.2f}; result={update['score_a']}")
    print("History:", history, "(one complete game object per JSONL line)")
    print("Persisted agents:", output / "agents")


if __name__ == "__main__":
    main()
