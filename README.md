# Tetris learning research prototype

A dependency-free, headless foundation for controlled learning experiments. One
step selects a complete legal tetromino placement. This is research scaffolding,
not a real-time game.

Run the demonstration:

```sh
python demo.py
```

Run the small Hour 2 three-condition plumbing demonstration:

```sh
python hour2_demo.py
```

Hour 2 compares equal placement budgets for ordinary control practice, a
rating-only tutorial, and a rating-plus-observable-history tutorial. The Elo
difficulty tiers and history severity scales are explicit prototype assumptions,
not empirically calibrated Tetris pedagogy. Tutorial selection is separate from
evaluation and does not update Elo.

Run tests:

```sh
python -m unittest discover -s tests -v
```

The demo writes complete per-step game records to `artifacts/games.jsonl` and
agent checkpoints to `artifacts/agents/*.json`. JSONL has one game object per
line; each step includes the piece, chosen rotation/x position, reward,
interpretable board features, and full board after the move.

## Architecture

`GameAdapter` is the game-independent boundary: create state, sample the next
environment event, enumerate evaluated legal actions, detect termination, and
serialize state. `TetrisAdapter` is its only implementation. `LearningAgent`
knows only feature vectors; its seeded softmax policy changes via an online
REINFORCE update and its weights and random-generator state persist. The match
runner logs games, while `EloRatings` independently evaluates paired games.

## Deliberately absent

There is no UI, timing, multiplayer/networking, replay importer, TETR.IO
compatibility, training-level selector, tutorial bonus, garbage/attacks, hold,
preview queue, wall kicks, T-spins, lock delay, combos, or Glicko-2. Head-to-head
evaluation simply gives both agents the same seeded piece stream and compares
lines cleared. Pieces are sampled independently rather than with a seven-bag.
