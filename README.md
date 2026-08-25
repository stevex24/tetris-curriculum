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

Run the Day 3 direct expert-imitation experiment:

```sh
python -m tetris_research.day3 smoke
python -m tetris_research.day3 final
```

Day 3 adds a replaceable linear behavior-cloning student and an equal-update random-label control.
The final held-out run achieved 54.2% expert agreement and 8.127 mean expert-relative regret for the
taught student, versus 12.5% and 60.434 for control; matched mean survival was 259.2 versus 25.4
placements. Students have no expert reference during evaluation, and serialized states are checked
for mutation. See [the Day 3 report](docs/day3_transfer.md). Direct imitation is the baseline for a
future equal-budget personalized-tutorial comparison, not the ultimate tutor method.

Run the frozen Hour 4 evaluation experiment (after its required three-replicate smoke run):

```sh
python hour4_experiment.py --replicates 50 --master-seed 400004 --output artifacts/hour4/experiment_50
```

Hour 4 predeclares mean successful placements per six unseen challenges as its primary
general-performance outcome and retains lines cleared as secondary. Its trajectory-averaged
observable component scores separately assess targeted learning. The frozen specification is
in `artifacts/hour4/evaluation_specification.json`; Hour 3 is not rescored or modified.

Run the Hour 5 learner-profile audit and small diagnostic demonstration (this does not train on
tutorials or run an effectiveness experiment):

```bash
python hour5_diagnostic.py --output artifacts/hour5/demo --calibration-count 80 --diagnostic-count 50
```

Run the predeclared Hour 3 controlled pilot (write to a new directory):

```sh
python hour3_pilot.py --replicates 50 --master-seed 300003 --output artifacts/hour3/pilot_50
```

Its primary measure is mean lines cleared per unseen evaluation challenge. Evaluation uses
disposable agent clones with learning disabled; all three conditions receive matched challenges
and equal placement budgets. CSV, challenge-level raw measurements, statistics, configuration,
seeds, timing, platform, and source-commit provenance are saved without changing Hour 1/2 files.

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
