# Adaptive Tetris Tutor

The Adaptive Tetris Tutor asks whether an AI can improve learning efficiency by examining a player's history, identifying recurring weaknesses, and generating practice situations targeted to those weaknesses. The current implementation uses machine students as a controlled experimental testbed.

The validated system forms this pipeline:

```text
play history → weakness diagnosis → targeted practice → ordinary learner update → independent evaluation
```

The teaching system changes **what the student practices**. It does not directly edit the student's hidden parameters, copy expert coefficients, or use shaped tutorial rewards. This is a research simulator built around complete tetromino placements, not a real-time game or a model of human learning speed.

Public repository: <https://github.com/stevex24/tetris-curriculum>

## Headline result

Day 7 compared three conditions from identical copies of the same trained machine student. Each received exactly 500 learning opportunities and parameter updates; frozen policies were then evaluated on 12 matched ordinary-game piece streams in each of eight paired replicates.

| Training condition | Mean line improvement |
| --- | ---: |
| Ordinary practice | +15.60 |
| Personalized tutorials | +35.06 |
| Direct expert imitation | +120.18 |

Personalized minus ordinary was **+19.46 lines**, with a 10,000-sample paired-bootstrap 90% CI of **[8.77, 30.44]**. Personalized training beat ordinary practice in six of eight paired replicates. The preregistered classification was **minimum success**: personalized tutorials cleared the criterion against ordinary practice, but did not beat direct imitation.

Direct imitation was substantially stronger. It reached the 500-placement evaluation cap in 54 of 96 evaluations, so its placement results are strongly ceiling-censored. Full results and the locked design are in the [Day 7 report](docs/day7_comparison.md), [final artifact](experiments/day7/final_results.json), and [preregistration](experiments/day7/preregistration.json).

The supported claim is narrow: **history-aware targeted practice outperformed ordinary practice in this machine-student experiment**. The result does not establish that personalized tutorials beat direct imitation, improve human players, or are optimal.

## Two research stages

### Stage I — rapid prototype, Hours 1–9

The initial stage built a dependency-free Tetris learner, observable histories, rating- and history-selected tutorial boards, matched evaluations, and an animated curriculum demo. Hours 4–7 exposed measurement and replication problems: Hour 4's uncalibrated diagnosis selected holes in all 50 cases; Hour 5 calibrated the profile; Hour 6 found a small history-aware advantage; and the locked Hour 7 replication did not reproduce the control comparison. Hour 9's frozen 1,000-replicate test estimated history-aware minus ordinary at +0.0013 successful placements per challenge (95% CI [−0.0825, 0.0852]), showing that the apparent general-performance advantage was not stable.

Stage I culminated at `c21fb78`. It was a useful proof of concept and diagnostic exercise, but its four-feature learner, short 40-placement training budgets, fixed tutorial boards, and initially miscalibrated weakness scales motivated a stricter rebuild. See [Hour 4](docs/hour4.md), [Hour 5](docs/hour5.md), [Hour 6](docs/hour6.md), [Hour 7](docs/hour7.md), [Hour 8A](docs/hour8a.md), and [Hour 9](docs/hour9.md).

### Stage II — validated architecture, Days 1–7

Beginning at `23a5407`, the project introduced a modular student interface and independent validators, then progressed through a validated search oracle, genuine expert-imitation transfer, an interpretable 18-feature reinforcement-learning student, history-only diagnosis, targeted situation generation, and the equal-budget comparison at `ce04a1c`.

| Milestone | Contribution | Documentation |
| --- | --- | --- |
| Day 1 | Modular student contract and independent validation | [Architecture](docs/day1_architecture.md) |
| Day 2 | Validated simulator-native Dellacherie search oracle | [Oracle](docs/day2_expert.md) |
| Day 3 | Expert labels transfer into an expert-free student policy | [Imitation](docs/day3_transfer.md) |
| Day 4 | 18-feature learner trained by sequential episodic RL | [RL](docs/day4_rl.md) |
| Day 5 | Weakness profiles inferred from replayable play history | [Diagnosis](docs/day5_diagnosis.md) |
| Day 6 | Replayable, profile-selected one-placement practice | [Tutorials](docs/day6_tutorials.md) |
| Day 7 | Ordinary, imitation, and personalized training at equal budgets | [Comparison](docs/day7_comparison.md) |

Phase 2 extends that comparison with a preregistered responsive full-profile tutor. See the [Phase 2 report](docs/phase2_adaptive_tutor.md), [locked configuration](experiments/phase2/preregistration.json), [frozen results](experiments/phase2/final_results.json), and [four-board replay demo](demo/phase2-comparison.html).

## What the terms mean

- **Student:** the trainable machine policy being taught and evaluated.
- **Oracle / expert:** the independently validated search policy used to label imitation examples or score decisions. Its preferences are a simulator-specific reference, not ground truth.
- **Placement:** one complete legal hard-drop action `(rotation, x)` for the current tetromino.
- **Exposure:** one placement and learning opportunity. **One exposure is not one full game.** Day 7's budget was 500 placements/updates per condition.
- **Training budget:** the matched number of exposures and updates, not CPU time, games, or resets.
- **Ordinary practice:** sequential play using the student's existing RL update and environment reward.
- **Direct imitation:** student-owned supervised updates toward oracle-selected placements.
- **Personalized tutorial:** practice situations allocated from a weakness profile, learned through the ordinary RL path without oracle labels.
- **Regret:** oracle-best heuristic value minus the value of the student's action; it is relative to this oracle, not eventual gameplay ground truth.
- **Expert agreement:** the fraction of held-out decisions on which student and oracle choose the same placement.
- **Weakness profile:** history-derived scores for recurring feature families; correlated categories are descriptive hypotheses, not causally independent defects.

## Watch the Day 4 classroom demo

The [four-board animated demo](demo/day4_rl_oracle_comparison.html) replays the untrained learner, RL after 750 exposures, RL after 3,000 exposures, and the Day 2 oracle on the same preregistered held-out piece stream. Learning is disabled during replay. The recorded outcomes are:

| Policy | Replay outcome |
| --- | --- |
| Untrained | 25 placements / 0 lines / game over |
| RL@750 | 70 placements / 12 lines / game over |
| RL@3000 | 300 placements / 108 lines / cap |
| Oracle | 300 placements / 116 lines / cap |

Launch locally, then open <http://127.0.0.1:8765/day4_rl_oracle_comparison.html>:

```bash
python hour8a_demo.py
```

The earlier Stage I presentation is [also available](demo/index.html); it illustrates one prespecified Hour 7 prototype replicate and is not the aggregate result.

## Inspect and reproduce

The repository has no runtime dependencies beyond Python's standard library. Run the complete test suite with:

```bash
python -m unittest discover -s tests -v
```

Each detailed report gives its exact rerun and validation commands. Final scientific outputs live under [`experiments/`](experiments/); prototype outputs live under [`artifacts/`](artifacts/). Core Stage II modules include the [student contract](tetris_research/student.py), [oracle](tetris_research/expert.py), [richer student](tetris_research/richer_student.py), [diagnosis](tetris_research/diagnosis.py), and [tutorial generator](tetris_research/tutorials.py). Reruns should write to new paths where a command supports `--output`; do not overwrite committed artifacts.

## Interpretation and limitations

Day 7 used a **static personalized curriculum** across the full 500-exposure budget: one shared profile determined the allocation before training, and weaknesses were not reassessed. A natural future extension is an adaptive loop:

```text
diagnose → short targeted practice → reassess → retarget
```

That loop is future work, not an explanation proven by Day 7. The project also does **not** establish that 500 machine placements correspond to a human training timescale, that weakness families are causally independent, or that the current static allocation is optimal. Results come from eight training replicates, one initial checkpoint, a linear machine student, one oracle, and a simplified 10×20 hard-drop simulator without hold, preview, SRS movement, or modern scoring. Human studies and broader model/seed/checkpoint validation are still required.
