# Hour 4: skill-sensitive evaluation

Hour 4 asked whether history-selected practice improved a diagnosed observable skill and ordinary held-out play more than control or rating-only practice. It replaced Hour 3's sparse primary line-clear measure with successful placements before top-out, while preserving lines as secondary.

## Design

In each of 50 replicates (master seed `400004`), a baseline four-feature `LearningAgent` accumulated 40 ordinary-practice placements of observable history. Three clones then began with identical weights, Elo, RNG state, and history: `control` received ordinary practice; `rating_only` received an Elo-selected tutorial board; and `rating_history` received a board selected from Elo plus history diagnosis.

All conditions received exactly 40 training placements through the same learning rule and matched piece stream. A placement was one complete tetromino action and learning opportunity, not a game. Before and after training, disposable non-learning clones played the same six held-out streams per replicate, capped at 120 successful placements each. Primary performance was mean successful placements across the six challenges. Secondary lines and trajectory-averaged holes, maximum height, bumpiness, and line efficiency supplied component scores.

The design was frozen before a three-replicate smoke gate and the confirmatory run. Analysis used paired differences, t-based 95% CIs, two-sided paired t tests, Cohen's `dz`, and exact two-sided Wilcoxon signed-rank tests. Seed domains were separated; evaluation did not mutate learners. Hour 3 artifacts were hash-checked and not rescored.

## Results

Mean placement improvements were `+0.2033` control, `+0.1200` rating-only, and `+0.1433` rating+history. History minus control was `−0.0600` (95% CI `[−0.5087, 0.3887]`, p=`0.7893`); history minus rating-only was `+0.0233` (`[−0.3407, 0.3873]`, p=`0.8980`). Lines were sparse: 89.44% of all 1,800 challenge observations cleared zero.

Every history was diagnosed as hole avoidance. Its target score improved `+0.7339` (`[−0.2528, 1.7205]`), an uncertain favorable direction. Hour 4 therefore established neither targeted learning nor general transfer. The placement metric itself was usable, but the lack of diagnostic diversity prevented cross-skill validation. Hour 5 later showed that incomparable hand-set severity scales forced the all-hole result.

## Reproduce and inspect

```bash
python hour4_experiment.py --replicates 3 --master-seed 400004 --output /tmp/hour4-smoke
python hour4_experiment.py --replicates 50 --master-seed 400004 --output /tmp/hour4-confirmatory
```

See the [frozen specification](../artifacts/hour4/evaluation_specification.json), [full report](../artifacts/hour4/hour4_report.md), [statistics](../artifacts/hour4/experiment_50/statistics.json), [raw challenge records](../artifacts/hour4/experiment_50/challenge_results.jsonl), [implementation](../tetris_research/hour4.py), and [tests](../tests/test_hour4.py).
