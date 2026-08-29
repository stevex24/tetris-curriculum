# Hour 6: calibrated personalized training

Hour 6 was the first fresh effectiveness experiment using the calibrated Hour 5 profile. It asked whether history-aware tutorial selection improved ordinary held-out survival more than ordinary practice or an Elo-only tutorial, and whether the diagnosed target skill itself improved.

## Locked procedure

The confirmatory run used 50 matched replicates and master seed `600006`. Each baseline produced 40 ordinary-play placements of observable history. Diagnosis reused the committed 80-history calibration (seed `500005`) without regeneration. Baselines were cloned into ordinary control, rating-only, and rating+history conditions with identical weights, Elo, RNG state, learning state, and history.

Each condition first played six matched held-out challenges through disposable non-learning clones, then received exactly 40 placements/updates on a matched training stream, then replayed the held-out streams. Each challenge was capped at 120 placements. One exposure meant one tetromino placement and ordinary learner update, not a game. The primary outcome was change in mean successful placements per challenge. Diagnosed target change was standardized by the corresponding frozen calibration SD.

Inference used paired mean differences, t-based 95% CIs, paired t tests, Cohen's `dz`, and exact Wilcoxon signed-rank tests. A three-replicate smoke run checked integrity and resolution without using treatment direction.

## Results

History-aware minus control was `+0.487` placements/challenge (95% CI `[0.096, 0.878]`, p=`0.0157`); history-aware minus rating-only was `+0.407` (`[0.078, 0.736]`, p=`0.0165`). Diagnoses were holes 23, height 13, and surface 14; history-aware material differed from rating-only material in all 50 replicates.

The intended mechanism was not demonstrated. Overall standardized target improvement was `+0.0028` SD (`[−0.2555, 0.2610]`), and every diagnosis-specific target CI included zero. Hour 6 therefore supported a small general-performance advantage in this run, but not diagnosed-skill learning or causal transfer. The locked Hour 7 replication subsequently failed to reproduce the control comparison.

## Reproduce and inspect

```bash
python hour6_experiment.py --phase smoke --replicates 3 --output /tmp/hour6-smoke
python hour6_experiment.py --phase confirmatory --replicates 50 --output /tmp/hour6-results
```

See the [specification](../artifacts/hour6/experimental_specification.json), [full report](../artifacts/hour6/hour6_report.md), [statistics](../artifacts/hour6/results/statistical_summary.json), [raw challenges](../artifacts/hour6/results/challenge_results.jsonl), [implementation](../tetris_research/hour6.py), and [tests](../tests/test_hour6.py).
