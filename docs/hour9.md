# Hour 9: large-sample tutorial-effectiveness test

Hour 9 resolved the inconsistent Hour 6 and Hour 7 prototype estimates by running their frozen design at 1,000 matched replicates. No mechanism, tutorial, reward, calibration, metric, or analysis was redesigned after outcomes.

## Frozen design

The experiment used master seed `900009`. Every replicate retained the Stage I procedure: 40 ordinary placements of baseline history; calibrated diagnosis; identical clones for ordinary control, rating-only tutorial, and history-aware tutorial; six matched non-learning held-out challenges before and after 40 training placements per condition; and a cap of 120 placements per challenge. One exposure was one placement/update, not a game.

The primary measure was improvement in mean successful placements per challenge. Paired history-minus-control and history-minus-rating-only contrasts used t-based 95% CIs, paired t tests, Cohen's `dz`, and exact Wilcoxon tests. The run included a predetermined 10-replicate validation/timing phase, the 1,000-replicate experiment, and an independent behavioral reconstruction of a fixed 10-replicate sample. Forty-three tests and all behavioral checks passed.

## Results

Mean improvements were `+0.0453` control, `+0.0735` rating-only, and `+0.0467` history-aware placements/challenge. History minus control was `+0.0013` (95% CI `[−0.0825, 0.0852]`, p=`0.9751`); history minus rating-only was `−0.0268` (`[−0.1052, 0.0515]`, p=`0.5019`). The intervals were 4.2–5.1 times narrower than the Hour 6/7 intervals. Thus neither earlier apparent general-performance advantage persisted with high precision.

The overall standardized target-skill change was `+0.0042` (`[−0.0424, 0.0508]`). Only the hole subgroup had a small positive interval excluding zero; this did not establish broad targeted teaching. Exploratory target/general correlations remained noncausal. Secondary line improvements were slightly negative in all conditions.

Hour 9 is the proper endpoint for the prototype evidence: the history-aware general advantage was not stable. This negative result motivated, rather than validated, the more rigorous Stage II architecture.

## Reproduce and inspect

```bash
python -m unittest discover -s tests -v
python hour9_experiment.py --phase validate --replicates 10 --output /tmp/hour9-reproduction
python hour9_experiment.py --phase timing --replicates 10 --output /tmp/hour9-reproduction
python hour9_experiment.py --phase experiment --replicates 1000 --output /tmp/hour9-reproduction
python hour9_experiment.py --phase finalize --replicates 1000 --output /tmp/hour9-reproduction
```

See the [full report](../artifacts/hour9_large_sample/final_report.md), [statistics](../artifacts/hour9_large_sample/results/statistical_summary.json), [behavioral validation](../artifacts/hour9_large_sample/behavioral_validation.json), [raw challenges](../artifacts/hour9_large_sample/results/challenge_results.jsonl), [implementation](../tetris_research/hour9.py), and [runner](../hour9_experiment.py).
