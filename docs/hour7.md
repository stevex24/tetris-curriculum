# Hour 7: locked replication

Hour 7 independently repeated Hour 6's frozen calibrated personalized-training experiment with fresh randomness. Its scientific implementation, calibration, conditions, budgets, endpoints, and analysis were unchanged.

## Procedure

The single prespecified run used master seed `700007`, 50 matched replicates, 40 baseline-history placements, 40 training placements per condition, six matched held-out challenges, and a 120-placement evaluation cap. The conditions were ordinary control, Elo-only tutorial, and calibrated history-aware tutorial. Evaluation used disposable non-learning clones on matched streams. One exposure was one tetromino placement/update, not one game.

The specification was locked before tests, a three-replicate integrity smoke run (`700107`), and outcomes. All 38 tests passed before collection. The exact Hour 5 calibration and Hour 6 implementation were reused; seed domains were disjoint and earlier artifacts were checked for integrity.

## Results

Mean improvement was `+0.287` placements/challenge for control, `−0.077` for rating-only, and `+0.143` for history-aware. History minus control was `−0.143` (95% CI `[−0.541, 0.254]`, p=`0.4725`), reversing Hour 6's point direction. History minus rating-only was `+0.220` (`[−0.177, 0.617]`, p=`0.2704`), directionally consistent but smaller and uncertain.

Diagnoses were holes 20, height 15, and surface 15; all mapped to the intended tutorial and differed from rating-only material. Overall standardized target improvement was `+0.121` SD (`[−0.088, 0.330]`); every target-specific CI included zero. The stronger target/general correlations remained exploratory and noncausal.

Hour 7 did not independently reproduce the history-over-control advantage and left the rating-only contrast inconclusive. The later exploratory Hour 6+7 fixed-effect summary did not resolve the control disagreement. This outcome was the key reason to run Hour 9 at much larger sample size.

## Reproduce and inspect

```bash
python hour7_experiment.py --phase smoke --replicates 3 --output /tmp/hour7-smoke
python hour7_experiment.py --phase confirmatory --replicates 50 --output /tmp/hour7-results
```

See the [locked specification](../artifacts/hour7/locked_replication_specification.json), [full report](../artifacts/hour7/hour7_report.md), [statistics](../artifacts/hour7/results/statistical_summary.json), [raw challenges](../artifacts/hour7/results/challenge_results.jsonl), [implementation](../tetris_research/hour7.py), and [tests](../tests/test_hour7.py).
