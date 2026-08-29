# Hour 5: calibrated learner profiles

Hour 5 audited why Hour 4 diagnosed hole avoidance in all 50 learners, then built a calibrated observable profile. It was a measurement and construct-validation milestone, not a tutorial-effectiveness experiment.

## Audit and method

Hour 4 ranked four incompatible severity scales. Hole severity ranged `1.1338–2.1913`, always above height (`0.5138–0.6675`), bumpiness (`0.3163–0.6300`), and line efficiency (`0.75–1.0`). The latter was also saturated for 46 of 50 histories. Thus the all-hole diagnosis was a scaling artifact, not evidence of a universal learner weakness.

Hour 5 generated 80 independent 40-placement baseline histories with calibration seed `500005`. It retained observable mean holes, maximum height, bumpiness, line-clear rate, lines, placement count, and Elo, then expressed holes, height, and surface weakness as z-scores against calibration means and sample SDs. Hidden weights were excluded. The largest two z-scores became primary and secondary diagnoses; their margin defined low/mixed (`<0.35`), moderate (`0.35–0.75`), or high (`≥0.75`) confidence. Sparse line efficiency remained descriptive only.

A separate natural diagnostic population used seed `500105`, 50 histories, and the same 40-placement observation length. It produced 26 hole, 14 surface, and 10 height primary diagnoses, but these were seed-driven variations of one learner process, not proven stable learner types. Synthetic histories independently raised one observable and recovered all three intended labels; these probes validate measurement discrimination, not ecological validity.

No tutorial training or held-out evaluation occurred. There is consequently no exposure comparison, gameplay cap, or effectiveness statistic for Hour 5.

## Validation, limits, and reproduction

Raw observables remain in every `LearnerProfile`; calibration and diagnostic seed domains are separate; synthetic labels never enter diagnosis; and tests check hidden-weight exclusion, calibration, confidence boundaries, and tutorial mapping. Correlated features, noisy 40-placement histories, population-relative z-scores, and provisional low-margin mappings remain limitations.

```bash
python hour5_diagnostic.py --output /tmp/hour5-demo --calibration-count 80 --diagnostic-count 50
python -m unittest tests.test_hour5 -v
```

See the [diagnostic audit](../artifacts/hour5/hour4_diagnostic_audit.md), [full report](../artifacts/hour5/hour5_report.md), [calibration parameters](../artifacts/hour5/demo/calibration_parameters.json), [construct results](../artifacts/hour5/demo/construct_validation_and_demo.json), [implementation](../tetris_research/hour5.py), and [tests](../tests/test_hour5.py).
