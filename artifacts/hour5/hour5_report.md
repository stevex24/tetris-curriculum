# Hour 5 learner-profile and diagnostic validation

## Result boundary

Hour 5 establishes that observable hole, height, and surface measurements can be retained in an
explicit profile, placed on a common empirical scale, used to recover three independently
constructed weakness patterns, and mapped to three existing tutorials. It does **not** establish
that the naturally generated agents form stable learner types, that any tutorial is effective, or
that personalization improves rating or play. No tutorial-effectiveness experiment was run.

## Why Hour 4 collapsed

The separate audit is `artifacts/hour5/hour4_diagnostic_audit.md`. Hour 4 directly ranked mean
holes/20, mean maximum height/20, mean bumpiness/40, and `max(0, 1 - 10 * line-clear rate)`.
Across its 50 histories the respective severity ranges were 1.1338–2.1913, 0.5138–0.6675,
0.3163–0.6300, and 0.75–1.0. Thus holes exceeded every other severity in every history. The
constants had no shared population interpretation; line efficiency was also saturated at 1.0 in
46/50 histories. This was a numerical scale failure, not evidence that every agent's true dominant
weakness was hole avoidance.

The corresponding raw distributions (min, Q1, median, Q3, max; mean, sample SD) were:

| Observable | Distribution across the 50 frozen Hour 4 histories |
|---|---|
| mean holes | 22.675, 28.844, 31.538, 33.625, 43.825; 31.7015, 4.2238 |
| mean maximum height | 10.275, 11.169, 12.000, 12.463, 13.350; 11.8775, 0.7812 |
| mean bumpiness | 12.650, 16.781, 17.788, 19.838, 25.200; 18.2065, 2.4432 |
| total lines in 40 placements | 0 in 46 histories and 1 in 4; 0.08, 0.2740 |

## Profile and calibration

`LearnerProfile` contains only observable information: `ability_elo`, `mean_holes`,
`mean_max_height`, `mean_bumpiness`, `line_clear_rate`, `lines_cleared`,
`placements_observed`, and derived `normalized_weakness`. Elo is an observable game-outcome rating.
Raw values remain present after normalization. There is no field for policy weights or other hidden
agent state.

Calibration used 80 newly generated 40-placement baseline histories, master seed **500005**. These
histories use the existing initial policy and learning process, but calibration identifiers and seed
domains are separate from the 50-history diagnostic population (master seed **500105**) and from
any future effectiveness experiment. Saved means were holes 30.5347, height 11.6588, bumpiness
18.2234; sample SDs were 4.5715, 0.8211, and 3.0283.

For each diagnostic dimension, normalized weakness is the ordinary z-score
`(player raw value - calibration mean) / calibration sample SD`. Larger is worse for all three. A
score of +1 means the player's 40-placement average was one calibration-population sample SD worse
than the calibration mean. This makes the units comparable without using tutorial outcomes.

Primary and secondary diagnoses are the largest two z-scores. Their difference is the margin.
Margins below 0.35 are reported `low/mixed`; 0.35–0.75 are `moderate`; at least 0.75 are `high`.
The threshold is an explicit interpretive convention, not an effectiveness-tuned constant.

## Natural diagnostic population

The independent 50-history population varied observably:

| Measure | min | mean | max | sample SD |
|---|---:|---:|---:|---:|
| mean holes | 20.350 | 31.385 | 46.725 | 4.543 |
| mean maximum height | 10.075 | 11.716 | 13.375 | 0.681 |
| mean bumpiness | 10.625 | 17.870 | 26.075 | 2.903 |
| line-clear rate | 0 | 0.0025 | 0.05 | 0.0104 |

Pearson correlations were:

| | holes | height | surface |
|---|---:|---:|---:|
| holes | 1.000 | 0.526 | -0.063 |
| height | 0.526 | 1.000 | 0.500 |
| surface | -0.063 | 0.500 | 1.000 |

All eight above/below-calibration-mean sign patterns occurred. Relative primary scores were holes
26, surface 14, and height 10. These counts are descriptive, not a diversity target. Because every
agent shares the same starting policy and learning process and differs principally by random seeds,
this is meaningful *measurement variation* but not evidence of stable or pedagogically distinct
natural learner types. Replicated histories per learner would be needed to separate stable skill
patterns from stream noise.

## Construct validity and demonstration

Synthetic constructs are unit tests only. Before classification, the generator fixed a baseline
center `(31.7 holes, 11.9 height, 18.2 bumpiness)` and independently raised exactly one target:
holes to 43.0, height to 15.5, or bumpiness to 28.0. Each of 40 steps receives fixed-seed symmetric
discrete jitter. The mixed construct raises holes to 40.0 and height to 13.45. Labels are never
included in the step records passed to measurement or diagnosis.

Confusion matrix (rows known construction, columns diagnosis):

| | holes | height | surface |
|---|---:|---:|---:|
| hole-management-poor | 1 | 0 | 0 |
| height-management-poor | 0 | 1 | 0 |
| surface-management-poor | 0 | 0 | 1 |

Five PROFILE -> DIAGNOSIS -> TUTORIAL examples follow. Values in parentheses are normalized
weaknesses ordered as holes, height, surface.

| Profile (raw means) | Scores | Diagnosis | Margin/confidence | Existing tutorial and justification |
|---|---|---|---|---|
| holes 42.825, height 11.838, bumpiness 18.300 | (2.688, 0.218, 0.025) | holes primary, height secondary | 2.471, high | `hole_avoidance`; contrasts placements that create cavities |
| holes 31.400, height 15.613, bumpiness 17.900 | (0.189, 4.815, -0.107) | height primary, holes secondary | 4.626, high | `stack_height`; makes vertical risk and stack reduction salient |
| holes 31.575, height 12.050, bumpiness 27.700 | (0.228, 0.476, 3.129) | surface primary, height secondary | 2.653, high | `bumpiness`; contrasts smooth and uneven landing surfaces |
| holes 39.800, height 13.587, bumpiness 18.175 | (2.027, 2.349, -0.016) | height primary, holes secondary, mixed | 0.322, low/mixed | provisional `stack_height`; selection records the near-tie and must not be read as a confident single-defect claim |
| holes 31.850, height 11.900, bumpiness 17.925 | (0.288, 0.294, -0.099) | height primary, holes secondary, mixed | 0.006, low/mixed | provisional `stack_height`; effectively ambiguous rather than strong personalization |

## Line-clearing limitation and concerns

Line efficiency remains in the raw profile but is excluded from primary diagnosis. In the natural
diagnostic population its mean was 0.0025 lines per placement, with most histories at zero. It lacks
enough resolution to support an independent weakness claim. Future work may require longer/richer
histories and observable opportunity-sensitive subskills such as completing available rows,
maintaining wells, and converting setups, plus simulator capabilities such as preview/hold; that
taxonomy is not built here.

Other concerns are that board measurements are correlated, z-scores are relative to this prototype
population rather than pedagogical standards, 40 placements may be noisy, and selecting a tutorial
for a low-margin profile is only a provisional mapping. The construct probes establish software and
measurement discrimination, not ecological validity or tutorial benefit.

## Reproduction

```bash
python hour5_diagnostic.py --output artifacts/hour5/demo --calibration-count 80 --diagnostic-count 50
python -m unittest discover -s tests -v
```
