# Hour 6 fresh controlled training-effectiveness experiment

## Result

With the frozen Hour 5 profile machinery and 50 fresh matched replicates, history-aware training
improved the primary general outcome more than ordinary practice by **0.487 placements/challenge**
(95% CI 0.096 to 0.878, paired t(49)=2.502, two-sided p=0.0157, Cohen's dz=0.354) and more than
rating-only training by **0.407** (95% CI 0.078 to 0.736, t(49)=2.483, p=0.0165, dz=0.351).
Exact signed-rank results were respectively p=0.0204 and p=0.0136.

The intended target skills were **not demonstrably taught**. The predeclared overall standardized
target improvement was 0.0028 frozen-calibration SD (95% CI -0.2555 to 0.2610). None of the three
diagnosis-specific raw target-improvement CIs excluded zero. Thus Hour 6 supports a general
performance advantage in this experiment but does not support the intended target-learning
mechanism or demonstrate transfer from target learning.

## Frozen procedure and provenance

The specification was saved before outcome collection in `artifacts/hour6/experimental_specification.json`.
The master seed was **600006**, a new Hour-6-specific seed domain disjoint from the Hour 3–5 master
seeds. The exact committed Hour 5 calibration was reused from
`artifacts/hour5/demo/calibration_parameters.json`: calibration seed **500005**, 80 independent
40-placement histories, SHA-256
`c69823decfe715ffbb8ec54885783881cef547bbbc1b59bb6a2de49bf114e476`. It was not regenerated.

For each replicate, a baseline agent generated 40 ordinary-practice placements. Its observable
history supplied raw holes, maximum height, bumpiness, and secondary line data. Hole, height, and
surface weakness were normalized using the frozen Hour 5 means and sample SDs; line efficiency
remained under-resolved and outside primary diagnosis. The largest and second-largest z-scores set
the primary and secondary weaknesses. A margin below 0.35 was kept as low/mixed; the frozen rule
used the primary tutorial while preserving the flag and retaining the replicate in all analyses.

The baseline agent was cloned into control, rating_only, and rating_history. Policy weights, Elo,
agent RNG state, learned-game state, and prior observable history were verified identical. All
clones received six matched unseen evaluation challenges with learning disabled, then exactly 40
training placements on the same piece-stream seed. Control received ordinary practice; rating_only
received the existing Elo-tier board without history; rating_history received the existing tutorial
mapped from the calibrated primary diagnosis. The same six held-out challenge seeds were used for
post-training matched evaluation, on disposable evaluation clones. Evaluation did not mutate
training state, weights, history, Elo, or diagnosis.

Target skill was evaluated from held-out challenge behavior, never from diagnostic history:
negative mean holes for hole avoidance, negative mean maximum height for height management, and
negative mean bumpiness for surface management. Higher is better. Overall target change divided
each replicate's raw improvement by the corresponding Hour 5 calibration SD fixed before Hour 6.

The complete 34-test suite passed before data collection. The three-replicate smoke test then had
general scores 19–34, 14 distinct general scores, no zero/cap scores, computable distinct target
scores, valid diagnosis/tutorial mappings, and all leakage/budget controls true. Treatment outcome
direction was not used for the smoke decision. The frozen 50-replicate experiment was then run once.

## Diagnosis distribution and tutorial specificity

Primary diagnoses were holes 23, height 13, and surface 14. Secondary diagnoses were holes 11,
height 29, and surface 10. Confidence was low/mixed 23, moderate 11, and high 16: **23 mixed and 27
non-mixed**.

Normalized weakness distributions (mean, sample SD, 95% CI) were holes -0.027, 0.947,
[-0.297, 0.242]; height -0.006, 0.943, [-0.274, 0.262]; and surface -0.304, 0.986,
[-0.584, -0.024]. Raw observable profiles, all z-scores, margins, confidence, primary/secondary
diagnoses, and tutorials are retained at replicate level.

| Diagnosis | Selected history-aware tutorial | n |
|---|---|---:|
| hole management | hole avoidance | 23 |
| height management | stack height | 13 |
| surface management | bumpiness | 14 |

Every diagnosis mapped to its intended distinct tutorial. The history-aware treatment differed
from the rating-only Elo-tier material in **50/50** replicates, so interpretation does not fail for
lack of treatment differentiation.

Descriptively by confidence, standardized target/general mean improvements were: low/mixed (n=23)
0.095/0.217; moderate (n=11) 0.272/-0.045; and high (n=16) -0.314/0.833. These strata were not used
for exclusion or confirmatory subgroup inference.

## General performance

Values are mean successful placements per six held-out challenges. Improvement is post minus pre.

| Condition | Pre mean | Post mean | Mean improvement | Improvement 95% CI |
|---|---:|---:|---:|---:|
| control | 27.257 | 27.127 | -0.130 | [-0.653, 0.393] |
| rating_only | 27.257 | 27.207 | -0.050 | [-0.493, 0.393] |
| rating_history | 27.257 | 27.613 | 0.357 | [-0.178, 0.892] |

The paired history-control difference was 0.487 [0.096, 0.878], t(49)=2.502, p=0.0157,
dz=0.354; exact Wilcoxon W+=581 with 40 nonzero pairs and p=0.0204. The paired history-rating
difference was 0.407 [0.078, 0.736], t(49)=2.483, p=0.0165, dz=0.351; exact Wilcoxon W+=440 with
34 nonzero pairs and p=0.0136.

## Target-skill learning

The valid frozen-SD aggregate had mean standardized improvement 0.0028 (SD 0.9086, 95% CI
-0.2555 to 0.2610). This scale combines only change scores after dividing by an independently fixed
dimension-specific calibration SD; raw unlike metrics are not pooled.

| Diagnosis / target (higher is better) | n | Pre | Post | Improvement mean (SD) | 95% CI |
|---|---:|---:|---:|---:|---:|
| holes / negative mean holes | 23 | -37.838 | -38.075 | -0.237 (4.043) | [-1.985, 1.512] |
| height / negative mean max height | 13 | -13.476 | -13.318 | 0.158 (0.635) | [-0.225, 0.542] |
| surface / negative mean bumpiness | 14 | -18.901 | -19.155 | -0.255 (3.298) | [-2.159, 1.650] |

These samples are reported separately because their raw scales differ. None demonstrates improvement
of its intended skill at conventional inferential resolution. The tutorials therefore did not
demonstrably teach their diagnosed targets in Hour 6.

## Transfer and secondary lines

Within rating_history, standardized target improvement and general improvement had Pearson r=0.244
(t(48)=1.744, two-sided p=0.0875) and Spearman rho=0.264 (p=0.0644). This exploratory positive
association is uncertain and cannot establish causation. Because target improvement itself was not
demonstrated, Hour 6 does **not** demonstrate transfer from intended target learning to general play.

Lines cleared remained secondary. Mean pre/post/improvement values were control 0.157/0.153/-0.003,
rating_only 0.157/0.127/-0.030, and rating_history 0.157/0.187/0.030 lines per challenge. The
replicate-mean zero proportions pre/post were control 0.40/0.40, rating_only 0.40/0.44, and
rating_history 0.40/0.32. These results do not replace or redefine the primary outcome.

## Interpretation boundaries and methodological findings

Hour 6 justifies that, under this exact prototype, calibrated history-aware tutorial selection led
to a small positive matched advantage on held-out successful placements versus both ordinary
practice and rating-only material. It also justifies that treatment differentiation and diagnostic
diversity were present, and that the intended target-skill effect was not detected.

It does **not** justify that the tutorials taught the diagnosed skills, that target learning
transferred, that the exploratory target/general correlation is causal, that personalization will
generalize beyond these agents/seeds/tutorial boards, that the diagnoses are stable learner traits,
or that lines cleared is a superior endpoint. The apparent general advantage alongside null target
learning means the mechanism may differ from the intended one. Additional limitations are noisy
40-placement histories, correlated observable dimensions, relative rather than pedagogical z-score
norms, only six evaluation challenges, and target trajectory metrics that summarize play on the
general held-out challenges rather than a separately engineered skill test. No thresholds, learning
rates, tutorials, metrics, seeds, sample size, or analysis choices were changed after outcomes.

## Files and reproduction

Implementation: `tetris_research/hour6.py`, runner: `hour6_experiment.py`, and validity tests:
`tests/test_hour6.py`. The `artifacts/hour6` tree contains the frozen specification, this report,
replicate CSV, challenge JSONL, diagnosis/profile CSV, target CSV, tutorial table, statistical
summary, transfer analysis, confidence description, and complete configuration/seed/calibration
provenance. Hours 3–5 artifacts passed byte-for-byte commit checks and were not overwritten.

Exact reproduction commands from repository root:

```bash
python -m unittest discover -s tests -v
python hour6_experiment.py --phase smoke --replicates 3 --output /private/tmp/hour6-smoke-600006
python hour6_experiment.py --phase confirmatory --replicates 50 --output artifacts/hour6/results
```

The final command requires an absent output directory because artifact overwrite is prohibited.
