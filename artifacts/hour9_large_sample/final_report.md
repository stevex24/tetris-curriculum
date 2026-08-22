# Large-sample frozen-design experiment

Master seed: **900009**; matched replicates: **1,000**.
Tests: **pass** (43 tests). Behavioral validator: **pass**.
Measured full experiment runtime: **2,151.485 seconds (35.86 minutes)**; frozen-runner core time was 2,027.386 seconds. Ten-replicate estimate: 33.68 minutes.

## General performance

| Condition | Pre | Post | Improvement | 95% CI |
|---|---:|---:|---:|---:|
| control | 27.224 | 27.269 | 0.045 | [-0.059, 0.149] |
| rating_only | 27.224 | 27.297 | 0.074 | [-0.031, 0.178] |
| rating_history | 27.224 | 27.270 | 0.047 | [-0.057, 0.151] |

## Primary paired comparisons

| Comparison | Mean additional successful pieces/challenge | SE | 95% CI | t(df) | p | dz | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|---:|
| rating_history_minus_control | 0.0013 | 0.0427 | [-0.0825, 0.0852] | 0.031 (999) | 0.975108 | 0.001 | 0.671726 |
| rating_history_minus_rating_only | -0.0268 | 0.0399 | [-0.1052, 0.0515] | -0.672 (999) | 0.501859 | -0.021 | 0.566214 |

## Hour 6 / Hour 7 / n=1000 and precision

| Comparison | Hour 6 | Hour 7 | n=1000 | CI narrower vs H6 / H7 |
|---|---:|---:|---:|---:|
| rating_history_minus_control | 0.487 | -0.143 | 0.001 | 4.66x / 4.75x |
| rating_history_minus_rating_only | 0.407 | 0.220 | -0.027 | 4.20x / 5.06x |

The history-minus-control estimate is essentially zero and intermediate between Hour 6's positive estimate and Hour 7's negative estimate. The history-minus-rating estimate is slightly reversed relative to both prior positive estimates. The much narrower intervals show that neither apparent general-performance advantage persisted with high precision.

## Diagnosis distribution

Primary: {"height_management": 264, "hole_management": 413, "surface_management": 323}. Confidence: {"high": 348, "low/mixed": 357, "moderate": 295}. Mixed: {"false": 643, "true": 357}.
Tutorial mapping: {"height_management -> stack_height": 264, "hole_management -> hole_avoidance": 413, "surface_management -> bumpiness": 323}.

## Target skills

Overall standardized target-skill change: 0.0042 (95% CI [-0.042433045322495776, 0.050766974773915625]).

| Target | n | Pre | Post | Improvement [95% CI] |
|---|---:|---:|---:|---:|
| height_management | 264 | -13.335 | -13.354 | -0.018 [-0.083, 0.047] |
| hole_management | 413 | -38.023 | -37.660 | 0.364 [0.012, 0.715] |
| surface_management | 323 | -18.859 | -19.073 | -0.214 [-0.467, 0.038] |

## Exploratory transfer

Pearson: r=0.2375, 95% Fisher CI=[0.1781383007213819, 0.295155739260422], p=2.73161e-14. Exploratory; no causal inference.
Spearman: r=0.2220, 95% Fisher CI=[0.16220586997499922, 0.2801056452189339], p=1.25506e-12. Exploratory; no causal inference.

## Secondary lines cleared

| Condition | Pre | Post | Improvement | Zero proportion pre/post |
|---|---:|---:|---:|---:|
| control | 0.1425 | 0.1372 | -0.0053 | 0.455 / 0.481 |
| rating_only | 0.1425 | 0.1408 | -0.0017 | 0.455 / 0.465 |
| rating_history | 0.1425 | 0.1340 | -0.0085 | 0.455 / 0.492 |

## Interpretation

Within this frozen simulation, the large experiment falsifies the apparent history-aware general-performance advantage as a stable effect of educational importance: history-control is essentially zero, and history-rating is a tiny reversal. This conclusion follows from magnitudes and narrow uncertainty intervals, not from statistical significance alone.

The hole-management subgroup has a small positive target-skill estimate whose CI excludes zero; the overall standardized target effect and the height and surface subgroup intervals include zero. This is limited evidence for one nominal skill, not evidence that the tutorial system broadly teaches its intended skills.

The positive target/general correlations are exploratory associations and do not establish causal transfer. The frozen simulation does not establish real-player effectiveness, causal educational mechanisms, or generalization beyond this implementation.

No mechanism, tutorial board, reward, policy weight, calibration, or analysis was redesigned after observing results.

## Exact reproduction command

```bash
python -m unittest discover -s tests -v
python hour9_experiment.py --phase validate --replicates 10 --output artifacts/hour9_large_sample_reproduction
python hour9_experiment.py --phase timing --replicates 10 --output artifacts/hour9_large_sample_reproduction
python hour9_experiment.py --phase experiment --replicates 1000 --output artifacts/hour9_large_sample_reproduction
python hour9_experiment.py --phase finalize --replicates 1000 --output artifacts/hour9_large_sample_reproduction
```
