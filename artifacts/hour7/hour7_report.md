# Hour 7 locked independent replication

## Result

The single prespecified 50-replicate run did **not closely reproduce both Hour 6 effects**. History-aware improvement minus control was **-0.143 placements/challenge** (95% CI -0.541 to 0.254, paired t(49)=-0.724, two-sided p=0.4725, Cohen's dz=-0.102; exact Wilcoxon p=0.5483). The direction reversed, and the Hour 6 estimate 0.487 is outside the Hour 7 CI, although the two CIs overlap. This weakens the Hour 6 control comparison and creates some cross-experiment tension.

History-aware improvement minus rating-only was **0.220** (95% CI -0.177 to 0.617, t(49)=1.115, p=0.2704, dz=0.158; exact Wilcoxon p=0.4130). Its direction agrees with Hour 6, its CI contains both zero and the Hour 6 estimate 0.407, and the effect is smaller. The estimates are reasonably compatible given uncertainty, but Hour 7 alone is inconclusive. This interpretation is based on estimates and uncertainty, not a binary p=.05 rule.

## Locked design, execution, and integrity

The replication specification was frozen before testing or outcome collection. Master seed **700007** was selected once as the sequential Hour 7 seed namespace and was disjoint from prior master seeds 300003, 400004, 500005, 500105, and 600006. The one smoke seed was 700107. Experimental seed derivation used the unchanged Hour 6 procedure.

Hour 7 delegated to the scientific implementation frozen at commit `3e71ee2`. The only experimental change was fresh master-seed randomness. It reused the exact committed Hour 5 calibration at `artifacts/hour5/demo/calibration_parameters.json`, seed 500005, SHA-256 `c69823decfe715ffbb8ec54885783881cef547bbbc1b59bb6a2de49bf114e476`; calibration was not regenerated.

All **38 tests passed before data collection**. They locked configuration parity, calibration identity, seed disjointness, 50 replicates, 40-placement baseline history and training exposure, six matched held-out evaluations, primary outcome and paired analyses, tutorial mapping, non-learning evaluation, exclusion of hidden policy weights, and byte identity of all committed Hour 3–6 artifacts. The single three-replicate smoke run passed integrity and measurement-resolution checks; its treatment direction was not inspected. The single confirmatory run then completed with identical matched starting states, equal budgets, matched evaluation seeds, unique seed records, and no primary floor/cap values. It was not rerun.

## Diagnoses and treatment differentiation

Primary diagnoses were hole management 20, height management 15, and surface management 15, compared descriptively with Hour 6's 23/13/14. Confidence was low/mixed 13, moderate 20, and high 17; 13 were mixed and 37 non-mixed, versus Hour 6's 23/11/16 and 23 mixed. The lower mixed count and higher moderate count are noteworthy natural sampling differences, not grounds for changing or excluding cases.

| Diagnosis | History-aware tutorial | Hour 7 n |
|---|---|---:|
| hole management | hole avoidance | 20 |
| height management | stack height | 15 |
| surface management | bumpiness | 15 |

All 50 diagnoses mapped to the intended tutorial. History-aware material differed from rating-only Elo-tier material in **50/50** replicates, so treatment differentiation did not fail.

## General performance and primary comparisons

The outcome is mean successful placements over six held-out challenges; improvement is post minus pre.

| Condition | Pre mean | Post mean | Mean improvement | Improvement 95% CI |
|---|---:|---:|---:|---:|
| control | 27.300 | 27.587 | 0.287 | [-0.145, 0.718] |
| rating-only | 27.300 | 27.223 | -0.077 | [-0.508, 0.355] |
| history-aware | 27.300 | 27.443 | 0.143 | [-0.360, 0.647] |

| Comparison | Hour 6 estimate [95% CI] | Hour 6 p / dz | Hour 7 estimate [95% CI] | Hour 7 p / dz |
|---|---:|---:|---:|---:|
| history - control | 0.487 [0.096, 0.878] | 0.0157 / 0.354 | -0.143 [-0.541, 0.254] | 0.4725 / -0.102 |
| history - rating-only | 0.407 [0.078, 0.736] | 0.0165 / 0.351 | 0.220 [-0.177, 0.617] | 0.2704 / 0.158 |

For history-control, signs disagree and the Hour 6 estimate is not in the Hour 7 CI. For history-rating-only, signs agree and the Hour 6 estimate is in the Hour 7 CI. Overall, Hour 7 **weakens and leaves ambiguous** the broad Hour 6 claim: it contradicts the point direction against control but remains compatible with a smaller positive advantage over rating-only.

## Target skills and exploratory transfer

The unchanged valid standardized aggregate was 0.121 frozen-calibration SD (95% CI -0.088 to 0.330). Every raw diagnosis-specific CI included zero.

| Diagnosed target | n | Pre | Post | Improvement mean [95% CI] |
|---|---:|---:|---:|---:|
| hole management / hole avoidance | 20 | -37.110 | -36.625 | 0.486 [-1.395, 2.366] |
| height management / stack height | 15 | -13.332 | -13.142 | 0.190 [-0.091, 0.471] |
| surface management / smoothness | 15 | -18.807 | -18.718 | 0.089 [-1.024, 1.202] |

Thus Hour 7 does not demonstrate that any intended target skill improved, despite all three point estimates being positive. Descriptively, standardized target/general improvements by confidence were low/mixed (n=13) 0.007/0.218, moderate (n=20) 0.229/0.008, and high (n=17) 0.079/0.245.

The unchanged exploratory within-history-aware transfer association was Pearson **r=0.511**, p=0.000148, and Spearman **rho=0.529**, p=0.0000783. These are stronger than Hour 6 (r=0.244, p=0.0875; rho=0.264, p=0.0644), but remain exploratory, do not establish causality, and do not override the uncertain aggregate target improvement.

## Secondary lines cleared

Lines per challenge pre/post/improvement were control 0.143/0.173/0.030, rating-only 0.143/0.143/0.000, and history-aware 0.143/0.143/0.000. The proportions of replicate means equal to zero pre/post were control 0.52/0.34, rating-only 0.52/0.46, and history-aware 0.52/0.48. Lines remain secondary and do not replace the primary outcome.

## Exploratory Hour 6+7 fixed-effect summary

This pooling was calculated only after Hour 7 was independently saved and interpreted. It is secondary and does not conceal disagreement.

| Comparison | Hour 6 | Hour 7 | Fixed-effect estimate [95% CI] |
|---|---:|---:|---:|
| history - control | 0.487 | -0.143 | 0.177 [-0.095, 0.449] |
| history - rating-only | 0.407 | 0.220 | 0.331 [0.083, 0.578] |

The inverse-variance summary does not resolve the cross-study directional disagreement against control. The rating-only summary is positive, but remains secondary to the independent replication.

## Conclusions and limits

Justified conclusions: Hour 7 preserved the locked design and treatment differentiation; it did not independently reproduce the history-control advantage; it produced a smaller, uncertain, directionally consistent history-rating advantage; and the two Hour 7 primary estimates differ in their compatibility with Hour 6. Diagnosis composition also shifted naturally, especially confidence mix. The exploratory target/general association was positive and stronger than in Hour 6.

Not justified: declaring complete replication from one agreeing sign; declaring universal failure from p-values above .05; claiming history-aware training beats control or rating-only generally; claiming the tutorials taught diagnosed skills; claiming causal transfer; promoting lines cleared; attributing the cross-hour difference to diagnosis distribution; generalizing beyond this simulation, duration, and evaluation; or redesigning the experiment in response.

Methodological concerns remain the six-challenge evaluation's uncertainty, noisy 40-placement histories, correlated diagnostic observables, relative calibration norms, target metrics measured on general rather than dedicated skill challenges, and only two independent experiments. The reversal against control is itself an important reproducibility concern. No scientific parameter was changed after any Hour 7 outcome, no prior artifact was overwritten, no Hour 8 work or Pewter integration was performed, and no commit was made.

## Reproduction

From repository root, after moving or removing an existing Hour 7 result directory so overwrite protection remains effective:

```bash
python -m unittest discover -s tests -v
python hour7_experiment.py --phase smoke --replicates 3 --output /private/tmp/hour7-smoke-700107
python hour7_experiment.py --phase confirmatory --replicates 50 --output artifacts/hour7/results
```
