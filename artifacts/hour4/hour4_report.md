# Hour 4 results

## Frozen design

The evaluation specification was frozen before either Hour 4 run. The primary general outcome
is mean successful placements before top-out across six ordinary held-out challenges (cap 120).
Lines cleared remains secondary. Each rating-history agent's target metric was selected by the
weakness diagnosed from its pre-existing observable placement history; the mapped metric was then
computed from matched, non-learning pre/post held-out play. No hidden weights enter diagnosis or
scoring. All conditions receive 40 ordinary learning placements.

The four small component scores are all oriented so higher is better:

- hole avoidance: negative trajectory-average holes;
- stack-height management: negative trajectory-average maximum column height;
- surface smoothness: negative trajectory-average adjacent-column bumpiness;
- line-clearing efficiency: trajectory lines cleared per successful placement.

Line clearing is a downstream outcome because row completion jointly depends on avoiding buried
holes, controlling height, maintaining usable surfaces, the piece stream, and placement choices.

## Smoke gate

Across the pooled 108 pre/post-condition challenge observations from three replicates, the primary
score had 14 distinct values, range 19–34, 0% zero, 0.926% at the observed minimum, and 0% at the
cap. No obvious floor or saturation remained, so the predeclared design proceeded unchanged.

## Fresh 50-replicate results

The procedure used master seed 400004. Within each replicate, a baseline agent first accumulated
40 placements of observable history. Three matched clones then received identical initial weights,
Elo, RNG state, and history. Pre-evaluation used the same six domain-separated challenge seeds for
all clones with learning disabled on disposable copies. Tutorial selection used observable history
only. Each condition then received 40 placements through the same ordinary learning update and
same training piece-stream seed. Post-evaluation repeated the matched held-out challenges with
learning disabled. Replicates had independently derived seed domains and fresh state.

Primary scores (mean placements/challenge; improvement CIs are t-based 95% CIs):

| Condition | Mean pre | Mean post | Mean improvement | SD improvement | 95% CI |
|---|---:|---:|---:|---:|---:|
| control | 27.1267 | 27.3300 | 0.2033 | 1.5948 | [-0.2499, 0.6566] |
| rating_only | 27.1267 | 27.2467 | 0.1200 | 1.6064 | [-0.3365, 0.5765] |
| rating_history | 27.1267 | 27.2700 | 0.1433 | 1.7626 | [-0.3576, 0.6442] |

Paired improvement comparisons:

| Comparison | Mean difference | 95% CI | t(49) | two-sided p | Cohen dz | Wilcoxon exact p |
|---|---:|---:|---:|---:|---:|---:|
| history - control | -0.0600 | [-0.5087, 0.3887] | -0.2687 | 0.7893 | -0.0380 | 0.8333 |
| history - rating_only | 0.0233 | [-0.3407, 0.3873] | 0.1288 | 0.8980 | 0.0182 | 0.8269 |

The exact signed-rank tests omitted zero paired differences and assigned average ranks to tied
absolute differences (39 nonzero/11 zero for history-control; 38 nonzero/12 zero for
history-rating). Parametric and nonparametric conclusions agree here.

All 50 rating-history agents were diagnosed with hole avoidance; no other tutorial was selected.
For that group the target score moved from -38.4315 before to -37.6976 after, a mean improvement
of 0.7339 (SD 3.4716, 95% CI [-0.2528, 1.7205]). The direction is favorable, but its interval spans
zero. Thus Hour 4 does not establish targeted learning. It also provides no evidence that the
personalized tutorial improved general transfer relative to either comparison condition. These
results do not prove equivalence or absence of small effects.

Secondary mean lines/challenge changed by 0.0033 in control (pre 0.1133, post 0.1167), 0.0233 in
rating_only (pre 0.1133, post 0.1367), and 0.0533 in rating_history (pre 0.1133, post 0.1667).
Across all 1,800 challenge observations, 89.44% still cleared zero lines, confirming the secondary
measure's severe sparsity. Hour 3 was neither rescored nor reinterpreted.

## Methodological findings and limits

The new primary was well resolved in the 50-replicate data: 21 distinct challenge-level values,
range 17–40, 0% zero, 0.056% at the observed minimum, and 0% at cap. It is markedly more sensitive
than line clears, though survival is still a proxy for general performance and this simplified
simulator omits many modern Tetris mechanics. More importantly, the existing diagnosis selected
only hole avoidance. Consequently, this run validates neither the other three tutorial types nor
the usefulness of the diagnosis for differentiating agents. The target analysis is mechanistic
and exploratory, and the single observed subgroup cannot support cross-skill causal claims.

Exact reproduction command:

```sh
python hour4_experiment.py --replicates 50 --master-seed 400004 --output artifacts/hour4/experiment_50_reproduction
```
