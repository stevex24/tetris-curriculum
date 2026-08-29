# Day 7: equal-budget tutoring comparison

Day 7 asks whether history-aware personalized tutorial training improves independent ordinary-game
play more than ordinary practice or direct expert imitation. This is a **machine-student result** in
the repository's simplified hard-drop Tetris simulator. It is not evidence about human learning or
human tutoring effectiveness.

## Locked design

The common student is the exact `rl@750` serialized state in the committed Day 4 final artifact,
with its 18 weights, learning settings, counters, and return baseline. It was chosen before the final
run because it was already competent (149.58 placements and 43.67 lines at the Day 4 cap of 300),
but weaker than `rl@3000` (192.17 and 62.08). Each of eight replicates reproducibly reseeded that
checkpoint's policy RNG, then cloned the resulting state into all three conditions. Within a
replicate, all trainable parameters, learning state, and RNG state were identical at the start.

History used Option A. Before condition cloning, the common frozen policy played six deterministic
80-placement games. Collection made no updates and contained no oracle. Day 5 then invoked the
external oracle only after each recorded decision. Its ranking began with hole management and
surface smoothness; Day 6 allocated the 500 tutorial states 270/230 between those families. The
same history/profile/tutorial set was used for every replicate, so history observation was not free
personalized learning.

Each condition received exactly 500 decision states and exactly 500 parameter updates:

- **Ordinary practice:** 500 sequential ordinary-game placements, stochastic policy choices, Day 4
  environment reward (`0.02 + cleared rows`), and unchanged episodic REINFORCE. It consumed 500
  ordinary states and reset 18–20 times depending on the replicate.
- **Direct imitation:** 500 policy-independent ordinary decision states, 500 beam-width-1 expert
  preferred-placement labels, and 500 student-owned 18-feature action-softmax cross-entropy updates
  at the locked rate 0.08. It copied no expert coefficient, value, feature, or search state.
- **Personalized tutorials:** 500 Day 6 selected situations, one placement/reset apiece, with the
  unchanged Day 4 choose/update/finish path and environment reward. It received no expert label,
  oracle call, family bonus, reward shaping, or hidden parameter edit during learning.

Thus updates and learner decision opportunities are matched, not CPU time or resets. The conditions
intentionally differ in data and objective: sequential RL, supervised labels, and targeted-state RL.

Training seeds were 2026090101–2026090108. Shared history seeds were 2026090001–2026090006;
tutorial generation used 2026090050. Evaluation game seeds were 2026091101–2026091112 and diagnostic
state seeds were 2026091201–2026091202. These domains are disjoint. Every frozen policy was evaluated
on the same 12 ordinary-game streams at cap 500; serialized state was identical before and after.

The primary endpoint was each replicate's change from its matched frozen baseline in mean held-out
lines. The paired analysis resampled eight replicate units 10,000 times and reports percentile 90%
bootstrap intervals. Minimum success required personalized minus ordinary to have mean at least 2.0
lines and CI lower bound above zero. Strong success additionally required the same against imitation.

## Final result

| Condition | Mean post lines | Mean improvement | Median improvement | Mean post placements |
| --- | ---: | ---: | ---: | ---: |
| Ordinary practice | 53.94 | +15.60 | +16.63 | 174.48 |
| Direct imitation | 158.51 | +120.18 | +122.33 | 424.03 |
| Personalized tutorials | 73.40 | +35.06 | +35.21 | 225.65 |

The common frozen baseline was 38.33 lines and 135.75 placements. Paired primary contrasts were:

| Contrast | Mean | Median | Paired-bootstrap 90% CI |
| --- | ---: | ---: | ---: |
| Personalized − ordinary | +19.46 | +19.63 | [8.77, 30.44] |
| Personalized − imitation | −85.11 | −87.75 | [−91.45, −77.69] |
| Imitation − ordinary | +104.57 | +114.29 | [88.24, 119.82] |

Personalized exceeded ordinary in six of eight paired replicates. The locked classification is
**minimum success**, not strong success: personalized training showed a nontrivial independent
gameplay advantage over ordinary practice, while direct imitation was substantially stronger.

Secondary post-training oracle diagnostics were `(agreement, mean regret)`: baseline
`(18.75%, 19.778)`, ordinary `(21.88%, 20.428)`, imitation `(39.84%, 9.212)`, and personalized
`(25.00%, 13.278)`. These explain policy behavior but did not determine success. Mean placement
improvements were +38.73 ordinary, +288.28 imitation, and +89.90 personalized.

There were 0 baseline, 3 ordinary, 54 imitation, and 1 personalized cap hits among 96 evaluations
per condition. The large imitation cap rate censors its true upper-tail strength, but cannot create
the personalized-over-ordinary finding because both had little censoring and shared the same cap.
No Elo-like rating is reported because there is no independently calibrated richer-student ladder.

## Validation and limitations

The independent validator passed all 18 checks, including identical starts, exact update/exposure
counts, nonlearning shared history, no expert coefficients, no expert or shaped reward in tutorial
learning, common RL code/reward for ordinary and personalized, post-decision oracle use, evaluation
nonmutation, seed separation, matched evaluation streams/cap, locked configuration, hidden-weight
exclusion, and paired replicate identity.

The experiment has only eight training replicates, one checkpoint, one fixed shared history/profile,
one tutorial round, a linear policy, and a simplified simulator. Replicates vary training randomness
but not evaluation streams or initial weights. Matching one update per opportunity does not make a
one-step tutorial reset equivalent to a sequential RL trajectory, and imitation uses a distinct
supervised objective as intended. The imitation endpoint is heavily ceiling-censored. Weakness
profile change was not measured, because the optional explanatory analysis would add substantial
oracle runtime and was not required for the locked primary question.

The personalized curriculum was static for all 500 exposures. A future system could instead run
`diagnose → short targeted practice → reassess → retarget`, but Day 7 neither implemented nor
tested that adaptive loop. It also does not map 500 machine placements to a human training duration
or establish that feature families are causally independent.

## Reproduce and inspect

```bash
python -m tetris_research.day7 smoke --output /tmp/day7-smoke.json
python -m tetris_research.day7 final --output /tmp/day7-final.json
```

The independent validator currently reads and rewrites the canonical artifact paths; its exact
command is `python -m tetris_research.day7_validator` and should therefore be run deliberately.
See the [preregistration](../experiments/day7/preregistration.json),
[final artifact](../experiments/day7/final_results.json),
[committed validation](../experiments/day7/validation.json),
[implementation](../tetris_research/day7.py), and [tests](../tests/test_day7_comparison.py).
