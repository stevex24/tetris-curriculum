# Day 5: behavioral weakness diagnosis from playing history

Day 5 adds a narrow `history -> weakness profile` mechanism for the 18-feature Day 4 student. It
does not inspect student parameters and does not generate tutorial boards. The student first makes
a deterministic, nonlearning placement; history collection has no oracle dependency. Diagnosis
later reconstructs the legal action from the recorded board and piece, then invokes the independently
validated Day 2 `DellacherieSearchExpert` with beam width 1.

## History and independent replay

Each game records its ID, seed, learning flag, placements/lines/terminal outcome, and final board.
Each step records game ID, contiguous step index, all 20 pre-action row bitmasks, piece, selected
`[rotation, x]`, `learning_enabled=false`, the selected action's 18-feature vector, all 20 afterstate
row bitmasks, and rows cleared. Diagnosis rejects an illegal action, a non-reconstructable afterstate,
or a feature vector that differs from independent recomputation. Unknown fields are ignored. Agent
IDs, weights, policy preferences, RNG state, training returns, and training labels are not inputs.

## Fixed feature families

- **hole management:** holes, hole depth, rows with holes, covered columns
- **stack height / danger:** aggregate height, maximum height, high-stack cells, landing height
- **surface smoothness:** height standard deviation, bumpiness, cliffs, surface blocks
- **well management:** cumulative wells, deep wells
- **transition / structure quality:** row transitions, column transitions
- **line-clear / recovery efficiency:** completed lines, eroded piece cells

The mapping covers each richer feature exactly once. Lower is treated as preferable for every feature
except completed lines and eroded piece cells, for which higher is preferable. This directional rule
is only an attribution aid; expert regret, not a feature difference, determines whether a decision is
an error.

## Attribution and score

For each recorded decision, the oracle ranks all reconstructed legal actions. Regret is oracle-best
value minus recorded-action value. Decisions below regret 1.0 contribute nothing. On a remaining
decision, direction-adjusted recorded-minus-best feature differences at or below 0.01 in the already
scaled Day 4 feature space are discarded. Positive differences are averaged within each family.
Each family's average is divided by the sum of family averages, allowing several families to share
the decision's regret.

For family `f`:

`score(f) = attributed_regret(f) / all_recorded_decisions * games_with_attribution(f) / all_games`

Thus the score combines severity, frequency, and cross-game consistency. Dividing by all decisions
prevents a small number of mistakes from masquerading as a frequent tendency; the game-coverage
factor further suppresses an isolated blunder. The score is transparent but not causal: correlated
features and temporary tactical deterioration remain possible.

## Predeclared validation

Six matched piece-stream seeds and at most 80 placements per game were used for every profile. The
two synthetic richer-student policies share fixed magnitude-40 penalties for undesirable features
and magnitude-40 rewards for line features. The hole-weak policy sets only hole-family coefficients
to zero; the well-weak policy sets only well-family coefficients to zero. Parameters construct the
policies but are never supplied to diagnosis. A first magnitude-8 two-game development smoke failed
to expose well weakness; the uniform magnitude was revised once before the final run and recorded in
the preregistration. No final-result tuning occurred. Budgets were matched, but exact playing strength
was not.

The preregistered criterion required each intended family in the top two, profile L1 distance at
least 0.01, invariance to irrelevant hidden-weight fields, and no student mutation. It passed. Hole
management ranked first for hole-weak (score 12.0067). Well management ranked second for well-weak
(1.5068), behind hole management (4.2499); this is a qualified recovery, not a clean top-one result.
Profile L1 distance was 10.4501. Both profiles attributed their intended family in all six games.

The frozen natural Day 4 `rl@3000` history produced this inferred ranking: surface smoothness
(76.5813), well management (10.2360), stack height/danger (3.8613), hole management (2.1205),
transition/structure (1.6501), and line-clear/recovery (0.0178). These are hypotheses for Day 6, not
known ground truth.

The independent validator replays every action/afterstate, recomputes hidden-field invariance, checks
the complete family mapping, verifies recorded-decision-before-oracle event order, confirms the
near-zero and cross-game rules, and audits non-mutation. All checks passed.

## Limitations

This is a simplified hard-drop simulator, a single oracle and feature representation, six short
matched histories, two hand-constructed adversarial policies, and one naturally learned policy.
The synthetic students are not exactly strength-matched. Feature-family evidence is correlated,
oracle preferences are not infallible, and the natural profile has no ground-truth label. The large
surface-smoothness score for the natural student warrants inspection rather than a causal claim.

**Day 5 diagnoses recurring behavioral weaknesses; it does not yet prove that training those
weaknesses improves rating.**

## Reproduce and inspect

```bash
python -m tetris_research.day5 smoke
python -m tetris_research.day5 run
python -m tetris_research.day5_validator experiments/day5/final_results.json
```

The `run` command writes the canonical result path, so use it deliberately. See the
[preregistration](../experiments/day5/preregistration.json),
[final artifact](../experiments/day5/final_results.json),
[committed validation](../experiments/day5/validation.json),
[implementation](../tetris_research/day5.py), [diagnosis logic](../tetris_research/diagnosis.py),
and [tests](../tests/test_day5_diagnosis.py).
