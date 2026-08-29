# Day 3: direct expert-imitation baseline

Day 3 asks whether expert examples cause a weak student's own policy to improve after the expert is
removed. Parameter movement alone is not learning. The required causal chain is expert labels,
cross-entropy updates to student-owned weights, expert-free student choice, and improved held-out
behavior. Direct imitation is a baseline for later comparison, **not** the ultimate Adaptive Tetris
Tutor method.

## Student and learning rule

Two exact clones start from four zero weights. They make the same canonical tie-broken placement on
the same state. The student scores each legal afterstate with the simulator's normalized holes,
maximum-height, bumpiness, and lines-cleared features. For each training state it applies one
multiclass softmax cross-entropy SGD update with learning rate 0.35:

`w <- w + 0.35 * (features(label) - softmax_expected_features)`

The taught label is the Day 2 expert's preferred legal placement. The control label is a seeded
uniformly random legal placement. Both see the same 80 boards and pieces and receive exactly 80
labels and 80 updates. Expert Dellacherie features, values, coefficients, and search state never
enter the student representation or checkpoint. The learned taught weights were
`[3.0778736, 0.6695667, 0.5592200, 0.0]`; none is an expert coefficient.

## Locked separation and criterion

The final run was declared in code and printed before evaluation. Training seeds were 2026082401–
2026082404. Held-out decision seeds were 2026082501–2026082503; game seeds were 2026082601–
2026082610. Besides disjoint seed domains, held-out `(board, piece)` hashes were rejected if present
in training. The 24 decision states and ten uniform-piece game streams were matched across all
students; games were capped at 300 placements.

Evaluation calls students with `learn=False`, checks serialized state before and after, and students
contain no expert reference. A student placement is recorded before the external evaluator invokes
the expert to score agreement and regret. No rating is reported because the repository has no clean
independently calibrated rating for this new student.

Success required all of the following without post-result tuning: taught agreement at least 10
percentage points above both initial and control; taught regret at least 10% below both; and taught
placements or lines strictly above both in matched games. The independent validator also rejects
expert coefficients, unequal budgets, evaluation mutation, seed or state leakage, reversed expert/
student call order, and parameter-only success claims.

## Final held-out result

| Condition | Expert agreement | Mean expert regret | Mean placements | Mean lines |
| --- | ---: | ---: | ---: | ---: |
| Initial weak student | 25.0% | 108.753 | 25.3 | 0.1 |
| Direct expert imitation | 54.2% | 8.127 | 259.2 | 96.7 |
| Random-label control | 12.5% | 60.434 | 25.4 | 0.0 |

All predeclared checks passed. This is evidence of genuine transfer within this simplified
simulator: after training and expert removal, the student's own linear policy changed its decisions,
had much lower held-out expert-relative regret, and played substantially longer. It does not show
that direct imitation is optimal, that the Day 2 heuristic is ground truth, or that results transfer
to modern guideline Tetris. The corpus is small, only one fixed run and one weak learner are used,
the policy has four coarse afterstate features, and gameplay estimates have no confidence intervals.

Days 4–7 remain out of scope. In particular, Day 4 should add genuine reinforcement learning; Day 5
history-based error diagnosis; Day 6 targeted level generation; and Day 7 an equal-budget comparison
of ordinary practice, this direct-imitation baseline, and history-aware personalized tutorials.

## Reproduce and inspect

```bash
python -m tetris_research.day3 smoke --output /tmp/day3-smoke.json
python -m tetris_research.day3 final --output /tmp/day3-final.json
python -m tetris_research.day3_validator /tmp/day3-final.json
```

See the [preregistration](../experiments/day3/preregistration.json),
[final artifact](../experiments/day3/final_results.json),
[committed validation](../experiments/day3/validation.json),
[implementation](../tetris_research/day3.py), and [tests](../tests/test_day3_transfer.py).
