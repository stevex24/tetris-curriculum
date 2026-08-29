# Day 4: richer interpretable student and sequential RL

Day 4 replaces the four-feature Day 3 proof-of-mechanism student for future research with an
18-parameter linear softmax policy. It remains a **machine-student experimental model**, not a
model of human cognition or human learning speed. Its parameters support separable weaknesses while
remaining inspectable. It has no expert reference, search, preview, hold, or hidden state.

## Features

All features describe the candidate afterstate except landing geometry and eroded cells. Scaling is
fixed for numerical stability; it does not encode expert coefficients.

- `holes`: empty cells below a column's top, /40; `hole_depth`: total distance from each hole to its
  column top, /200; `rows_with_holes`: distinct rows containing holes, /20.
- `aggregate_height`: sum of column heights, /200; `maximum_height`: tallest column, /20;
  `height_stddev`: population SD of ten heights, /20.
- `bumpiness`: adjacent absolute height differences, /200; `cliffs`: fraction of adjacent pairs
  differing by at least three rows.
- `landing_height`: bottom y plus the placed piece's mean cell y, /20.
- `completed_lines`: rows cleared by the move, /4; `eroded_piece_cells`: cleared-row count times the
  number of placed cells in cleared rows, /16.
- `row_transitions`: occupied/empty transitions with occupied side walls, /400;
  `column_transitions`: transitions with an occupied floor, /400.
- `cumulative_wells`: sum of 1+...+depth for empty cells bounded on both sides, /200;
  `deep_wells`: number of such cells at depth at least three, /200.
- `covered_columns`: fraction of columns containing at least one hole; `high_stack_cells`: column
  height above row 14 summed across columns, /60; `surface_blocks`: number of nonempty rows, /20.

On the fixed diagnostic board `(28, 60, 62, 30)` from the floor upward with a T piece, isolated
penalties for holes, maximum height, and cumulative wells choose three different legal placements.
Tests also check unique names and that at least 15 of 18 feature response vectors differ on that
board; related height measures intentionally correlate but have different definitions.

## Reinforcement learning

Training uses episodic REINFORCE. The student samples from its own softmax placement policy. After
each action it receives only `0.02 + rows_cleared`. At top-out the trajectory ends; short games
therefore contain fewer future survival rewards. For every step, `G_t = r_t + 0.99 G_(t+1)`.
The update is `w += 0.008 * clip(G_t - b, -10, 10) * grad(log pi)`, where `b` is the mean return
from completed earlier trajectories. Thus line clears and continued survival after an action affect
that earlier action. A behavioral test changes only a later reward and verifies that the earlier
policy update changes.

The expert supplies no action, value, regret, coefficient, feature, or reward during training.
`train_exposure` does not accept an expert and records zero expert calls. An identical ordinary-play
control consumes the same indexed 3,000-piece stream but performs no updates.

## Evaluation and limitations

The locked checkpoints are 0, 250, 750, 1,500, and 3,000 placements. Learning is disabled during
12 deterministic held-out games at each checkpoint; serialized state is compared before and after.
Placements and lines are primary. On 16 separate states the expert is called only after all frozen
student choices, producing secondary agreement/regret diagnostics. Training and both evaluation
seed domains are disjoint. The preregistration is `experiments/day4/preregistration.json`.

No Elo-like number is reported: this repository has no independently calibrated ladder for the new
policy, and inventing one would be misleading. Results concern a simplified hard-drop simulator,
one training run, one initial policy, and a small held-out sample. They do not establish human-rate
learning, modern-guideline Tetris skill, weakness diagnosis, or personalized tutorial efficacy.

## Locked final result

The one preregistered run passed every success check. Mean `(placements, lines)` by exposure was:
0 `(24.67, 0.08)`, 250 `(30.42, 0.00)`, 750 `(149.58, 43.67)`, 1,500
`(143.67, 42.17)`, and 3,000 `(192.17, 62.08)`. The ordinary-play no-update control was
`(24.67, 0.08)`. The non-monotonic intermediate curve is reported without smoothing or tuning.

Secondary expert agreement changed from 0.0% to 43.75%; mean expert-relative regret changed from
92.834 to 16.361. These diagnostics did not determine success and the expert was first invoked only
after frozen student decisions. The independent validator passed all structural and behavioral
artifact checks.

## Reproduce and inspect

```bash
python -m tetris_research.day4 smoke --output /tmp/day4-smoke.json
python -m tetris_research.day4 final --output /tmp/day4-final.json
python -m tetris_research.day4_validator /tmp/day4-final.json
```

See the [preregistration](../experiments/day4/preregistration.json),
[final artifact](../experiments/day4/final_results.json),
[committed validation](../experiments/day4/validation.json),
[implementation](../tetris_research/day4.py), [richer student](../tetris_research/richer_student.py),
and [tests](../tests/test_day4_rl.py).
