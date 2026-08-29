# Day 6: targeted tutorial situations

Day 6 asked whether a frozen Day 5 weakness profile could select legal, replayable practice situations that emphasize its leading feature families while preserving the Day 4 learner's ordinary update path. It validated generation and training plumbing; it did not test held-out improvement.

## Starting point and generation

The input was the committed natural `rl@3000` Day 5 history and weakness profile. A situation contains 20 legal 10-bit board-row masks plus the current tetromino. It contains no preferred action, expert value, stored reward, student parameter, or hidden diagnosis label.

The generator retained plausible observed states (4–120 occupied cells, maximum height 16), then added one deterministic legal-placement neighbor with a seeded next piece. For each supported family—surface smoothness, well management, and hole management—it measured action sensitivity as the average range of that family's unchanged Day 4 features over all legal placements. It retained the top eight unique board/piece pairs per family, requiring at least four distinct historical origins. Overlap among families was allowed; sensitivity establishes emphasis, not exclusivity.

The natural profile's top two supported families were surface smoothness and well management. Their nonnegative scores were normalized and allocated the 24-exposure budget by deterministic largest remainder: 21 surface and 3 well exposures. A contrasting synthetic ranking allocated 22 hole and 2 well exposures, confirming that profile changes alter selection. A profile-blind generic control sampled the same plausible candidate pool without reading the profile.

## Exposure and validation

**One exposure is one tetromino placement and learning opportunity, not a full game.** Every exposure resets to one situation and calls the standard Day 4 sequence: stochastic `choose_placement(learn=True)`, `update` with only `0.02 + rows cleared`, then `finish_episode(learned=True)`. Targeted and generic smoke learners each received exactly 24 placements, updates, and resets. There was no expert call, preferred label, feature bonus, reward shaping, coefficient copying, or parameter edit.

All three targeted sets exceeded the preregistered 1.25 targeting-to-generic sensitivity ratio, every situation had at least two legal placements, selection differed for different profiles, diversity and history immutability checks passed, both learners changed weights through the same standard agent version/path, and the independent validator passed. Generation seed was `2026083101`. There were no games, held-out evaluation cap, replicates, confidence intervals, or effectiveness statistics in this milestone.

The one-step reset makes immediate consequences learnable by the linear policy, but does not teach explicit deep planning. Sequential well decisions remain limited by the student's lack of search. Day 7 alone tests whether these static targeted situations improve independent play.

## Reproduce and inspect

```bash
python -m tetris_research.day6 smoke
python -m tetris_research.day6 run
python -m tetris_research.day6_validator experiments/day6/final_results.json
```

The `run` command writes the canonical artifact, so use it deliberately. See the
[preregistration](../experiments/day6/preregistration.json),
[final artifact](../experiments/day6/final_results.json),
[committed validation](../experiments/day6/validation.json),
[implementation](../tetris_research/day6.py), [tutorial generator](../tetris_research/tutorials.py),
and [tests](../tests/test_day6_tutorials.py).
