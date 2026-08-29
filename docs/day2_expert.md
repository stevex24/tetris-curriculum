# Day 2: strong reference placement policy

## Selection and provenance

Cold Clear 2 was investigated first because it is a credible modern open-source versus bot. Its
public README describes column-major bitboards, multithreaded search, a transposition-aware game
tree, and MCTS-inspired expansion; it is dual MIT/Apache-2.0 licensed. Direct integration was
rejected because this simulator has normalized hard-drop-only placements, no SRS kicks, hold,
preview, garbage, spins, combos, back-to-back scoring, or controller timing. Restricting Cold Clear
would change both its action generator and much of its objective, and its asynchronous TBP interface
does not expose a stable value for every canonical action.

The implemented reference is therefore a faithful simulator-native adaptation of the classic
Dellacherie placement evaluator, using its documented landing height, eroded piece cells, row
transitions, column transitions, holes, and cumulative-well feature definitions with the widely
reproduced six-feature coefficient set.
It adds a deterministic depth-two beam expectimax: for each of the strongest first placements it
averages the best second placement across the seven equiprobable next pieces. Aggregate height and
a near-ceiling penalty are explicitly project additions, not attributed to Dellacherie. The method
is a **strong reference policy**, not claimed to be state of the art.

Sources consulted: Cold Clear 2 (`MinusKelvin/cold-clear-2`, main branch as inspected 2026-08-23,
MIT or Apache-2.0); Cold Clear (`MinusKelvin/cold-clear`, MPL-2.0); and Thiery & Scherrer,
*Building Controllers for Tetris* (2009), which evaluates the Dellacherie feature controller.

## Exact shared game

The board is 10 by 20, with row zero at the floor. The seven tetrominoes and their unique normalized
rotations are defined in `tetris_research/tetris.py`. An action is `(rotation, x)` followed by a
vertical hard drop. There are no kicks or side moves beneath overhangs. Full rows clear
simultaneously. A placement intersecting the ceiling is unavailable; a game tops out when the
current piece has no legal placement. Pieces are independent uniform draws, not a seven-bag. Hold
and next preview are absent. Both policies receive the same canonical `ActionEvaluation` list and
only the current piece. The second search ply is consequently an expectation, not privileged
preview information.

## Policy contract and scores

`DellacherieSearchExpert.rank_placements` returns every legal placement, deterministic descending
heuristic value, one-based rank, static feature details, searched depth, beam width, and expected
continuation when searched. A score is heuristic utility: static rich-board utility plus 0.35 times
the second-ply estimate. It is neither probability nor confidence. `preferred_placement` is rank 1.

The legacy learner is only four normalized afterstate features (holes, maximum height, bumpiness,
lines cleared) and a linear argmax. The reference explicitly models landing geometry, destroyed
piece cells, boundary transitions, covered cells, deep wells, height, and likely next-piece best
replies.

`ActionRegret` is `value(best) - value(chosen)` in these utility units. It is zero for the reference
choice and nonnegative for ranked alternatives. It is regret relative to this approximate evaluator,
not mathematical ground truth or a promise about eventual survival.

## Predeclared evidence and limitations

The immutable design is in `experiments/day2/preregistration.json`. The timing-only fallback is
recorded separately. The fixed validation corpus covers empty/ordinary stacks, holes, wells, uneven
and high stacks, recovery, line clears, ambiguous surfaces, staircases, and covered cells. An
independent drop/replay implementation checks legality and transitions.

The matched benchmark is intentionally an oracle-strength test, not training or curriculum work.
Administrative censoring at 500 placements may compress a very strong policy's upper tail. Uniform
piece generation, no preview/hold/SRS, shallow beam approximation, and heuristic-relative regret
limit generalization to modern competitive Tetris. Day 3 may use rankings, values, metadata and
regret only if the saved certificate says the predeclared strength threshold passed.

## Locked benchmark result

The timing-only pilot triggered both registered fallbacks before outcomes were available: 50 games
and beam width 3. The completed run took approximately 65 minutes, exceeding the desired 30-minute
window; the configuration was not changed after launch. Master seed was `2026082302`.

Legacy survival was mean 250.82, median 234, range 133–500 placements. The reference reached the
500-placement administrative cap in all 50 games. Relative mean improvement was 99.3%, with paired
bootstrap 95% CI 82.4%–118.7%. This passes the predeclared requirement of at least 50% point
improvement and a lower confidence bound above 20%.

Secondary outcomes point in the same direction: mean lines were 83.76 versus 197.98; final holes
21.64 versus 0.22; final maximum height 19.44 versus 4.38; and final aggregate height 187.32 versus
20.42 (legacy versus reference). These endpoints are descriptive. Survival is right-censored, so
500 is a lower bound on reference mean uncapped survival and the benchmark cannot estimate its
upper tail.

The deterministic example rule yielded eight first disagreements and two agreement controls. Seven
of eight disagreements had positive legacy regret; one was an exact expert-value tie resolved by
canonical action order. No example was selected for visual attractiveness or favorable outcome.

The independent certificate passes all rule, legality, repeatability, replay, ranking, and regret
checks. Accordingly, Day 3 may use this policy as a simulator-specific placement oracle. It should
not describe it as a modern-guideline or globally state-of-the-art Tetris engine, and should address
runtime before generating a large imitation corpus.

## Reproduce and inspect

The committed runtime fallback has already selected 50 games and beam width 3. To avoid replacing
scientific artifacts during inspection, validation can read the committed benchmark directly:

```bash
python -m tetris_research.day2 validate
python -m unittest tests.test_day2_expert -v
```

The original benchmark command is `python -m tetris_research.day2 run`; it writes the canonical
Day 2 files in place and should only be used deliberately. See the
[preregistration](../experiments/day2/preregistration.json),
[runtime decision](../experiments/day2/runtime_decision.json),
[benchmark](../experiments/day2/benchmark.json), [certificate](../experiments/day2/certificate.txt),
[implementation](../tetris_research/day2.py), [expert policy](../tetris_research/expert.py), and
[tests](../tests/test_day2_expert.py).
