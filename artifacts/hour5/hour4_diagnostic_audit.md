# Hour 5: audit of the frozen Hour 4 diagnosis

This audit was completed before changing the diagnosis implementation. Its source is the 50
saved `diagnostic_evidence` objects in
`artifacts/hour4/experiment_50/target_skill_results.csv`, produced at commit `bce5132`.
The Hour 4 code takes the last 40 successful placements. It computes four dimensionless numbers,
calls larger values weaker, and selects their uncalibrated maximum.

## Exact formulas and observed distributions

Quartiles below are Python `statistics.quantiles(..., n=4)` exclusive quartiles. SD is sample SD.

| Dimension | Raw observable | Hour 4 severity transformation and reference | Raw distribution (min, Q1, median, Q3, max; mean, SD) | Severity distribution (min, Q1, median, Q3, max; mean, SD) |
|---|---|---|---|---|
| hole avoidance | mean holes after each placement | mean holes / 20; 20 was an uncalibrated prototype constant | 22.675, 28.844, 31.538, 33.625, 43.825; 31.7015, 4.2238 | 1.1338, 1.4422, 1.5769, 1.6813, 2.1913; 1.5851, 0.2112 |
| stack-height management | mean maximum column height after each placement | mean height / board height (20) | 10.275, 11.169, 12.000, 12.463, 13.350; 11.8775, 0.7812 | 0.5138, 0.5584, 0.6000, 0.6231, 0.6675; 0.5939, 0.0391 |
| surface smoothness / bumpiness | mean sum of absolute adjacent-column height differences after each placement | mean bumpiness / 40; 40 was an uncalibrated prototype constant | 12.650, 16.781, 17.788, 19.838, 25.200; 18.2065, 2.4432 | 0.3163, 0.4195, 0.4447, 0.4959, 0.6300; 0.4552, 0.0611 |
| line-clearing efficiency | total lines cleared in 40 placements (also rate per placement) | max(0, 1 - lines / (40/10)); equivalently max(0, 1 - 10 x line rate). The reference target is one line per 10 placements. | total lines: 0 for 46 histories, 1 for 4; mean 0.08, SD 0.2740 (rates 0 or 0.025) | 0.75 for 4 histories, 1.0 for 46; mean 0.98, SD 0.0685 |

Distinct observed values were 50, 41, 47, and 2 respectively. The pairwise Pearson correlations
among the severity values were:

| | holes | height | bumpiness | line efficiency |
|---|---:|---:|---:|---:|
| holes | 1.000 | 0.436 | -0.133 | -0.241 |
| height | 0.436 | 1.000 | 0.430 | -0.137 |
| bumpiness | -0.133 | 0.430 | 1.000 | -0.136 |
| line efficiency | -0.241 | -0.137 | -0.136 | 1.000 |

## Why the result was hole avoidance 50/50

The severity scales were not empirically comparable. In every history, hole severity exceeded 1.13,
while height never exceeded 0.668 and bumpiness never exceeded 0.630. Line severity was 1.0 for 46
histories and 0.75 for four, yet even the smallest hole severity (1.1338) exceeded it. The hole score's
lead over the next-highest score ranged from 0.1338 to 1.4413 (median 0.5919). Therefore `max(...)`
necessarily selected holes in all 50 cases.

This is primarily a reference-scale failure: 20 is inappropriate as a normalization denominator
for cumulative board holes in these trajectories, and none of the constants was calibrated to a
common population interpretation. Line efficiency is additionally saturated and sparse: 92% of
histories have exactly the same maximum weakness score. The moderate holes-height and
height-bumpiness correlations show shared board-state structure, but correlation does not explain
the deterministic scale separation. The continuous measures do vary, so the histories are not
literally identical; whether that variation represents distinct learner types rather than ordinary
seed variation requires a separate diagnostic-population analysis. Hour 4 cannot answer that.

Consequently, the 50/50 result is not evidence that all agents genuinely share hole avoidance as
their dominant weakness. Nor is it evidence for any alternative diagnosis. It is evidence that the
hand denominators made direct cross-dimension ranking invalid, compounded by an under-resolved
line metric and a homogeneous agent-generating process.
