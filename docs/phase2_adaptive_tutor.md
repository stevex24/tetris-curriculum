# Phase 2 — Responsive Adaptive Tetris Tutor

Phase 2 asks whether diagnosing a machine player's weaknesses and generating targeted practice improves learning efficiency. It compares four equal-budget training strategies for the artificial Tetris learner:

1. ordinary practice;
2. direct expert imitation;
3. static personalized training; and
4. responsive personalized training, which reassesses the full 18-feature profile and retargets practice in short blocks.

This is a controlled simulator study, not evidence about human learning.

## Frozen experiment

The locked run began every condition from the `rl@750` checkpoint and used eight paired independent replicates. Each condition received exactly 500 learner updates. Responsive training used ten 50-update blocks, with non-learning diagnosis between blocks and after the final block. All policies were evaluated on the same 12 held-out streams per replicate with a 500-placement cap.

The primary outcome is gain in a matched-stream simulator performance rating against fixed reference policies. The rating is a Bradley–Terry/Elo-style simulator scale anchored to the reference ladder; it is **not FIDE Elo and is not externally calibrated Elo**. Lines, placements, expert diagnostics, profile evolution, curriculum switches, and exposure allocation are secondary measures.

## Final results

The frozen artifact is [`experiments/phase2/final_results.json`](../experiments/phase2/final_results.json), with its locked design in [`preregistration.json`](../experiments/phase2/preregistration.json).

| Condition | Mean rating gain | Median rating gain |
| --- | ---: | ---: |
| Ordinary practice | +7.32 | +76.99 |
| Direct imitation | +371.54 | +394.16 |
| Static personalized | +109.60 | +106.29 |
| Responsive personalized | +7.26 | +36.00 |

Paired replicate contrasts (responsive and static/imitation comparisons use the same eight replicates):

| Contrast | Mean difference | 90% paired-bootstrap CI |
| --- | ---: | ---: |
| Responsive − ordinary | −0.06 | [−110.31, 103.34] |
| Responsive − static personalized | −102.34 | [−182.51, −22.30] |
| Responsive − imitation | −364.28 | [−437.04, −293.23] |
| Static personalized − ordinary | +102.28 | [+16.49, 198.82] |
| Imitation − ordinary | +364.23 | [+292.50, 448.75] |

Static personalized training substantially outperformed ordinary practice. Direct imitation was the strongest method for this artificial learner. The tested responsive algorithm did not outperform ordinary practice and performed worse than static personalization, so the preregistered result is **negative**.

This does not show that responsive personalization in general is ineffective. It tests one particular, computationally elaborate candidate-selection algorithm. A simpler fixed-interval loop—full-profile diagnosis → brief training → reassessment → retargeting—remains a reasonable future experiment without changing this frozen result.

The implementation uses artificial learners only. Direct machine imitation may suit this learner especially well and need not translate directly into human instruction. A later human study could compare rating or playing-strength improvement per equal training-time budget against ordinary practice and other forms of generated practice. Human benefit remains a hypothesis.

## Four-board replay demo

[`demo/phase2-comparison.html`](../demo/phase2-comparison.html) shows the four trained methods left to right as **Ordinary**, **Direct Imitation**, **Static Personalized**, and **Responsive Personalized**. It uses replicate 0 and held-out seed `2026101101`, selected by the predetermined rule “first frozen Phase 2 evaluation seed.” All boards consume the same piece stream, learning is disabled, and game-over boards freeze while others continue. Policies are reconstructed from frozen serialized states; no retraining, rating recomputation, cherry-picking, or new scientific evaluation is performed. Static rating captions are copied from the frozen artifact.

To open it locally:

```bash
cd ~/tetris-curriculum
python -m http.server 8000
```

Visit <http://localhost:8000/demo/phase2-comparison.html>.

The replay builder is [`demo/build_phase2_comparison.py`](../demo/build_phase2_comparison.py), and its generated data are [`demo/phase2_comparison.json`](../demo/phase2_comparison.json).

Earlier Hours 1–9 and Days 1–7 remain the historical foundation of this work; their documentation and frozen artifacts are unchanged.
