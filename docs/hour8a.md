# Hour 8A: animated adaptive-curriculum demo

Hour 8A turned one committed Hour 7 replicate into a local browser presentation. It introduced no new experiment and made no aggregate effectiveness claim.

## Selection and replay

The selection rule was fixed before replay inspection: use the first high-confidence history-aware learner in saved Hour 7 replicate order. That selected replicate 0, whose calibrated primary diagnosis was height management and whose assigned material was the stack-height tutorial. The presentation uses the first saved held-out challenge, seed `1370957669`.

The baseline and three actual Hour 7 condition clones are shown: ordinary control, rating-only material selected without history, and rating+history material selected from the calibrated profile. Training used the original 40-placement budgets. Replays are deterministic reconstructions of committed seeds and code, and the after panels use the same piece stream with learning disabled.

The selected baseline survived 28 placements. Control, rating-only, and rating+history each survived 26 placements on this one replay. Those are placements, not games. The tie and decline are intentionally retained: the animation is illustrative and neither overrides nor estimates the aggregate Hour 7 effect.

## Integrity and use

Tests verify the prespecified learner mapping, common baseline clone, matched challenge seed, exact saved placement and line counts, scientific condition labels, and displayed Hour 6/7 aggregate statistics. The generated JSON contains board frames rather than running an experiment in the browser.

```bash
python demo/build_demo_data.py
python -m unittest tests.test_hour8a_demo -v
python hour8a_demo.py
```

Open <http://127.0.0.1:8765/> after starting the server. See the [demo](../demo/index.html), [presentation notes](../demo/NOTES.md), [committed replay data](../demo/demo_data.json), [builder](../demo/build_demo_data.py), [server](../hour8a_demo.py), and [tests](../tests/test_hour8a_demo.py).

The separate [Day 4 four-board demo](../demo/day4_rl_oracle_comparison.html) was added during Stage II and compares richer RL checkpoints with the validated oracle.
