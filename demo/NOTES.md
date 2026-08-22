# Demo provenance and presentation notes

Selection rule (fixed before replay inspection): **the first high-confidence history-aware learner in saved Hour 7 replicate order**. This selects Hour 7 replicate 0. No alternative learner or challenge was visually screened. The first saved held-out challenge (index 0, seed 1370957669) is used before and after.

The baseline is cloned into the actual three Hour 7 conditions: control (ordinary practice), rating-only (generic rating-selected tutorial without history), and rating+history (the calibrated-profile-selected stack-height tutorial). All three post-training agents replay challenge seed 1370957669 in synchronized panels. This is **one matched replicate — illustrative, not the aggregate statistical result**. The baseline makes 28 successful placements; each condition makes 26 on this particular replay. A tie here is neither evidence of equivalence nor of general treatment superiority. The aggregate screen is the inferential evidence.

The static replay data are deterministically reconstructed from saved Hour 7 seeds and the committed simulator/agent/training implementation. `build_demo_data.py` reconstructs only this selected learner and its three exact condition clones; it does not run an experiment batch. The tutorial is the learner model's selected stack-height tutorial; no claim is made that its intended teaching mechanism has been demonstrated.

Zoom sequence: **Profile → Before → Conditions → Tutorial → Three-way After → Results → Future**. Start the Before replay, advance; explain the clone diagram; briefly start the tutorial, advance; use Start/Pause/Reset/Next step on the synchronized three-way replay; then advance through aggregate results and future work. Target duration: 3–5 minutes.
