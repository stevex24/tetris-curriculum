# Day 6: targeted tutorial situations

Day 6 converts a frozen Day 5 weakness profile into replayable practice situations. A situation is 20 legal 10-bit board-row masks plus a current tetromino. It contains no preferred action, expert value, or reward.

The generator starts with naturally observed rl@3000 Day 5 states. It retains plausible states (4–120 occupied cells, height at most 16) and adds a deterministic one-legal-placement neighbor with a seeded next piece. For surface smoothness, well management, and hole management, it ranks candidates by action sensitivity: across legal placements, compute each unchanged Day 4 feature's range and average ranges within the target family. The top eight unique board/piece pairs are retained, with at least four historical origins. Family overlap is expected; the metric establishes emphasis, not exclusivity.

Personalization restricts the Day 5 ranking to supported families, takes the top two, normalizes their nonnegative diagnosis scores, and uses deterministic largest remainder to allocate 24 exposures. A profile-blind generic control shuffles and samples the same plausible candidate pool without inspecting the profile.

Each exposure resets to one situation and performs exactly the standard Day 4 calls: stochastic `choose_placement(learn=True)`, `update` with `gameplay_reward` (0.02 plus simulator-cleared rows), then `finish_episode(learned=True)`. Thus one exposure is one placement, update, episode boundary, and reset. There is no expert call, feature bonus, reward shaping, parameter setting, or search inside the learner.

Legality is checked by row-mask validation and replay through `TetrisAdapter`; every retained situation has at least two legal placements. Diversity uses unique board/piece pairs and distinct historical origins. Fixed seeds make both selection and exact equal-budget replay reproducible.

The one-step reset makes immediate placement consequences learnable within the current linear policy, but it does not teach explicit deep planning. Sequentially dependent well choices remain limited by the learner's lack of oracle-style lookahead.

**Day 6 validates targeted tutorial generation and training plumbing. It does not yet establish that personalized tutorials improve rating faster.** That comparison belongs to Day 7.
