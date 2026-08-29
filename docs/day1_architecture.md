# Day 1 agent architecture

The committed Hours 1–9 implementation and artifacts are a frozen scientific baseline. The
four-weight `LearningAgent` is now a compatibility baseline and pedagogical placeholder, not the
target model of expert Tetris skill.

```text
Tetris rules / state
        ↓
StudentAgent final-placement interface
        ↓
student implementation (four-feature adapter today; search/imitation/RL later)
        ↓
curriculum / training
        ↓
held-out evaluation on disposable, non-learning clones
        ↓
independent behavioral validator
```

`Placement(rotation, x, use_hold=False)` is the research action. Final placements isolate the
quality of a Tetris decision from keyboard timing, autorepeat, controller latency, and path
planning. They also match the canonical transition engine, allow every policy to receive the same
legal choice set, and make preference rankings and regret comparable. Hold is reserved in the type
without changing today's no-hold rules.

`StudentAgent` requires a stable identifier/version, independent cloning, placement selection,
optional action scores, optional experience updates, serialization/save, a learning capability
flag, and deterministic selection where practical. It deliberately does not require weights.
The legacy adapter delegates stochastic selection and online learning to the unchanged
`LearningAgent.choose`; deterministic mode is an argmax over the exact historical feature score.

Tomorrow's engine enters through a separate seam:

```text
Tetris state
    ↓
ExpertPolicy.rank_placements(..., deterministic=True)
    ↓
ranked PlacementEvaluation values + search/confidence metadata
    ↓
ActionRegret
    ↓
diagnosis + training (future work)
```

No expert implementation, expert values, regret calculation, reinforcement-learning learner, or
taxonomy redesign is included on Day 1.

## Legacy coupling audit

The active `training.train` and Hour 4 held-out evaluator now accept `StudentAgent`; neither needs
weights or a particular feature count to choose placements. Legacy weight fields are emitted only
when the compatibility adapter is present, preserving the historical JSON schema. Hour 6's clone
fingerprint now uses serialized agent state rather than reaching into RNG or weight fields.

The following historical coupling remains intentionally:

- Hours 3–6 construct `LearningAgent` from their locked four-weight configurations. This is the
  baseline factory choice, not a restriction in training or evaluation.
- Hours 3 and 4 retain their original inline initial-state records (`weights`, RNG, history) because
  changing old experiment result schemas would impair reproduction.
- `hour9._trained_agents`, `_preferred`, and `behavioral_validate` remain the exact historical Hour
  9 reconstruction/audit and directly inspect four-feature weights. New validation uses
  `tetris_research.validator` and does not call that code.
- Diagnosis still consumes the committed placement-history schema and its four observable board
  metrics. Redesigning the taxonomy/history construct is explicitly outside Day 1.
- Seed derivation and matched stream layouts remain frozen in the Hour experiment modules. Agent
  RNG state is now opaque to generic training/evaluation and travels through serialization.

## Validator independence

Agent-bundle validation imports neither training nor curriculum, experiment, rating, or statistics
code. It independently compares serialized states, recomputes seed intersections and SHA-256
hashes, obtains deterministic and matched-RNG choices on fixed boards, and replays complete held-out
piece streams to compare actions, counts, lines, and final-board hashes. It shares the canonical
Tetris rules and the format-specific state loader; duplicating game physics would create a more
dangerous second rules implementation.

The committed Hour 9 artifacts predate the bundle schema and do not contain raw initial agents or
action-level evaluation replays. Their certificate therefore marks those two claims as legacy
attestations and returns `WARNING`, never an unsupported `PASS`. Budgets, seed separation,
performance means, and every committed artifact blob are recomputed independently.

Generate the preserved Hour 9 certificate with:

```bash
python -m tetris_research.validator artifacts/hour9_large_sample
```

Day 1 itself is architecture and validation work rather than a training experiment: it has no
exposure budget, evaluation cap, or outcome estimate. Run its focused tests with:

```bash
python -m unittest tests.test_day1_architecture -v
```

Relevant files are the [student interface](../tetris_research/student.py),
[legacy adapter](../tetris_research/legacy_student.py), [validator](../tetris_research/validator.py),
[Hour 9 certificate input](../artifacts/hour9_large_sample), and
[architecture tests](../tests/test_day1_architecture.py).
