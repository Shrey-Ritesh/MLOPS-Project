# P4: Handwriting and learner personalization

This note records the current P4 implementation and the proposed handoff
contract. The phase plan remains the source of the team's acceptance criteria.

## Handwriting OCR

The app uses OCR.space Engine 3 through the adapter in `src/handwriting/`.
Images are validated as PNG/JPEG, limited to the free tier's documented 1 MB
file size, and checked for valid image content and dimensions before upload.
The learner must review and correct the returned text before it is marked
confirmed. OCR.space does not provide a reliable confidence value in the
response contract used here, so the UI does not claim a confidence score.

The repository contains extracted source/question image assets, but no
representative handwritten learner answers. OCR quality on subscripts,
charges, reaction arrows, and formulas is therefore **not yet measured**.
Before claiming feasibility, compare Engine 3 against a small, consented set
that covers those cases, record the raw OCR and corrected text, and report
character/symbol errors. Do not include learner identifiers in the examples.
Keep the correction UI as the safe path for misreads.

## Attempt contract proposal

Only persist an attempt after the shared evaluator returns an outcome. P4 does
not grade answers. The proposed minimal record is:

| Field | Type / allowed values | Purpose |
|---|---|---|
| `attempt_id` | opaque unique string | Deduplicate and address a record |
| `learner_id` | opaque pseudonymous string | Scope history to one learner |
| `question_id` | stable string | Identify the attempted question |
| `concept_ids` | non-empty list of stable strings | Aggregate performance by concept |
| `outcome` | `correct`, `incorrect`, `partial`, `uncertain` | Evaluator result; `uncertain` is not scored |
| `answer_mode` | `typed`, `handwritten` | Compare input paths without storing image bytes |
| `attempted_at` | timezone-aware timestamp | Apply the recent-history window |

Answer text, uploaded image bytes, names, and email addresses are not stored.
The evaluator owner should review this proposal before connecting writes.

## Initial deterministic recommendation rules

`src/learner/history.py` provides a SQLite store and a pure recommendation
function. Defaults are deliberately explicit and configurable:

- Use evaluated attempts from the most recent 30 days.
- Count `correct` as 1, `incorrect` as 0, and `partial` as 0.5.
- Exclude `uncertain` outcomes from accuracy and minimum evidence.
- Require at least 3 evaluated attempts for a weak-area recommendation.
- Treat accuracy below 70% as a practice priority; always return a reason and
  evidence count.
- Until the minimum evidence exists, recommend practice as a way to collect
  evidence, not as proof that the concept is weak.
- Among matching practice candidates, prefer the least recently attempted
  question, using question ID as a stable tie-breaker.

These are starting rules for team review, not a learned mastery model. The
store's database path defaults to `data/learner_history.sqlite3`, which is
excluded from Git. Use an opaque learner ID and keep reads scoped to that ID.
The current app has no account/authentication layer, so the store is not wired
into the UI until the team chooses how learner identity is provided.

## Integration boundary

When Person 3's evaluator contract is available, the integration should pass
the learner-confirmed transcription as `answer_text` with
`answer_mode="handwritten"`. After evaluation, map the evaluator's outcome,
question ID, concept IDs, learner ID, and timestamp to `AttemptRecord`, save it,
load that learner's recent records, and call `recommend_practice` with the
question candidates available to the app. Typed submissions should use the
same attempt store and recommendation path.
