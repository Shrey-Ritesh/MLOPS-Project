# Phase 2: One-Week Work Split, Roles, and Daily Milestones

**Project:** Chemistry-focused AI learning assistant for JEE preparation  
**Planning horizon:** 7 working days  
**Purpose:** Give each of the four team members an unambiguous owner, daily deliverables, acceptance criteria, dependencies, and integration checkpoints.  
**Release target:** A tested, reproducible **release candidate** and a documented path to production—not a public production deployment.

> **Planning rule:** Treat this as a working agreement, not evidence that work is already complete. Verify the repository, current schemas, retrieval index, and MLflow runs before changing or claiming anything. Adapt proposed interface examples to the actual codebase rather than replacing existing contracts without review.

---

## 1. Shared product outcome

The app should behave like a Chemistry learning assistant, not merely a question-fetching interface:

1. Present a JEE Chemistry question with its answer key and concept metadata available to the backend.
2. Accept a typed answer or a handwritten solution image.
3. Convert handwritten input to editable text; let the learner correct uncertain transcription before evaluation.
4. Evaluate the answer using question-specific answer criteria and relevant NCERT material.
5. Explain mistakes with evidence grounded in retrieved Chemistry material, and state uncertainty when evidence is insufficient.
6. Record attempts and use concept-level performance to recommend useful next practice.
7. Compare the baseline embedding model with a candidate on the same versioned retrieval benchmark; log evidence and the model/index pairing.

### Explicit scope boundaries

- **In scope:** one subject (Chemistry), typed-answer path, handwritten-input path behind a replaceable recognition interface, grounded feedback, basic learner history, transparent rule-based recommendations, retrieval benchmark, MLflow tracking, CI, Docker smoke tests, release documentation.
- **Not a goal for this week:** training an LLM from scratch, a multi-agent architecture, a learned recommendation model, full public deployment, or a large rewrite of working pipelines.
- **Important separation:** answer evaluation and retrieval quality are different problems. A better embedding model does not automatically make answer grading correct.

## 2. Roles and ownership

Each person is the **single directly responsible owner** for the deliverables below. Other members review and integrate, but ownership should not be ambiguous.

| Member | Role | Owns | Does not own | Required end-of-week evidence |
|---|---|---|---|---|
| **Person 1** | DevOps & release engineering | Branch/PR workflow, GitHub Actions CI, Docker build and local smoke test, configuration validation, health/readiness checks, release and rollback instructions, deployment-readiness checklist | Cloud deployment, cloud credentials, feature-specific grading logic, embedding-model selection | Green PR checks where permissions allow; reproducible Docker smoke test; documented environment/configuration and known deployment blockers |
| **Person 2** | Embedding model & ML evaluation | Retrieval benchmark, query/relevance labels, baseline and candidate evaluation, Recall@5/MRR and latency, MLflow evidence, model/index compatibility manifest, promotion/rejection recommendation | App endpoint implementation, answer grading, changing production retrieval behavior without agreement | Reproducible baseline report; candidate comparison if feasible; MLflow runs/artifacts; explicit promotion decision and reproducible index-build instructions |
| **Person 3** | Backend & answer evaluation | Question/answer evaluation service, typed-answer API flow, NCERT-grounded feedback integration, attempt persistence contract, backend tests, retrieval/evaluation interface | OCR model internals, CI ownership, choosing embedding model solely on intuition | Tested typed-answer vertical slice: question → answer → evaluation → evidence-grounded feedback → attempt record |
| **Person 4** | Handwriting & learner personalization | Image-input validation, replaceable handwriting-recognition adapter, transcription correction flow, common answer interface integration, attempt history and concept-level performance, transparent recommendation rules | Rebuilding retrieval index, model promotion decisions, duplicating grading logic | Image → transcription review → common evaluator path; attempt history; recommendations with reasons and evidence thresholds |

### Team-wide responsibilities

- Everyone opens small, reviewable PRs; no one commits directly to protected `main`.
- Agree on schemas and API contracts on Day 1. Do not silently rename fields or change response shapes.
- Use fixtures/mocks for external or expensive operations in fast CI tests.
- Do not reset or rebuild the shared Chroma index without coordinating with Person 2 and recording the corpus/config/model used.
- Record unresolved assumptions in the repository, not only in chat.
- If blocked for more than half a day, raise it and agree on a fallback rather than waiting until Day 6.

---

## 3. Daily milestone plan

## Day 1 — Repository audit, contracts, and a runnable baseline

**Milestone exit condition:** all members can explain the existing pipeline and have agreed interfaces, branch workflow, and acceptance tests. No one should begin a risky rewrite before the current state is checked.

| Owner | Tasks to finish by end of Day 1 | Concrete artifact / evidence | Acceptance criteria | Dependencies / coordination |
|---|---|---|---|---|
| **Person 1 — DevOps** | Inspect repository and existing workflows; document current branch state; establish or verify `dev` and short-lived feature branches; configure PR checks that can run quickly (lint, unit tests, config validation); verify `.gitignore` excludes secrets, local databases, Chroma persistence, model caches, and generated data where appropriate. | Branch/PR workflow note; first CI workflow PR; brief list of current CI failures or missing checks. | A small PR can trigger CI; no credentials or local index artifacts are committed; workflow does not run full-corpus indexing on every PR. | Ask team which test/lint commands already work. Do not invent commands before inspecting project config. |
| **Person 2 — Retrieval/ML** | Inspect `configs/retrieval.yaml`, corpus manifest, current embedding configuration, Chroma collection metadata/count, and available MLflow runs. Determine whether the full corpus index exists and which embedding revision/config created it. Define benchmark query format and relevance-label rules. | Baseline-state note; proposed benchmark schema; index/model compatibility gaps list. | The team can distinguish verified facts from unknowns; no claim that reset/rebuild succeeded without count/manifest evidence. | Coordinate with Person 3 on retrieval result shape and Person 1 on benchmark job execution. |
| **Person 3 — Backend/evaluation** | Map current FastAPI routes, Pydantic schemas, retrieval calls, question/answer fields, and storage layer. Define evaluation outcomes for MCQ, numerical, chemical expression, short text, and ambiguous/unsupported answers. Agree the common request/response contract. | Existing-flow map; proposed evaluation schema and example payloads; acceptance-test cases. | Existing code is reused where practical; correct, incorrect, invalid, and ambiguous cases are represented; feedback has source/provenance fields. | Owns question/evaluation contract. Review with Person 4 before interface is frozen. |
| **Person 4 — Handwriting/personalization** | Inspect frontend answer submission and backend upload support; test feasibility of one pretrained handwriting/OCR option on representative Chemistry handwriting; define transcription response including uncertainty/error; propose attempt-history fields and concept-level recommendation rules. | Short OCR feasibility note; transcription adapter contract; attempt-history/recommendation schema proposal. | OCR limitations are explicit (subscripts, charges, arrows, formulas); user can review/edit transcription before evaluation; no unreviewed OCR text silently treated as certain. | Use Person 3's common answer contract; do not commit to a model before testing representative examples. |
| **All** | Agree API/schema ownership, branch policy, demo scenario, and minimum acceptance criteria. Create a shared issue/task board with owner and due day for each deliverable. | One-page interface agreement and task board. | Every task has one owner, an acceptance test, and a target day; integration checkpoint booked for Day 3/4. | Resolve naming and schema conflicts now. |

**Day 1 decision gate:** If the current index, backend, or model state is unclear, prioritize discovery and reproducible inspection over destructive resets or broad refactoring.

## Day 2 — Independent foundations

**Milestone exit condition:** each workstream has a minimal implementation or executable baseline that can be reviewed independently.

| Owner | Tasks to finish by end of Day 2 | Concrete artifact / evidence | Acceptance criteria | Dependencies / coordination |
|---|---|---|---|---|
| **Person 1 — DevOps** | Complete fast CI for lint/unit tests/config checks; add a multi-stage Dockerfile or improve the existing one; add environment-variable validation and a health endpoint/check if missing; document local startup commands. | CI PR; Docker build log; environment variable example file with placeholders only. | CI runs without cloud credentials; Docker image builds; missing required configuration fails clearly without printing secrets. | Use a small fixture dataset; keep image size and build time reasonable. |
| **Person 2 — Retrieval/ML** | Build a fixed Chemistry query set and relevance labels; pin corpus snapshot and embedding config; implement benchmark evaluation for Recall@5 and MRR, plus latency where practical; run baseline against a verified index. | Versioned benchmark files and baseline JSON/CSV/report. | Same queries, corpus snapshot, and relevance labels are reusable for candidate comparison; metrics can be reproduced from documented command/config. | If labels are not ready, curate a smaller, defensible benchmark rather than report fabricated metrics. |
| **Person 3 — Backend/evaluation** | Implement typed-answer evaluation service and API path for at least MCQ and numerical questions; add explicit handling for missing answer, malformed input, unsupported answer type, and ambiguity; return feedback/evidence separately from correctness result. | Evaluation service/API PR; unit tests and example request/response. | Correct and incorrect test cases pass; invalid/ambiguous inputs do not crash or receive unjustified confident grading. | Confirm actual question schema and answer-key representation with team. |
| **Person 4 — Handwriting/personalization** | Implement image validation and a replaceable recognizer interface; prototype selected pretrained recognizer against a small sample; create editable transcription review UI or API handoff; define persistence for attempts and concept tags. | OCR adapter spike; sample transcription cases; history schema/migration proposal. | Unsupported file types and oversized images are rejected; recognizer can be swapped without rewriting grading service; learner correction is represented in flow. | Avoid hard-coupling a specific OCR provider/model to the evaluator. |
| **All** | Review each other's interface proposals asynchronously; add fixtures for one correct and one incorrect Chemistry question. | Reviewed contracts and shared fixture(s). | Both typed and handwritten paths converge on the same answer-evaluation input contract. | Keep fixtures free of private learner data. |

## Day 3 — First integration checkpoint

**Milestone exit condition:** at least one end-to-end typed-answer path works against the backend; handwriting output can be passed to the same evaluator; benchmark can run from a clean documented command.

| Owner | Tasks to finish by end of Day 3 | Concrete artifact / evidence | Acceptance criteria | Dependencies / coordination |
|---|---|---|---|---|
| **Person 1 — DevOps** | Add integration-test stage or documented local integration command; make CI test the API with fixtures/mocks; confirm Chroma/MLflow are optional or explicitly configured for tests rather than assumed available in every PR runner. | CI integration-test PR; short troubleshooting note. | Fast PR checks remain reliable and do not require developer-specific local state. | Coordinate service dependencies with Persons 2 and 3. |
| **Person 2 — Retrieval/ML** | Connect benchmark to the actual retrieval interface; verify returned learning-unit IDs, text, metadata, and provenance; record corpus/index manifest and baseline run in MLflow if environment is healthy. | Benchmark execution log; baseline run/artifacts; retrieval response examples. | Each result can be traced to a source block/chapter/page where metadata exists; baseline is labelled with model and corpus identity. | Coordinate result schema with Person 3; do not modify API contracts unilaterally. |
| **Person 3 — Backend/evaluation** | Integrate retrieval into feedback generation; return relevant NCERT evidence with provenance; persist or emit an attempt record through the agreed interface; add endpoint-level tests. | Typed-answer vertical-slice PR; integration tests. | Question → typed answer → correctness result → grounded explanation/evidence → attempt record works for at least one correct and one incorrect example. | Use retrieval response contract from Person 2; attempt schema agreed with Person 4. |
| **Person 4 — Handwriting/personalization** | Connect recognizer output to editable transcription and then to Person 3's evaluation request; add basic attempt-history read/write and concept tagging; implement the first rule-based recommendation function. | Handwriting-to-evaluator demo; history tests; recommendation examples with reasons. | Corrected transcription uses the same evaluator as typed input; recommendations state why a concept/question was selected. | Use Person 3's evaluator API; recommendation logic must not duplicate answer grading. |
| **All** | Hold a 30-minute integration review; assign every contract mismatch and integration defect to one owner. | Integration checklist with owner and severity. | No unresolved critical interface mismatch is left ownerless. | Target integration now; do not postpone all merges until the end of the week. |

## Day 4 — Vertical slice and candidate-model decision preparation

**Milestone exit condition:** the main user journey is demonstrable; benchmark and baseline are stable enough to compare a candidate model fairly.

| Owner | Tasks to finish by end of Day 4 | Concrete artifact / evidence | Acceptance criteria | Dependencies / coordination |
|---|---|---|---|---|
| **Person 1 — DevOps** | Run the app and dependencies through Docker Compose or documented local services; add startup/readiness checks and graceful failure for unavailable dependencies; review logging for accidental secrets or learner content exposure. | Docker smoke-test record; config/startup notes. | Fresh setup instructions work on a clean environment as far as available tooling permits; dependency failure is diagnosable. | Ask teammates to test instructions rather than only testing on own machine. |
| **Person 2 — Retrieval/ML** | Freeze baseline benchmark inputs; choose one feasible candidate embedding model/revision; verify vector dimension, normalization, tokenizer/sequence limits and index compatibility; define promotion gate before inspecting candidate results. | Candidate evaluation plan and promotion gate in version control. | Candidate and baseline use the same benchmark and evaluation protocol; threshold is explicit and includes quality plus operational considerations. | A new embedding space generally requires re-embedding documents and rebuilding the index. Do not compare incompatible vectors in one index. |
| **Person 3 — Backend/evaluation** | Extend evaluator to chemical expression/short-text cases where feasible; ensure uncertainty is explicit for unsupported free-form answers; refine feedback so retrieved evidence supports explanation rather than merely appearing beside it. | Expanded evaluator tests; grounded feedback examples. | Tests include accepted variants/normalization, unsupported response, and evidence unavailable; system does not imply retrieval alone proves answer correctness. | Prioritize predictable grading over broad but unreliable free-text grading. |
| **Person 4 — Handwriting/personalization** | Test representative handwriting cases including formula notation; add correction/edit path and error states; implement minimum-evidence threshold and concept-level accuracy aggregation. | Test-case table; learner history and recommendation examples. | OCR uncertainty/error is visible; one wrong answer alone does not permanently mark a concept as weak; recommendation has a human-readable rationale. | If OCR is unreliable for formula notation, keep manual correction and document known limits. |
| **All** | Run the complete demo flow from a fresh session and record defects. | Demo checklist and defect list. | The team can demonstrate typed and handwritten submissions, feedback, evidence, history, and a next-practice recommendation—or clearly identify any missing part. | Feature completeness must be judged by the actual running app, not screenshots alone. |

## Day 5 — Quality, model experiment, and failure handling

**Milestone exit condition:** important failure paths are tested; model selection is evidence-based or explicitly deferred for insufficient evidence.

| Owner | Tasks to finish by end of Day 5 | Concrete artifact / evidence | Acceptance criteria | Dependencies / coordination |
|---|---|---|---|---|
| **Person 1 — DevOps** | Add security-oriented configuration checks, non-root container where practical, dependency pinning/review, upload-size/time limits at the appropriate layer, safe error handling and production-readiness checklist. | Hardening PR; security/release checklist. | No secrets in image/repository; uploads are bounded; errors do not expose stack traces or sensitive configuration to users. | Deployment is out of scope; document remaining controls that require hosting/auth configuration. |
| **Person 2 — Retrieval/ML** | Run candidate benchmark if ready; compare Recall@5, MRR, latency, model size/memory and rebuild cost; log config, artifacts and results in MLflow; decide promote/reject/defer. | Baseline-vs-candidate report; MLflow run links/IDs or exported artifacts; model/index manifest. | Decision follows the predeclared gate; if candidate run is incomplete or noisy, report “inconclusive” instead of forcing a winner. | Ensure baseline and candidate runs are comparable; verify existing MLflow state before retraining or duplicating runs. |
| **Person 3 — Backend/evaluation** | Test retrieval unavailable/empty, malformed question records, absent answer keys, database errors and evidence insufficiency; improve API error semantics and regression tests. | Failure-path tests and backend test report. | Failures are explicit and recoverable where appropriate; app does not invent NCERT evidence or confidently grade unsupported cases. | Align error response format with frontend and Person 4. |
| **Person 4 — Handwriting/personalization** | Test invalid image, OCR timeout/failure, empty/low-confidence transcription, corrected transcription, repeat attempts and insufficient concept history; verify recommendations change with evidence. | Failure-path tests; recommendation rules and rationale documentation. | Failed OCR does not trigger grading on junk text; user can retry/type manually; recommendations respect evidence thresholds. | Reuse common evaluator errors and attempt schema. |
| **All** | Review unresolved P0/P1 defects and make explicit scope decisions. | Prioritized defect list with owner and disposition. | Every P0 is fixed or blocks release; P1 gaps are documented with workaround/owner. | Do not hide unfinished work to preserve the schedule. |

## Day 6 — Full integration, Docker, and release-candidate checks

**Milestone exit condition:** code is integrated, automated checks pass, and the app can be smoke-tested from documented setup instructions.

| Owner | Tasks to finish by end of Day 6 | Concrete artifact / evidence | Acceptance criteria | Dependencies / coordination |
|---|---|---|---|---|
| **Person 1 — DevOps** | Finalize CI stages, Docker image and compose/startup instructions; verify health checks; document environment variables, logs, known limitations and rollback/revert procedure; ensure branch protection/check requirements are configured where permissions allow. | Release-engineering PR; successful CI link/log; Docker smoke-test notes. | A teammate can build and run the app without undocumented local state; `main` remains protected/release-ready. | No CD or public deployment is required. Do not add cloud secrets. |
| **Person 2 — Retrieval/ML** | Ensure benchmark command and manifest are reproducible; document how to build/reset index safely; verify selected model and index are a compatible pair; attach final report/artifacts. | Final benchmark package and index manifest. | A fresh index can be built from canonical data/config; model revision, vector dimension, corpus version and build parameters are recorded. | Do not commit Chroma persistence files as the source of truth. |
| **Person 3 — Backend/evaluation** | Merge backend/evaluator changes; run unit and integration tests; verify response contracts, source attribution and attempt persistence; fix critical integration issues. | Backend test report and merged PR(s). | Correct/incorrect/ambiguous cases pass; NCERT evidence provenance is retained; API contract is stable. | Coordinate final API changes with Person 4 before merge. |
| **Person 4 — Handwriting/personalization** | Merge handwriting and learner-history changes; run end-to-end tests; document OCR limitations, correction workflow and recommendation thresholds; verify UI/API behavior on failures. | End-to-end test evidence and user-flow notes. | Handwritten and typed inputs converge on one evaluator; recommendation explanations are visible; failure states have a safe fallback. | If a full UI integration is blocked, deliver a tested API path and document the exact remaining integration gap. |
| **All** | Freeze feature scope; review PRs; run regression suite on integrated branch. | Release candidate tag/commit or clearly named candidate branch; test summary. | No unreviewed changes, no known P0 bugs, and no unexplained failing checks. | Only cherry-pick or merge fixes that can be tested before the cutoff. |

## Day 7 — Clean-environment demo and handoff

**Milestone exit condition:** the team demonstrates the release candidate, hands over reproducible artifacts and documentation, and distinguishes completed work from known gaps.

| Owner | Tasks to finish by end of Day 7 | Concrete artifact / evidence | Acceptance criteria | Dependencies / coordination |
|---|---|---|---|---|
| **Person 1 — DevOps** | Run setup from a clean environment or with a teammate following the instructions; finalize release checklist, rollback notes, secrets/config guide, and deployment blockers. | Release-readiness document and clean-start test record. | Another teammate can follow setup instructions; deployment blockers and security work still required are explicit. | No public deployment during this milestone. |
| **Person 2 — Retrieval/ML** | Present benchmark protocol, baseline/candidate result, MLflow evidence and final model/index decision; provide rerun instructions. | Final ML evaluation report with artifacts and decision. | Results are reproducible and limitations/sample size are disclosed; no unsupported claim that candidate is better. | If candidate evaluation did not finish, provide baseline and a dated follow-up task rather than inventing a result. |
| **Person 3 — Backend/evaluation** | Demonstrate typed-answer flow and explain evaluation boundaries; finalize API/schema documentation and tests. | API contract and test summary. | Demo shows at least one correct and one incorrect answer with grounded explanation and saved attempt. | Use the agreed demo fixture to avoid last-minute data variability. |
| **Person 4 — Handwriting/personalization** | Demonstrate image upload, transcription correction, evaluation, attempt history and next-practice recommendation; summarize OCR limitations. | User-flow notes and personalization rules. | Demo shows at least one corrected transcription and one transparent recommendation; gaps are documented. | If OCR cannot reliably parse a particular notation, show manual correction and state the limitation. |
| **All** | Conduct final acceptance review; capture demo evidence, known issues, owners, and post-week priorities. | Final sign-off checklist and follow-up backlog. | Team can state what is done, what is not done, how to reproduce tests, and what remains before deployment. | Release candidate is not described as production-ready unless production requirements are actually met. |

---

## 4. Interface contracts to settle on Day 1

These are **conceptual minimums**, not a mandate to replace the project's existing Pydantic schemas. Adapt names and field types to the repository, then version the agreed contract.

### 4.1 Question/evaluation contract — owner: Person 3

| Concept | Minimum information | Notes |
|---|---|---|
| Question | Stable question ID, subject, question type, prompt, answer key/accepted answers, concept IDs, difficulty if available | Avoid copying answer keys into user-facing question text if the product flow should hide them. |
| Answer submission | Question ID, answer text, input mode, optional user correction/transcription metadata, attempt/session identifier | Typed and handwritten paths must converge here. |
| Evaluation result | Correct / incorrect / partial / uncertain (as supported), score if applicable, feedback, normalized answer where safe, evaluator version | Do not imply reasoning is correct solely because the final answer matches. |
| Evidence | Retrieved unit IDs, source/chapter/page/block provenance, short supporting excerpts where appropriate | Evidence should support feedback and be traceable. |
| Attempt record | Learner/session ID appropriate to the app's privacy model, question ID, concept IDs, result, timestamp, answer mode | Collect only what is needed; protect learner data. |

### 4.2 Handwriting contract — owner: Person 4

| Field / behavior | Requirement |
|---|---|
| Input | Image bytes/file handle plus validated MIME/type and size; do not trust file extension alone. |
| Output | Transcribed text, recognizer/model identity, confidence or uncertainty metadata if available, warnings/errors. |
| Review | User can edit/correct transcription before submission; corrected text, not raw uncertain OCR, goes to the evaluator. |
| Failure | Clear failure/low-confidence state with retry or manual typing; never silently submit empty or clearly broken text. |
| Replaceability | Evaluator depends on the contract, not a specific OCR vendor/model. |

### 4.3 Retrieval contract — owner: Person 2; app integration: Person 3

| Field / behavior | Requirement |
|---|---|
| Input | Query text and top-k/config options supported by the current retriever. |
| Output | Ranked learning-unit ID, text, score if meaningful, and provenance metadata. |
| Reproducibility | Benchmark records query IDs, relevant unit IDs, corpus snapshot, embedding model revision, retrieval configuration and run date/version. |
| Index compatibility | Document embedding model/revision, dimension, normalization, corpus version and index-build settings together. |
| Safety | Retrieval results are evidence candidates; they are not automatically the answer key or proof that a student's reasoning is correct. |

### 4.4 Learner-history/recommendation contract — owner: Person 4

| Field / behavior | Requirement |
|---|---|
| History | Record attempts and concept tags using the agreed question/evaluation schema. |
| Aggregation | Calculate concept-level performance from multiple attempts; document the window and minimum evidence threshold. |
| Recommendation | Return next question/practice item plus a human-readable reason, such as repeated errors on a concept. |
| Initial approach | Deterministic rules first; do not train a recommendation model in this one-week scope. |
| Privacy | Use only necessary identifiers and data; avoid exposing one learner's history to another. |

---

## 5. Definition of done by workstream

| Workstream | Minimum definition of done | Not sufficient on its own |
|---|---|---|
| DevOps/release | PR-based CI, Docker build and smoke test, validated config, health/readiness behavior, reproducible setup, documented remaining production controls | “Dockerfile added”; workflow exists but never ran; credentials copied into repository |
| Embedding/ML | Fixed benchmark, baseline metrics, candidate result if feasible, MLflow evidence, predeclared gate, model/index compatibility manifest, reproducible index build | A model was changed because it “seems better”; one-off metric with no benchmark labels; incompatible embeddings mixed in one index |
| Backend/evaluation | Tested typed-answer path, explicit uncertain/unsupported handling, NCERT-grounded feedback, provenance, attempt recording, failure-path tests | A correctness boolean with no explanation/evidence; free-text grading claimed reliable without tests |
| Handwriting/personalization | Validated image input, replaceable recognizer, correction-before-evaluation, shared evaluator path, attempt history, transparent concept recommendations and thresholds | OCR text is accepted without review; one error triggers a permanent weak-concept label; recommendations have no rationale |
| Integrated app | Reproducible demo for typed and handwritten input; feedback, source evidence, attempt history and next-practice suggestion; green required checks or documented blockers | A screenshot-only demo, undocumented local setup, or a public URL with no security review |

---

## 6. Priority and fallback rules

Use these priorities if the week slips. The team should explicitly report any deferred item rather than imply completion.

| Priority | Items | Decision rule |
|---|---|---|
| **P0 — must protect** | Git/PR workflow and CI; verified current-state audit; reproducible retrieval baseline; typed-answer evaluation; grounded feedback; basic integration tests | If these fail, do not claim a release candidate is tested. Fix or document a blocking defect. |
| **P1 — expected this week** | Embedding candidate comparison and MLflow; attempt history; transparent concept recommendations; Docker end-to-end smoke test | Deliver the smallest reproducible implementation. If model comparison lacks defensible labels/results, report it as inconclusive and preserve the baseline. |
| **P2 — scope-risk items** | High-quality Chemistry handwriting recognition across complex notation; final polish; optional benchmark automation | Keep the adapter and correction flow working; disclose recognition limits. Do not pretend difficult formula OCR is solved if it is not. Any deferral must name an owner and next step. |

**Important:** Handwritten input is part of the agreed product scope. P2 describes the risk of high-quality recognition and polish, not permission to omit the handwriting pathway without recording the gap and agreeing with the team.

---

## 7. Suggested branch and review workflow

1. Create short-lived branches such as `feat/answer-evaluation`, `feat/handwriting-adapter`, `feat/retrieval-benchmark`, and `chore/ci-docker`.
2. Open PRs into `dev`; require CI checks and review where repository permissions permit.
3. Merge small, tested changes. Do not bundle unrelated workstreams into one large PR.
4. Prepare a release PR from `dev` to protected `main` after integration review.
5. Keep `main` release-ready. No automatic public deployment is required this week; do not add cloud credentials just to make a CI pipeline appear complete.
6. If a contract changes, update the schema, example payloads, tests, and relevant consumers in coordinated PRs.

### CI guidance

- **On every PR:** lint/format checks, unit tests, config/schema validation, lightweight security checks if already available.
- **Integration:** use fixtures and test services; avoid dependence on personal local Chroma/MLflow state.
- **Docker:** build and smoke-test the image as a separate or later stage if it keeps PR feedback fast enough.
- **Retrieval benchmark:** run on demand or when retrieval/model/data changes; do not re-embed the entire corpus on every PR.
- **Later deployment:** add protected environment approval, appropriate secret management and short-lived identity/OIDC where supported, then a live smoke test. This is future work, not a Day 7 requirement.

---

## 8. Final acceptance checklist

Mark each item only after observing the evidence. A checkbox is not a claim that it has already passed.

- [ ] A clean checkout can follow documented setup instructions.
- [ ] CI checks run on PRs; required checks and branch protection are configured where permissions allow.
- [ ] Docker image builds and the local smoke test passes, or a blocker is explicitly documented.
- [ ] One typed correct answer and one typed incorrect answer produce tested outcomes and NCERT-grounded feedback.
- [ ] An image submission can be transcribed, corrected, and sent through the same evaluator as typed text.
- [ ] OCR failure/uncertainty has a safe retry/manual-entry path.
- [ ] Attempts are recorded and concept-level recommendations explain their rationale and evidence threshold.
- [ ] Retrieval benchmark query set, relevance labels, corpus snapshot, and baseline metrics are versioned.
- [ ] Candidate embedding model is promoted, rejected, or marked inconclusive using a predeclared gate.
- [ ] MLflow artifacts/results and the model/index compatibility manifest are available.
- [ ] No Chroma persistence files, secrets, or local-only databases are accidentally treated as canonical source data.
- [ ] Known limitations, production blockers, owners, and follow-up dates are documented.
- [ ] The final demo shows the working application, not only static screenshots.

## 9. Day 7 handoff format

Each owner should provide this short summary in the final review:

| Prompt | Required answer |
|---|---|
| What did I deliver? | Link to PR/commit, document, report, or runnable artifact. |
| What evidence shows it works? | Test command/results, CI link, benchmark artifact, or reproducible demo steps. |
| What remains incomplete? | Specific gap and impact—no vague “polish needed.” |
| What could fail in production? | Concrete risk and mitigation or reason it is deferred. |
| Who owns the next step and by when? | Named team role/person and target date. |

**Final release language:** if the checklist passes, describe the outcome as a **tested release candidate with documented production blockers**. Use “production-ready” only after deployment-specific security, reliability, monitoring, access-control, and operational requirements have actually been validated.
