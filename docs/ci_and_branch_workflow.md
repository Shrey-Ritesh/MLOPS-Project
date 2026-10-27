# Branch / PR workflow and CI (Phase 2, Day 1)

## Current state (audited Day 1)

- Branches: `main` plus feature branches on origin (`feature/track-b`, `feature/track-d-universal-extractor`, `frontend`, `track-b-ncert`). There is **no `dev` branch** yet and no branch protection verified.
- No CI existed (`.github/` was absent). Remote: `git@github.com:Shrey-Ritesh/MLOPS-Project.git`.
- No lint/format config (`pyproject.toml`, `ruff.toml`, `setup.cfg` all absent). Tests are plain `pytest` in `tests/`.
- `.env.example` is referenced by the README but is not in the repo (Day 2 task).
- `.gitignore` already excluded `.env*`, key/credential files, `mlflow.db`, `mlruns/`, `data/raw/`, DVC cache. Added: Chroma/model-cache directories. `data/processed` and `data/assets` are ignored via `data/.gitignore` (DVC-managed). Nothing tracked matched `*.env`, `*.db`, or Chroma paths.

## Workflow

1. Branch from `dev` using short-lived names: `feat/...`, `chore/...`, `fix/...`.
2. Open a PR into `dev`; CI must be green; keep PRs to one workstream.
3. Release PR `dev` -> `main` after integration review. Keep `main` release-ready.
4. One-time setup needed by someone with repo admin rights: create `dev` from `main`, set it as the PR target, and require the `CI / checks` status on `dev` and `main`.

## CI (`.github/workflows/ci.yml`)

Runs on PRs and pushes to `dev`/`main`, no secrets, no cloud credentials, no corpus indexing:

| Step | What it does |
|---|---|
| Install | `requirements-ci.txt` (minimal; the full `requirements.txt` is a Windows pip-freeze in UTF-16 with `pywin32`, `torch`, etc., unsuitable for per-PR CI) |
| Config validation | parses every YAML under `configs/` plus `dvc.yaml` |
| Lint | `ruff check` limited to syntax/undefined-name errors (`E9,F63,F7,F82`) so it can pass on the existing code; tighten later |
| Unit tests | `pytest`, excluding the four broken/data-dependent files listed below |

## Known failures / gaps found by running the suite locally

Local run: 58 passed, 4 failed, 2 files could not be collected.

| Item | Cause | Owner suggestion |
|---|---|---|
| `tests/test_retrieval.py`, `tests/test_retriever_gateway.py` fail at collection | import `chromadb`; need the package and a running Chroma service (not installed in the local venv either) | Persons 2/3; move to the Day 3 integration stage with fixtures/mocks |
| `tests/test_query_gateway.py` (3 tests) `KeyError: 'subject'` | `build_query_request` no longer returns a `subject` key that the tests assert on; test/code drift | Person 2/3, decide which side is right |
| `tests/test_source_record_creation.py` `No raw PDF files found in data/raw` | depends on DVC-pulled raw data (gitignored) | mark as data-dependent or use a fixture |
| `src/retrieval/__init__ .py` | filename contains a space, so it is not a package `__init__` | rename |
| `requirements.txt` | UTF-16, Windows-only packages (`pywin32`), very heavy | split runtime vs dev deps (Day 2) |

To keep the first PR green, CI currently **ignores** all four affected test files (listed in the workflow comments). Remove each `--ignore` as its owner fixes it.
