# NCERT retrieval comparison — MLflow registration

This package registers previously computed MiniLM and ChEmbed metrics and their
supporting retrieval artefacts. It does not rerun retrieval, rebuild Chroma, or
recalculate metrics.

## Before registering: pin both model revisions

Edit `configs/retrieval_mlflow.yaml` and set `models.minilm.model_revision`
and `models.chembed.model_revision` to the exact 40-character Hugging Face
commit SHA used for the corresponding index build / benchmark. Both original
index manifests currently record `model_revision: null`, so the committed
historical provenance does not prove the exact revisions that produced the
vectors. Do not represent a newly resolved `main` revision as historical proof.

For reference, Hugging Face currently reports MiniLM revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`; this can be pinned for a future
rebuild, but it should only be used as the historical revision if it is confirmed
against the local Hugging Face cache used for the benchmark. ChEmbed's exact
revision must be recovered from the machine/cache that built `chembed_index`
or resolved by the team and clearly treated as a future pinned version.

### Recover the revision from the indexing machine

On the Windows machine that built each collection, inspect the cached `main` ref
before downloading or updating that model again:

```powershell
$hub = if ($env:HF_HOME) { Join-Path $env:HF_HOME "hub" } else { Join-Path $env:USERPROFILE ".cache\huggingface\hub" }

Get-Content (Join-Path $hub "models--sentence-transformers--all-MiniLM-L6-v2\refs\main")
Get-Content (Join-Path $hub "models--BASF-AI--ChEmbed-full\refs\main")
```

Copy the 40-character commit IDs into the corresponding `model_revision` fields.
This is the best route to recover the revision cached locally at index-build time;
if that cache/ref was refreshed after the experiment, it is not proof of the
historical revision. In that case, document the limitation rather than guessing.

ChEmbed is gated. Each teammate needs access to BASF-AI/ChEmbed-full and must set
`HF_TOKEN` in the root `.env`. The registration script loads `.env` without
overriding existing environment variables. Never commit tokens.

## Register

From the repository root:

```bash
pip install -r requirements.txt
python -m src.retrieval_mlflow.register
```

The script follows `src.track_b.tracking`: `MLFLOW_TRACKING_URI` overrides the
default `sqlite:///mlflow.db`. To inspect the local store:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Open http://localhost:5001. Each teammate registers to their own local MLflow
store; Git shares registration inputs and code, not MLflow databases.

The YAML explicitly records model identity, model revision, tokenizer repository /
loader / revision policy, embedding dimension, maximum sequence length, pooling,
normalization, and ChEmbed's prefixes / remote-code / safetensors settings. The
`model_artifacts` block defines full-repository snapshot capture and SHA-256
checksum policy. At registration time, each run records metrics, source
JSON/config/index manifests, immutable model revision, tokenizer/config/weights/
custom-code snapshot, per-file SHA-256 checksums, and a snapshot tree hash. Model files are downloaded to a
temporary directory and logged to MLflow; they are not committed to Git.
Re-registering identical model snapshots, metrics and input artefacts is skipped
using a content fingerprint.

## Metrics provenance

Metrics in `configs/retrieval_metrics_v1.json` are the supplied judged results
for 10 NCERT Chemistry queries at k=5. They are logged verbatim, not recomputed.
Recall is pool-based over judged retrieval results, not exhaustive corpus recall.

Run the focused tests with `pytest tests/test_retrieval_mlflow.py`.
