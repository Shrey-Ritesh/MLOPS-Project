# Embedding model artefact policy

- Keep model ID, immutable revision, tokenizer and inference configuration in Git.
- Download the exact HF snapshot during MLflow registration and log it as a run artefact.
- Include the model weights, tokenizer, model config, custom remote-code files and
  SHA-256 hashes. Do not commit large duplicate model snapshots into Git.
- Keep `HF_TOKEN` in local `.env`; never log or commit it.
- `trust_remote_code=true` is part of ChEmbed provenance and its custom code is
  included in the snapshot artefact.
- Check the model licence against intended use before deployment or distribution.
