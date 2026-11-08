"""Read-only integrity audit for the ChEmbed Chroma collection.

Run from repository root:
    python scripts/check_chembed_index.py
"""
from __future__ import annotations
import argparse, hashlib, json, math, sys
from datetime import datetime, timezone
from pathlib import Path
import chromadb
import yaml

ROOT = Path(__file__).resolve().parents[1]

def rp(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p

def unique_path(folder, stem):
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"{stem}.json"
    if not p.exists():
        return p
    for i in range(1, 10000):
        p = folder / f"{stem}_{i:02d}.json"
        if not p.exists():
            return p
    raise RuntimeError("Could not allocate unique report filename")

def read_jsonl(path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {n}: {e}") from e
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at line {n}")
            rows.append(row)
    return rows

def expected_docs(rows, dc):
    key = dc.get("id_field", "unit_id")
    template = dc["text_template"]
    skip_empty = bool(dc.get("skip_empty_text", True))
    expected, skipped = {}, 0
    for n, row in enumerate(rows, 1):
        rid = row.get(key)
        if not isinstance(rid, str) or not rid.strip():
            raise ValueError(f"Missing/invalid {key!r} in source record {n}")
        if rid in expected:
            raise ValueError(f"Duplicate source ID: {rid}")
        if not str(row.get("text", "")).strip():
            if skip_empty:
                skipped += 1
                continue
            raise ValueError(f"Empty source text: {rid}")
        try:
            doc = template.format(**row).strip()
        except (KeyError, ValueError, IndexError) as e:
            raise ValueError(f"Template formatting failed for {rid}: {e}") from e
        if not doc:
            if skip_empty:
                skipped += 1
                continue
            raise ValueError(f"Empty formatted document: {rid}")
        expected[rid] = doc
    return expected, skipped

def scan_collection(col, batch_size):
    ids, empty_docs = [], []
    seen, duplicates = set(), []
    total = col.count()
    for offset in range(0, total, batch_size):
        r = col.get(limit=min(batch_size, total-offset), offset=offset,
                    include=["documents"])
        for rid, doc in zip(r.get("ids") or [], r.get("documents") or []):
            if rid in seen:
                duplicates.append(rid)
            seen.add(rid)
            ids.append(rid)
            if not isinstance(doc, str) or not doc.strip():
                empty_docs.append(rid)
    return ids, empty_docs, duplicates

def inspect_vectors(col, sample_size, expected_dim):
    r = col.get(limit=sample_size, include=["embeddings"])
    vectors = r.get("embeddings")
    if vectors is None or len(vectors) == 0:
        return {"passed": False, "sample_size": 0, "dimension": None,
                "finite": False}
    dims = {len(v) for v in vectors}
    finite = all(math.isfinite(float(x)) for v in vectors for x in v)
    dim = next(iter(dims)) if len(dims) == 1 else None
    dim_ok = expected_dim is None or dim == expected_dim
    return {"passed": len(dims) == 1 and finite and dim_ok,
            "sample_size": len(vectors), "dimension": dim,
            "expected_dimension": expected_dim,
            "consistent_dimensions": len(dims) == 1,
            "finite": finite, "dimension_matches_config": dim_ok}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/retrieval_chembed.yaml")
    ap.add_argument("--expected-count", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--embedding-sample-size", type=int, default=5)
    a = ap.parse_args()
    if a.batch_size < 1 or a.embedding_sample_size < 1:
        ap.error("batch/sample sizes must be >= 1")

    config_path = rp(a.config)
    if not config_path.is_file():
        ap.error(f"Config not found: {config_path}")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dc, vc, ec = cfg["document"], cfg["vector_store"], cfg.get("embedding", {})
    source = rp(dc["input_path"])
    if not source.is_file():
        ap.error(f"Source JSONL not found: {source}")

    rows = read_jsonl(source)
    expected, skipped = expected_docs(rows, dc)
    host, port = vc.get("host", "localhost"), int(vc.get("port", 8000))
    cname = vc.get("collection_name", "chembed_index")
    metric_expected = vc.get("distance_metric", "cosine")
    dim_expected = ec.get("embedding_dimension")
    dim_expected = int(dim_expected) if dim_expected is not None else None

    print(f"Connecting to Chroma at {host}:{port} ...")
    client = chromadb.HttpClient(host=host, port=port)
    try:
        col = client.get_collection(name=cname)
    except Exception as e:
        raise RuntimeError(f"Collection {cname!r} not found; audit will not create it.") from e

    count = col.count()
    meta = col.metadata or {}
    metric_actual = meta.get("hnsw:space")
    ids, empty_docs, duplicates = scan_collection(col, a.batch_size)
    actual_set, expected_set = set(ids), set(expected)
    missing, unexpected = sorted(expected_set-actual_set), sorted(actual_set-expected_set)
    vector_check = inspect_vectors(col, a.embedding_sample_size, dim_expected)

    manifest_dir = rp(cfg.get("outputs", {}).get(
        "manifest_dir", "reports/retrieval/chembed"))
    manifests = sorted(manifest_dir.glob("index_manifest_*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True) if manifest_dir.exists() else []
    manifest_path = manifests[0] if manifests else None
    manifest_checks = {"found": manifest_path is not None,
                       "path": str(manifest_path) if manifest_path else None}
    if manifest_path:
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_checks.update({
            "collection_name_matches": m.get("collection_name") == cname,
            "collection_count_matches": m.get("collection_count_after_build") == count,
            "source_sha256_matches": m.get("input_sha256") ==
                hashlib.sha256(source.read_bytes()).hexdigest(),
            "manifest_embedding_dimension": m.get("embedding", {}).get("dimension")
        })

    checks = {
        "collection_non_empty": count > 0,
        "expected_count_matches": a.expected_count is None or count == a.expected_count,
        "source_indexable_count_matches": len(expected) == count,
        "all_expected_ids_present": not missing,
        "no_unexpected_ids": not unexpected,
        "no_duplicate_ids": not duplicates,
        "no_empty_stored_documents": not empty_docs,
        "distance_metric_matches_config": metric_actual == metric_expected,
        "sample_embeddings_valid": vector_check["passed"],
        "manifest_found": manifest_path is not None,
    }
    if manifest_path:
        checks.update({
            "manifest_collection_matches": manifest_checks["collection_name_matches"],
            "manifest_count_matches": manifest_checks["collection_count_matches"],
            "manifest_source_hash_matches": manifest_checks["source_sha256_matches"],
        })

    warnings = []
    if ec.get("pooling") is None:
        warnings.append("Pooling is not explicitly configured; verify ChEmbed's documented pooling.")
    if not ec.get("document_prefix") and not ec.get("query_prefix"):
        warnings.append("No task prefixes configured; verify whether ChEmbed requires them.")
    if metric_actual is None:
        warnings.append("Collection metadata did not expose hnsw:space.")
    if not manifest_path:
        warnings.append("No index manifest found in configured manifest directory.")

    report = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path), "source_path": str(source),
        "collection": {"name": cname, "host": host, "port": port, "count": count,
                       "metadata": meta, "configured_distance_metric": metric_expected,
                       "actual_distance_metric": metric_actual},
        "source": {"records_read": len(rows), "records_expected_in_index": len(expected),
                   "records_skipped_empty": skipped,
                   "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
        "id_integrity": {"stored_ids": len(ids), "unique_stored_ids": len(actual_set),
                         "missing_count": len(missing), "missing_sample": missing[:20],
                         "unexpected_count": len(unexpected), "unexpected_sample": unexpected[:20],
                         "duplicate_count": len(duplicates), "empty_document_count": len(empty_docs),
                         "empty_document_sample": empty_docs[:20]},
        "embedding_sample_check": vector_check, "manifest": manifest_checks,
        "checks": checks, "passed": all(checks.values()), "warnings": warnings
    }

    outdir = rp(cfg.get("outputs", {}).get("audit_dir", "reports/retrieval/chembed"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = unique_path(outdir, f"index_audit_{stamp}")
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n",
                      encoding="utf-8")

    print("\n=== ChEmbed index audit ===")
    for name, ok in checks.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\nCollection: {cname} ({count} records)")
    print(f"Embedding sample: {vector_check['sample_size']} vectors; "
          f"dimension={vector_check['dimension']}; finite={vector_check['finite']}")
    for warning in warnings:
        print(f"[WARN] {warning}")
    print(f"\nOverall: {'PASS' if report['passed'] else 'FAIL'}")
    print(f"Report: {output}")
    return 0 if report["passed"] else 2

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"AUDIT ERROR: {exc}", file=sys.stderr)
        raise
