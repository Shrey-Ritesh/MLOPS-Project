# Data Contract — JEE Tutor Data Pipeline

**Contract version:** `1.0.0`  
**Status:** Draft for team approval  
**Canonical format:** UTF-8 JSON Lines (`.jsonl`)  
**Validation:** Pydantic models under `src/models/`

---

## 1. Purpose and scope

This document defines the shared data contract for source inventory, NCERT extraction, asset extraction, and JEE Main question extraction. It enables Tracks A–D to work independently while producing compatible, traceable, reproducible outputs.

Original PDFs remain immutable source inputs. Canonical JSONL records are structured representations derived from those sources. Retrieval chunks, embeddings, vector indexes, and learning-unit records are downstream derived artifacts, not replacements for source records.

This contract does not define retrieval chunking, embeddings, model training, mastery classification, or the final schema for canonical learning units. Passing validation does not prove that extracted content or labels are semantically correct.

## 2. Shared conventions

### 2.1 JSONL

Each `.jsonl` file contains one complete JSON object per line, encoded as UTF-8. Each file may contain many records. Generated JSONL is pipeline output: do not hand-edit it to fix extraction errors. Correct the source configuration, extraction logic, or an explicitly tracked review input, then rerun the pipeline.

### 2.2 Required, unknown, and empty values

- Required fields must be present.
- Use JSON `null` for an unknown or inapplicable nullable scalar field.
- Use `[]` for a list with no known entries.
- Do not use `"N/A"`, `"unknown"`, or `"null"` as substitutes for JSON null unless a field explicitly defines such an enumerated value.
- Do not fabricate metadata, answers, solutions, associations, or difficulty labels.
- Nullability must be enforced by the Pydantic model.

### 2.3 Encoding and page numbering

- Text files are UTF-8.
- `page_start`, `page_end`, `source_page`, and content-block `page` use **one-based PDF page indices**. PDF page 1 is the first physical page in the PDF, regardless of **printed page numbering.**
- If printed page labels are needed, store them separately.
- Preserve meaningful mathematical and chemical notation, signs, superscripts/subscripts, option labels, headings, table structure, and content order as far as the extraction method permits.

### 2.4 Status values

Initial status values:

- `extraction_status`: `success`, `needs_review`, or `failed`.
- Asset `review_status`: `unreviewed`, `verified`, or `needs_review`.
- Source `inventory_status`: `registered`, `verified`, or `quarantined`.
- JEE `difficulty_status`: `provisional`, `reviewed`, `validated`, or `unknown`.

`success` means the extraction process completed according to its current rules; it does not guarantee semantic correctness. Records marked `needs_review` must remain visible in quality reports.

## 3. Source manifest — `source_manifest.jsonl`

**Granularity:** one record per registered source-document version.

| Field | Type | Required | Description |
|---|---|---:|---|
| `source_id` | string | Yes | Stable, unique ID for this specific registered document version. |
| `logical_source_id` | string or null | Yes | Optional ID linking related editions/revisions of the same logical textbook or paper series; use null if not maintained. |
| `source_type` | enum | Yes | `ncert` or `jee_main`. |
| `original_filename` | string | Yes | Filename as obtained from the source. |
| `sha256` | string | Yes | SHA-256 of exact PDF bytes, lowercase hexadecimal, 64 characters. |
| `file_size_bytes` | integer | Yes | Exact file size in bytes; greater than zero. |
| `subject` | string or null | Yes | Subject where known. |
| `language` | string or null | Yes | BCP 47 language tag where known, e.g. `en`, `hi`. |
| `publication_year` | integer or null | Yes | Publication year where applicable and known, particularly for NCERT. |
| `exam_year` | integer or null | Yes | Examination year where applicable and known, particularly for JEE Main. |
| `edition` | string or null | Yes | Edition/revision designation where known. |
| `class_level` | integer or null | Yes | Class level where applicable and known. |
| `session` | string or null | Yes | JEE Main session where applicable and known. |
| `shift` | string or null | Yes | JEE Main shift where applicable and known. |
| `inventory_status` | enum | Yes | `registered`, `verified`, or `quarantined`. |

All nullable keys above remain present in serialized v1 records; use JSON `null` if unknown or inapplicable. Source-type-specific validation must not require NCERT-only metadata on JEE records or vice versa.

### 3.1 Identity and checksum rules

These fields answer different questions:

- `source_id` identifies the registered document version within this dataset.
- `sha256` identifies the exact byte contents of the PDF.
- `logical_source_id`, when used, groups related versions of the same logical publication or paper series.

If a revised textbook edition or corrected paper has different bytes, register it as a distinct document version with a distinct `source_id`. It may share a `logical_source_id` with related versions. Do not overwrite an existing manifest entry or silently replace the old file.

A SHA-256 match indicates identical bytes with extremely high confidence; it does not prove authenticity, edition, or semantic correctness. A checksum mismatch after transfer must block verification until investigated.

### 3.2 Inventory rules

- Compute the checksum from exact file bytes and record byte size before extraction.
- Verify the checksum after copying or downloading a file to another machine.
- Preserve the original PDF unchanged.
- Do not infer publication/examination metadata solely from an unverified filename. Leave unknown values null and flag them for review if they affect use of the source.
- IDs must not depend on local absolute paths.

## 4. NCERT chapter manifest — `ncert_chapters.jsonl`

**Granularity:** one record per chapter in a specific source document version and language.

A chapter record is the canonical extracted representation of a textbook chapter. It preserves chapter-level provenance and ordered internal structure. It is not itself a retrieval chunk or a learning unit.

| Field | Type | Required | Description |
|---|---|---:|---|
| `chapter_id` | string | Yes | Stable, unique ID for this chapter record. Different source versions/languages have distinct chapter IDs. |
| `source_id` | string | Yes | Must refer to a registered NCERT source. |
| `subject` | string | Yes | Subject as represented in the source. |
| `class_level` | integer or null | Yes | Class level where known. |
| `language` | string | Yes | Language tag for this chapter record. |
| `chapter_number` | string or null | Yes | Original chapter number/label, if present. |
| `chapter_title` | string or null | Yes | Chapter title as extracted, if present. |
| `page_start` | integer | Yes | One-based PDF page index where the chapter begins. |
| `page_end` | integer | Yes | One-based PDF page index where the chapter ends; at least `page_start`. |
| `text_raw` | string | Yes | Full chapter text in extraction order before cleaning. |
| `text_cleaned` | string | Yes | Deterministically cleaned whole-chapter text. |
| `content_blocks` | array | Yes | Ordered representation of internal content structure, as defined below. May be empty only for a failed extraction record. |
| `asset_ids` | array of strings | Yes | Known associated asset IDs; may be empty. |
| `extraction_status` | enum | Yes | `success`, `needs_review`, or `failed`. |

`text_raw` and `text_cleaned` provide whole-chapter text views. `content_blocks` preserves structure and order for downstream learning-unit construction. The block list is nested in the chapter record in contract v1.0.0; it is not a separate JSONL file.

### 4.1 `content_blocks` object

| Field | Type | Required | Description |
|---|---|---:|---|
| `block_id` | string | Yes | Unique within its chapter; stable across deterministic reruns of the same source and extraction configuration. |
| `block_type` | enum | Yes | `heading`, `paragraph`, `equation`, `table`, `figure`, `caption`, `list`, or `other`. |
| `text_raw` | string or null | Yes | Extracted block text before cleaning; null for non-textual blocks with no extracted text. |
| `text_cleaned` | string or null | Yes | Cleaned block text; null when no text content exists. |
| `page` | integer | Yes | One-based PDF page index containing the block. |
| `asset_ids` | array of strings | Yes | Associated assets, possibly empty. |
| `order_index` | integer | Yes | Zero-based order within the chapter; unique and contiguous. |
| `original_label` | string or null | Yes | Original heading/number/label if present, without imposing a numbered-section taxonomy. |

Preserve order, headings, tables, equations, captions, and figure references as faithfully as possible. If a logical element spans pages, the implementation must document whether it becomes one block or multiple ordered blocks and apply that policy consistently.

Do not assume every block is a semantic unit or should be independently retrieved. Related paragraphs, equations, tables, and figures may need to be grouped when deriving learning units.

### 4.2 Chapter and learning-unit relationship

- The chapter is the canonical extracted source record.
- A learning unit is a downstream educational/conceptual record and will have its own future contract.
- A learning unit should trace back to its parent `chapter_id` and, where practical, relevant `block_id` values and source page range.
- Chapter boundaries and original textbook section labels are not synonymous with learning-unit boundaries.
- Different language editions are separate source/chapter records. Cross-language equivalence links may be introduced later as reviewed metadata.

### 4.3 Text-cleaning rules

Cleaning must be deterministic and preserve meaningful content. Preserve mathematical/chemical notation, negative signs, superscripts/subscripts, labels, table structure, headings, and content order as far as practical. Remove recurring headers/footers only when they can be identified reliably. Keep `text_raw` so cleaning decisions can be revisited without re-extracting the PDF, where possible.

## 5. Asset manifest — `assets_manifest.jsonl`

**Granularity:** one record per extracted image, diagram, figure, or page crop. Binary files are stored separately from the JSONL manifest.

| Field | Type | Required | Description |
|---|---|---:|---|
| `asset_id` | string | Yes | Stable, unique ID for the asset record. |
| `source_id` | string | Yes | Source PDF; must exist in the source manifest. |
| `sha256` | string | Yes | SHA-256 of exact stored asset bytes, lowercase hexadecimal, 64 characters. |
| `asset_type` | enum | Yes | `image`, `diagram`, `page_crop`, or `other`. |
| `file_path` | string | Yes | Relative path from the documented data root to the stored binary. |
| `source_page` | integer | Yes | One-based PDF page index. |
| `bbox` | object or null | Yes | Optional source-page bounding box `{x0, y0, x1, y1}`; coordinate origin and units must be documented consistently. |
| `associated_record_ids` | array of strings | Yes | Known related chapter/question/block IDs; may be empty if association is unknown. |
| `review_status` | enum | Yes | `unreviewed`, `verified`, or `needs_review`. |

### 5.1 Asset rules

- Keep binaries separate from JSONL records.
- Compute SHA-256 from stored asset bytes and verify after transfer.
- Use paths relative to one documented data root, never machine-specific absolute paths.
- Retain source provenance even when semantic association is unknown.
- An image is not automatically associated with a question or chapter merely because it came from the same page.
- Exact duplicate bytes may be reported for deduplication. Do not automatically merge semantically similar but byte-different diagrams.
- For vector drawings or composite figures, document whether the stored asset is an extracted object, rendered page, or crop.
- For crops, record source page and bounding box when available.
- Dataset-level validation must resolve each `associated_record_ids` value against the correct record namespace.

## 6. JEE Main question manifest — `jee_questions.jsonl`

**Granularity:** one record per question. This is the canonical question-level representation, not the final retrieval chunk.

| Field | Type | Required | Description |
|---|---|---:|---|
| `question_id` | string | Yes | Stable, unique question record ID. |
| `source_id` | string | Yes | Must refer to a registered JEE Main source. |
| `year` | integer or null | Yes | Examination year where known; should agree with source metadata. |
| `session` | string or null | Yes | Session where known. |
| `shift` | string or null | Yes | Shift where known. |
| `subject` | string | Yes | Subject of the question. |
| `question_text` | string | Yes | Extracted question text; may be empty only for a failed record explicitly marked accordingly. |
| `options` | array | Yes | Array of option objects; may be empty if absent or extraction failed. |
| `correct_answer` | string or null | Yes | Answer supported by a source/answer key; never fabricated. |
| `solution_text` | string or null | Yes | Extracted solution where available; never fabricated. |
| `source_page` | integer | Yes | One-based PDF page index. |
| `asset_ids` | array of strings | Yes | Associated asset IDs, possibly empty. |
| `extraction_status` | enum | Yes | `success`, `needs_review`, or `failed`. |
| `difficulty` | enum or null | Yes | `easy`, `medium`, `hard`, or null when no defensible label is available. |
| `difficulty_source` | enum | Yes | `expert_review`, `empirical`, `model_estimate`, or `unknown`. |
| `difficulty_status` | enum | Yes | `provisional`, `reviewed`, `validated`, or `unknown`. |

All keys above must be present in serialized v1 records. Use JSON `null` for unknown nullable values.

### 6.1 `options` object

Each option has:

| Field | Type | Required | Description |
|---|---|---:|---|
| `label` | string | Yes | Original option label, e.g. `A`, `B`, `C`, `D`, or the source's actual label. |
| `text` | string | Yes | Extracted option text. Preserve notation and symbols. |

Preserve source order and labels. Do not assume every question has four options.

### 6.2 Difficulty-label policy

Difficulty labels support difficulty-aware retrieval, but a schema-valid label is not proof that the label is correct.

- `expert_review`: assigned or confirmed by a knowledgeable reviewer.
- `empirical`: estimated from student response data under a documented method. Record the method and dataset/run provenance in a linked review or evaluation artifact.
- `model_estimate`: produced by an automated model and not yet independently established.
- `unknown`: no defensible label is available.

Suggested meaning of `difficulty_status`:

- `provisional`: initial label, not yet reviewed.
- `reviewed`: inspected by a reviewer; not necessarily supported by empirical student outcomes.
- `validated`: passed a documented validation process with recorded criteria.
- `unknown`: no usable difficulty label.

When no defensible label exists, use `difficulty: null`, `difficulty_source: "unknown"`, and `difficulty_status: "unknown"`. Do not silently assign `medium` as a default. Model-estimated difficulty must not be represented as empirical difficulty.

### 6.3 Answer and provenance rules

- Never invent a correct answer, solution, exam year, session, shift, or difficulty label.
- Preserve answer-key provenance in source/review metadata or a linked review artifact when it is not evident from the source record alone.
- Partially extracted questions must be marked `needs_review` or `failed` under the agreed extraction rules and included in the quality report.
- Schema validation does not establish answer correctness.

## 7. IDs and deterministic reruns

IDs must be unique within their entity type and must not depend on machine-specific absolute paths.

- `source_id`: registered document version.
- `logical_source_id`: optional grouping for related versions.
- `chapter_id`: chapter in one source version/language.
- `question_id`: one question in one source document.
- `asset_id`: one extracted asset record.
- `block_id`: one block within a parent chapter; `(chapter_id, block_id)` must be unique.

Use a documented ID-generation method. IDs derived from stable source IDs and stable extraction keys are preferable to IDs based only on row order, timestamps, or absolute paths. If ordinal numbering is used, keep ordering deterministic and document what happens when records are inserted.

A rerun with identical source bytes, configuration, and pinned dependencies should produce equivalent canonical records. ID changes caused by extractor changes must be understood and reviewed, not silently accepted.

## 8. Pydantic validation and writing policy

Pydantic models live under `src/models/`, with one model module per record type (e.g. `source.py`, `ncert_chapter.py`, `asset.py`, `jee_question.py`). Use nested models for NCERT content blocks and JEE options.

### 8.1 Before writing JSONL

1. The extractor constructs a candidate record.
2. The corresponding Pydantic model validates it.
3. Invalid candidates are captured in a structured error/quarantine output with enough context to investigate; they are not silently dropped.
4. Validated records are serialized as one UTF-8 JSON object per line.
5. The pipeline records accepted, rejected, failed, and review-needed counts.

Pydantic validates structure, types, allowed values, and local invariants. It cannot establish semantic correctness or verify every external reference by itself.

### 8.2 Dataset-level validation

After record-level validation, a separate dataset validator checks:

- Unique IDs.
- Existence and source-type compatibility of every `source_id`.
- Existence of referenced `asset_id`, `chapter_id`, `question_id`, and `block_id` values in their correct namespaces.
- Existence and checksum integrity of asset files.
- Valid page ranges and block page references.
- Valid asset source-page references.
- Duplicate content/checksum reports.
- Counts of extraction failures, review-needed records, and quarantined records.

Distinguish fatal integrity errors from reviewable extraction uncertainty. Broken references, checksum mismatches, and duplicate IDs fail the relevant integrity gate. Uncertain extraction remains visible for review and need not invalidate unrelated records.

### 8.3 Safe output promotion

Write outputs to a temporary or run-specific location first. Validate the output and references, produce a quality report, and only then promote it to the canonical output location. This prevents a failed run from leaving a partial file that appears complete.

## 9. Data-quality report

Each extraction/validation run should produce a machine-readable report, for example `reports/data_quality/data_quality_report.json`, containing:

- Contract version and run identifier.
- Source files and checksums used.
- Pipeline/configuration version and dependency/environment reference.
- Record counts by entity type.
- Accepted, rejected, failed, quarantined, and review-needed counts.
- Schema errors and representative error details.
- Broken reference counts.
- Checksum mismatch counts.
- Duplicate ID/content counts.
- Validation outcome and failed gates.

Illustrative shape (counts are examples only):

```json
{
  "contract_version": "1.0.0",
  "run_id": "sample_run_001",
  "sources_seen": 12,
  "ncert_chapters": 150,
  "jee_questions": 420,
  "assets": 87,
  "schema_errors": 0,
  "broken_references": 0,
  "checksum_mismatches": 0,
  "records_needing_review": 8,
  "quarantined_records": 3
}
```

Define each metric consistently as a count of records, files, or errors. Do not mix units under the same metric name.

## 10. Versioning and change control

- This document becomes the source of truth for contract v1.0.0 once approved by the team.
- Changes to required fields, field meaning, nullability, enum values, or record granularity require review by all track owners.
- Do not change shared fields independently in one branch and expect other pipelines to adapt.
- Track schema/model changes in Git, updating sample records and tests in the same change.
- If a change breaks compatibility with existing records, version the contract explicitly or provide a documented migration.
- Include the contract version in pipeline reports and, where useful, DVC pipeline metadata.

## 11. Track ownership and integration

- **Track A — Contract and reproducibility:** owns this contract, ID/checksum conventions, Pydantic interfaces, common validators, DVC/reproducibility setup, and quality-report conventions.
- **Track B — NCERT extraction:** produces chapter records and ordered content blocks conforming to Section 4.
- **Track C — Asset extraction:** produces asset binaries and manifest records conforming to Section 5.
- **Track D — JEE Main extraction:** produces question records conforming to Section 6, including honest difficulty provenance/status.
- All tracks use the same source manifest and path conventions. Each track provides tests and a representative sample before processing the full corpus.

## 12. Decisions to confirm before freezing v1.0.0

Before treating this as frozen, the team should explicitly confirm:

1. Final ID-generation format and whether `logical_source_id` is needed immediately.
2. Exact `content_blocks` representation and supported block types after Member B tests representative PDFs.
3. Coordinate system and units for asset bounding boxes.
4. Allowed subject and language-code conventions.
5. Initial difficulty-label method and who may mark a label `validated`.
6. The precise definition of fatal integrity failures versus reviewable extraction uncertainty.

Until these are confirmed, treat this document as the v1.0.0 candidate contract, not permission for each track to invent its own interpretation.
