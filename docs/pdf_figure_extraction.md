# PDF figure extraction and asset management

Diagnostics report XObject metadata without assuming every object is a figure.
The extractor uses placed-image bounding boxes, joins associated soft masks,
and renders the composed page region as a fallback. Vector geometry groups
remain review candidates until explicitly selected for rendering. XObjects may
be masks, page backgrounds, or implementation details.

Run diagnostics from the repository root:

```powershell
python -m src.data.extract_images "path/to/document.pdf"
python -m src.data.extract_images "path/to/document.pdf" --json
```

The output reports one-based page numbers, page size in PDF points, extracted
text length, vector drawing rectangles, image XObject dimensions/formats and
soft-mask references, candidate geometric regions, and page render success.
Rendering is performed in memory at 36 DPI by default. `--dpi` changes this
diagnostic resolution. Page failures are included in the report and make the
command exit non-zero. The source PDF is never written.

`src.data.vector_regions.group_drawing_regions` accepts PyMuPDF drawing
objects and page dimensions. Coordinates use PyMuPDF page space (top-left
origin, points). Rectangles are clipped to page bounds; invalid and invisible
rectangles are reported with their original zero-based drawing indices.
Rectangles are joined when they overlap or their nearest edges are within
`distance_fraction × min(page_width, page_height)`. The conservative default
`distance_fraction=0.006` is configurable. A sweep over x coordinates limits
pair checks to nearby candidates. Results are sorted deterministically and
include each candidate region's bounds, drawing indices, and object count.

The output is intentionally geometric evidence only. It does not filter by
drawing count, infer semantic content, or link a region to a question/chapter.
To extract placed raster images and list vector candidates for a registered
source, run:

```powershell
python -m src.data.extract_pdf_assets "path/to/document.pdf" --source-id src-<sha256>
```

After reviewing a candidate, render a selected region using one-based page
number and PDF point coordinates:

```powershell
python -m src.data.extract_pdf_assets "path/to/document.pdf" --source-id src-<sha256> --region 4:120,180,360,420
```

Repeat `--region` to render multiple boxes. Only provide
`--associate-record-id` when the question or section association has been
confirmed. Otherwise the asset is preserved with an empty association list.

Stored files use SHA-256 content-addressed paths under
`data/assets/sha256/<prefix>/<digest>.<ext>`. Exact stored bytes share one
binary file; separate occurrence rows retain source page and crop provenance.
`assets_manifest.jsonl` is incrementally updated, and unreviewed assets and
unclassified vector regions appear in `assets_review.jsonl`.

Validate manifest rows, file existence/readability, checksums, source IDs, and
optional record links with:

```powershell
python -m src.data.validate_assets_manifest data/processed/assets_manifest.jsonl \
  --source-manifest data/processed/source_manifest.jsonl \
  --source-manifest data/interim/ncert_source_manifest.jsonl \
  --records data/processed/jee_questions.jsonl \
  --records data/processed/ncert_chapters.jsonl
```

Validation refreshes the review list and exits non-zero when an integrity
check fails. If a complete set of record manifests is supplied and some
legacy links point to missing records, add
`--clear-unresolved-associations` to remove only those dangling links and mark
the affected assets `needs_review`; source/page/crop provenance remains intact.
