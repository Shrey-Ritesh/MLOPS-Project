import json
from pathlib import Path

path = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "ncert_chapters.jsonl"
)
if not path.is_file():
    raise FileNotFoundError(f"NCERT chapter records not found: {path}")

record_count = 0
figure_count = 0
linked_figure_count = 0
unlinked_figure_count = 0
for line_no, line in enumerate(path.open(encoding="utf-8"), start=1):
    if not line.strip():
        continue
    record = json.loads(line)
    record_count += 1
    chapter_assets = record.get("asset_ids", [])
    blocks = record.get("content_blocks", [])

    unlinked_figures = [
        block.get("block_id")
        for block in blocks
        if block.get("block_type") == "figure"
        and not block.get("asset_ids")
    ]
    figures = [
        block
        for block in blocks
        if block.get("block_type") == "figure"
    ]
    figure_count += len(figures)
    unlinked_figure_count += len(unlinked_figures)
    linked_figure_count += len(figures) - len(unlinked_figures)
    linked_blocks = [
        block.get("block_id")
        for block in blocks
        if block.get("asset_ids")
    ]

    if chapter_assets or linked_blocks or unlinked_figures:
        print({
            "line": line_no,
            "chapter_id": record.get("chapter_id"),
            "chapter_asset_ids": chapter_assets,
            "blocks_with_asset_ids": linked_blocks,
            "figures_without_asset_ids": unlinked_figures,
        })

print(
    f"Scanned {record_count} NCERT chapter records: "
    f"{figure_count} figures, {linked_figure_count} linked, "
    f"{unlinked_figure_count} unlinked."
)
if record_count == 0:
    print(f"No records found in {path}. Check the pipeline report and source PDFs.")