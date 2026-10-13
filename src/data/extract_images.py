"""Read-only PDF diagnostics for image and vector figure investigation.

Run from the repository root::

    python -m src.data.extract_images path/to/document.pdf [--json]

This utility inspects image metadata and renderability. It does not extract
image XObjects or modify the source PDF; XObjects may be masks/backgrounds.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pymupdf

from src.data.vector_regions import group_drawing_regions


def _rect_values(rect: Any) -> list[float] | None:
    values = [float(value) for value in rect]
    return values if all(math.isfinite(value) for value in values) else None


def inspect_pdf(pdf_path: str | Path, *, dpi: int = 36) -> dict[str, Any]:
    """Inspect each page without writing to or changing the PDF.

    Page-level failures are included in the returned report and do not hide
    diagnostics for later pages. Opening an invalid/unreadable PDF raises a
    ``ValueError`` with the path and underlying PyMuPDF error.
    """
    path = Path(pdf_path)
    if dpi < 1:
        raise ValueError("dpi must be a positive integer")
    try:
        document = pymupdf.open(path)
    except Exception as exc:
        raise ValueError(f"Cannot open PDF {path}: {exc}") from exc

    pages: list[dict[str, Any]] = []
    try:
        for page_index in range(document.page_count):
            page_number = page_index + 1
            item: dict[str, Any] = {
                "page_number": page_number,
                "page_size": None,
                "text_length": None,
                "vector_drawing_count": None,
                "image_xobject_count": None,
                "images": [],
                "drawing_bboxes": [],
                "render": {"success": False, "error": None},
                "errors": [],
            }
            try:
                page = document.load_page(page_index)
                item["page_size"] = {
                    "width": float(page.rect.width),
                    "height": float(page.rect.height),
                    "units": "PDF points",
                    "origin": "top-left (PyMuPDF page coordinates)",
                }
                item["text_length"] = len(page.get_text("text"))

                drawings = page.get_drawings()
                item["vector_drawing_count"] = len(drawings)
                for index, drawing in enumerate(drawings):
                    bbox = _rect_values(drawing.get("rect", ()))
                    item["drawing_bboxes"].append({"drawing_index": index, "bbox": bbox})
                item["drawing_regions"] = group_drawing_regions(
                    drawings,
                    page_width=page.rect.width,
                    page_height=page.rect.height,
                )

                image_refs = page.get_images(full=True)
                item["image_xobject_count"] = len(image_refs)
                for ref in image_refs:
                    xref = int(ref[0])
                    smask = int(ref[1]) if len(ref) > 1 else 0
                    image: dict[str, Any] = {
                        "xref": xref,
                        "soft_mask_xref": smask or None,
                        "width": int(ref[2]) if len(ref) > 2 else None,
                        "height": int(ref[3]) if len(ref) > 3 else None,
                        "bits_per_component": int(ref[4]) if len(ref) > 4 else None,
                        "colorspace": ref[5] if len(ref) > 5 else None,
                        "format": None,
                        "metadata_error": None,
                    }
                    try:
                        extracted = document.extract_image(xref)
                        image["format"] = extracted.get("ext")
                        image["width"] = extracted.get("width", image["width"])
                        image["height"] = extracted.get("height", image["height"])
                    except Exception as exc:
                        image["metadata_error"] = str(exc)
                    item["images"].append(image)

                try:
                    # Rendering validates the page display path; the temporary
                    # pixmap is kept in memory and never saved.
                    page.get_pixmap(dpi=dpi, alpha=False)
                    item["render"] = {"success": True, "error": None}
                except Exception as exc:
                    item["render"] = {"success": False, "error": str(exc)}
                    item["errors"].append(f"render failed: {exc}")
            except Exception as exc:
                item["errors"].append(f"page inspection failed: {exc}")
            pages.append(item)
    finally:
        document.close()

    return {"pdf_path": str(path), "page_count": len(pages), "pages": pages}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="PDF file to inspect (read-only)")
    parser.add_argument("--dpi", type=int, default=36, help="diagnostic render resolution (default: 36)")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()
    try:
        report = inspect_pdf(args.pdf, dpi=args.dpi)
    except ValueError as exc:
        parser.error(str(exc))
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"PDF: {report['pdf_path']} ({report['page_count']} pages)")
        for page in report["pages"]:
            size = page["page_size"]
            dimensions = "unavailable" if size is None else f"{size['width']:.1f} × {size['height']:.1f} pt"
            print(
                f"Page {page['page_number']}: {dimensions}; text_chars={page['text_length']}; "
                f"drawings={page['vector_drawing_count']}; images={page['image_xobject_count']}; "
                f"render={'ok' if page['render']['success'] else 'FAILED'}"
            )
            for image in page["images"]:
                print(
                    f"  image xref={image['xref']} {image['width']}×{image['height']} "
                    f"format={image['format'] or 'unknown'} soft_mask={image['soft_mask_xref']}"
                )
            for error in page["errors"]:
                print(f"  ERROR: {error}")
    if any(not p["render"]["success"] or p["errors"] for p in report["pages"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
